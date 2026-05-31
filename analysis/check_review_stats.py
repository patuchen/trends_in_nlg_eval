import os
import re
import sys
import numpy as np
import pandas as pd
from scipy import stats

# Path setups
analysis_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, analysis_dir)
from helpers import load_results, SIGGEN_VENUES

RESULTS_CSV = os.path.join(analysis_dir, "../output/results.csv")
RESULTS_JSONL = os.path.join(analysis_dir, "../output/results.jsonl")
RESCUED_CSV = os.path.join(analysis_dir, "../output/rescued_results.csv")

print("Loading data...")
df, _ = load_results(RESULTS_JSONL, RESULTS_CSV)
df = df[df["year"] < 2026].copy()
NON_ENGLISH_VENUES = {"jep", "recital", "tal", "taln", "rocling", "ijclclp", "ccl"}
df = df[~df["venue"].isin(NON_ENGLISH_VENUES)].copy()

# Add venue group
def get_venue_group(venue):
    v = str(venue).lower().strip()
    if v in SIGGEN_VENUES:
        return "generation"
    if v in ["acl", "emnlp", "naacl", "eacl", "aacl", "ijcnlp"]:
        return "core_nlp"
    if v in ["tacl", "cl"]:
        return "journals"
    return "other"

df["venue_group"] = df["venue"].apply(get_venue_group)

# Incorporate overrides (simplifying logic from rq_analysis.py)
_flt_counts = pd.read_csv(
    os.path.join(analysis_dir, "../output/filtered_papers.csv"),
    usecols=lambda c: c in {"id", "autoeval_count", "humeval_count"},
    encoding="utf-8", on_bad_lines="skip", low_memory=False,
).rename(columns={"id": "paper_id"})

_flt_counts["autoeval_count"] = pd.to_numeric(_flt_counts["autoeval_count"], errors="coerce").fillna(0)
_flt_counts["humeval_count"] = pd.to_numeric(_flt_counts["humeval_count"], errors="coerce").fillna(0)

df = df.merge(_flt_counts, on="paper_id", how="left")
df["autoeval_count"] = df["autoeval_count"].fillna(0)
df["humeval_count"] = df["humeval_count"].fillna(0)

df.loc[df["autoeval_count"] >= 2, "has_auto_eval"] = True

# Read rescued results if available
if os.path.exists(RESCUED_CSV):
    rescued = pd.read_csv(RESCUED_CSV, encoding="utf-8", on_bad_lines="skip")
    bool_cols = ["has_human_eval", "has_auto_eval", "has_llm_judge"]
    for col in bool_cols:
        if col in rescued.columns:
            rescued[col] = rescued[col].astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if "venue" not in rescued.columns or rescued["venue"].isna().all():
        from helpers import _extract_venue
        rescued["venue"] = rescued["paper_id"].apply(_extract_venue)
    rescued = rescued[rescued["year"] < 2026]
    rescued = rescued[~rescued["venue"].isin(NON_ENGLISH_VENUES)]
    rescued = rescued[~rescued["paper_id"].isin(df["paper_id"])]
    df = pd.concat([df, rescued], ignore_index=True, sort=False)

df["venue_group"] = df["venue"].apply(get_venue_group)

print(f"Total processed papers: {len(df):,}")

# --- 1. INLG analysis ---
print("\n=== 1. INLG Series Analysis ===")
inlg_df = df[df["venue"].astype(str).str.lower() == "inlg"].copy()
print(f"Total INLG papers: {len(inlg_df)}")
if len(inlg_df) > 0:
    he_inlg_rate = inlg_df["has_human_eval"].mean()
    auto_inlg_rate = inlg_df["has_auto_eval"].mean()
    llm_inlg_rate = inlg_df["has_llm_judge"].mean()
    print(f"Overall rates in INLG:")
    print(f"  Human Evaluation: {he_inlg_rate:.1%}")
    print(f"  Auto Evaluation: {auto_inlg_rate:.1%}")
    print(f"  LLM Judge: {llm_inlg_rate:.1%}")
    
    # Recent INLG trends
    print("\nINLG recent years (2018-2025):")
    inlg_recent = inlg_df[inlg_df["year"].between(2018, 2025)]
    inlg_yr = inlg_recent.groupby("year")["has_human_eval"].agg(["sum", "count"])
    inlg_yr["rate"] = inlg_yr["sum"] / inlg_yr["count"]
    print(inlg_yr)
    
    # Compare with non-INLG generation venues
    gen_other_df = df[(df["venue_group"] == "generation") & (df["venue"].astype(str).str.lower() != "inlg")]
    print(f"\nOther generation venues (n={len(gen_other_df)}): HE rate={gen_other_df['has_human_eval'].mean():.1%}")
    core_nlp_df = df[df["venue_group"] == "core_nlp"]
    print(f"Core NLP venues (n={len(core_nlp_df)}): HE rate={core_nlp_df['has_human_eval'].mean():.1%}")

# --- 2. LLM Judge vs Output Length / Annotation Cost ---
print("\n=== 2. LLM-as-a-judge by Task (Length/Cost proxy) ===")
# Explode tasks
df["task_list"] = df["inferred_tasks"].fillna("").str.split("|")
df_tasks = df.explode("task_list").rename(columns={"task_list": "task"})
df_tasks = df_tasks[df_tasks["task"].str.strip() != ""].copy()
df_tasks["task"] = df_tasks["task"].str.strip().str.lower()

# Tasks classification by length/cost
# High: Code Gen, Dialogue, Summarization, Story Gen (General NLG)
# Low: MT, Paraphrase, Captioning, Data-to-text, Question Gen
high_cost_tasks = {"code_generation", "dialogue", "summarization", "general_nlg"}
low_cost_tasks = {"machine_translation", "paraphrase", "captioning", "data_to_text", "question_generation"}

task_stats = []
for task in df_tasks["task"].unique():
    sub = df_tasks[df_tasks["task"] == task]
    n = len(sub)
    if n < 10:
        continue
    llm_j = sub["has_llm_judge"].sum()
    rate = llm_j / n
    he = sub["has_human_eval"].sum()
    he_rate = he / n
    
    cost_group = "Unknown"
    if task in high_cost_tasks:
        cost_group = "High (Long output / Complex annotation)"
    elif task in low_cost_tasks:
        cost_group = "Low (Short output / Simpler annotation)"
        
    task_stats.append({
        "task": task,
        "n": n,
        "llm_judge_count": llm_j,
        "llm_judge_rate": rate,
        "he_rate": he_rate,
        "cost_group": cost_group
    })

task_stats_df = pd.DataFrame(task_stats).sort_values("llm_judge_rate", ascending=False)
print(task_stats_df.to_string(index=False))

# Group level comparison
high_sub = task_stats_df[task_stats_df["cost_group"].str.startswith("High")]
low_sub = task_stats_df[task_stats_df["cost_group"].str.startswith("Low")]
print(f"\nAverage LLM-as-a-judge rate in High-Cost/Long-Output tasks: {high_sub['llm_judge_count'].sum() / high_sub['n'].sum():.2%}")
print(f"Average LLM-as-a-judge rate in Low-Cost/Short-Output tasks: {low_sub['llm_judge_count'].sum() / low_sub['n'].sum():.2%}")

# Chi-square test on High vs Low cost LLM judge counts
high_n = high_sub['n'].sum()
high_llm = high_sub['llm_judge_count'].sum()
low_n = low_sub['n'].sum()
low_llm = low_sub['llm_judge_count'].sum()
contingency_table = [
    [high_llm, high_n - high_llm],
    [low_llm, low_n - low_llm]
]
chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)
print(f"Chi-square test comparing LLM-judge prevalence: chi2={chi2:.4f}, p-value={p_val:.4e}")

# Try statsmodels first
try:
    import statsmodels.formula.api as smf
    print("Running statsmodels logistic regression...")
    
    # 1. Human eval trend (overall, last decade 2015-2025)
    df_recent = df[df["year"].between(2015, 2025)].copy()
    df_recent["has_human_eval"] = df_recent["has_human_eval"].astype(int)
    m_he = smf.logit("has_human_eval ~ year", data=df_recent).fit(disp=False)
    print(f"HE Trend (2015-2025): coeff={m_he.params['year']:.4f}, p={m_he.pvalues['year']:.4e}")
    
    # 2. LLM judge trend (2020-2025)
    df_llm = df[df["year"].between(2020, 2025)].copy()
    df_llm["has_llm_judge"] = df_llm["has_llm_judge"].astype(int)
    m_llm = smf.logit("has_llm_judge ~ year", data=df_llm).fit(disp=False)
    print(f"LLM-judge Trend (2020-2025): coeff={m_llm.params['year']:.4f}, p={m_llm.pvalues['year']:.4e}")
    
    # 3. Faithfulness trend (among HE papers, 2015-2025)
    from rq_analysis import _read_paper_text, CRITERIA_RES
    f_regex = CRITERIA_RES["faithfulness"]
    
    df_he = df[df["has_human_eval"] & df["year"].between(2015, 2025)].copy()
    print(f"Extracting faithfulness for {len(df_he)} recent HE papers...")
    
    has_f = []
    for idx, row in df_he.iterrows():
        text = _read_paper_text(row["paper_id"])
        # Fallback to criteria_norm if full text missing
        if text:
            hit = bool(f_regex.search(text))
        else:
            norm_crit = str(row.get("criteria_norm") or "")
            hit = "faithfulness" in norm_crit.split("|")
        has_f.append(int(hit))
        
    df_he["has_faithfulness"] = has_f
    m_faith = smf.logit("has_faithfulness ~ year", data=df_he).fit(disp=False)
    print(f"Faithfulness Trend (2015-2025, HE only): coeff={m_faith.params['year']:.4f}, p={m_faith.pvalues['year']:.4e}")

except Exception as e:
    print(f"statsmodels logit failed or not installed ({e}). Running scipy linregress on proportions...")
    # Fallback to simple linear regression on proportions
    df_recent = df[df["year"].between(2015, 2025)]
    he_props = df_recent.groupby("year")["has_human_eval"].mean()
    res = stats.linregress(he_props.index, he_props.values)
    print(f"HE proportion linear trend slope={res.slope:.4f}, p={res.pvalue:.4e}")
    
    df_llm = df[df["year"].between(2020, 2025)]
    llm_props = df_llm.groupby("year")["has_llm_judge"].mean()
    res_llm = stats.linregress(llm_props.index, llm_props.values)
    print(f"LLM-judge proportion linear trend slope={res_llm.slope:.4f}, p={res_llm.pvalue:.4e}")

