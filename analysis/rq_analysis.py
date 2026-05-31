"""
NLG Evaluation Trends -- Research Question Analysis
Run from the analysis/ directory:  python rq_analysis.py

RQ1  Prevalence and trajectory of human evaluation over time and across venues
RQ2  LLM-as-a-judge adoption: instruction-LLM vs. ML-model-as-evaluator
RQ3  Human evaluation criteria: which have risen, stalled, or faded?
RQ4  Evaluation modality mix: how often is human evaluation skipped entirely?
RQ5  Task-based differences in evaluation practice
RQ6  Multilingual and non-English evaluation practices
RQ7  Scale and rigour of human evaluation studies (annotator counts, agreement)
"""

import os, re, sys, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

_stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(_stdout_reconfigure):
    _stdout_reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from helpers import (
    load_results, classify_language, apply_norm,
    dual_axis_trend, wilson_ci, save_fig, SIGGEN_VENUES, FIGURES_DIR,
)

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "figure.figsize": (10, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _style_axes(ax, *, y_grid=True):
    """Paper-style aesthetics: no top/right spines, no vertical gridlines."""
    if y_grid:
        ax.grid(True, axis="y", alpha=0.35)
    else:
        ax.grid(False)
    ax.grid(False, axis="x")
    for spine in ("top", "right"):
        if spine in ax.spines:
            ax.spines[spine].set_visible(False)


def _style_year_axis(ax, *, step=5, rotate=45, fontsize=8):
    """Use every Nth year as a major tick to avoid label overlap."""
    ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(rotate)
        lab.set_ha("right")
        lab.set_fontsize(fontsize)

RESULTS_CSV         = "../output/results.csv"
RESULTS_JSONL       = "../output/results.jsonl"
RESCUED_CSV         = "../output/rescued_results.csv"   # from rescue_pipeline.py
PAPERS_CSV          = "../papers.csv"
os.makedirs("figures", exist_ok=True)
os.makedirs("tables",  exist_ok=True)

# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------
print("Loading data...")
df, _ = load_results(RESULTS_JSONL, RESULTS_CSV)

# Year filter: exclude partial 2026 year
df = df[df["year"] < 2026].copy()

# Non-English venue exclusion (French and Chinese venues, unreliable extraction)
NON_ENGLISH_VENUES = {"jep", "recital", "tal", "taln", "rocling", "ijclclp", "ccl"}
_before = len(df)
df = df[~df["venue"].isin(NON_ENGLISH_VENUES)].copy()
print(f"Dropped {_before - len(df)} non-English venue papers. Remaining: {len(df):,}")

# ---------------------------------------------------------------------------
# Shared utilities: full-text reading + keyword detection dictionaries
# ---------------------------------------------------------------------------

TXT_DIR = Path("../txt_papers")

_REFS_RE = re.compile(
    r'\n(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY|'
    r'Appendix|APPENDIX|Appendices|APPENDICES)\n',
)

def _read_paper_text(paper_id):
    """Return main body text (before References section), or '' if unavailable."""
    p = TXT_DIR / f"{paper_id}.pdf"
    if p.exists() and p.stat().st_size > 0:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            m = _REFS_RE.search(text)
            return text[:m.start()] if m else text
        except Exception:
            return ""
    return ""

# Annotation METHOD patterns (what type of elicitation was used)
METHOD_RES = {
    "likert": re.compile(
        r'\blikert\b'
        r'|\d[\s-]?point\s+(scale|rating|likert)'
        r'|scale\s+of\s+\d[^\.]*?to\s+\d'
        r'|\d[\s-]to[\s-]\d\s+scale'
        r'|\b\d(?:\.\d+)?\s*/\s*[57]\b',
        re.IGNORECASE,
    ),
    "ranking": re.compile(
        r'\branking\s+(evaluat|annot|study)'
        r'|human\s+ranking'
        r'|rank\s+(order\s+)?(the\s+output|outputs|the\s+response|the\s+translation|the\s+summar)'
        r'|relative\s+ranking\s+of'
        r'|pairwise\s+(comparison|preference|evaluat|judg)'
        r'|side[\s-]by[\s-]side'
        r'|forced[\s-]choice'
        r'|\bA/B\s+test(ing)?'
        r'|preference\s+(study|test|judg)',
        re.IGNORECASE,
    ),
    "categories": re.compile(
        r'binary\s+(annotation|evaluat|judg|rating|choice)'
        r'|yes[\s/]no\s+(evaluat|judg|rating)'
        r'|acceptability\s+judg'
        r'|fluent\s+or\s+not'
        r'|\bcategories:\s+[A-Z]'
        r'|annotat\w*.{0,50}\bcategor\w*'
        r'|annotat\w*\s+(?:in|into|as|with)\s+["\u201c\u2018]',
        re.IGNORECASE,
    ),
    "span": re.compile(
        r'span\s+annotation'
        r'|error\s+(span|mark|annot)'
        r'|span[\s-]?(level|based)\s+(evaluat|annot)'
        r'|error\s+(highlight|tag)'
        r'|\bMQM\b|Error\s+Span\s+Annotation',
        re.IGNORECASE,
    ),
    "best_worst_scaling": re.compile(
        r'best[\s-]worst\s+scal'
        r'|\bBWS\b',
        re.IGNORECASE,
    ),
    "direct_assessment": re.compile(
        r'direct\s+assessment'
        r'|\bDA\b\s+(evaluat|score|protocol|rating)'
        r'|adequacy[\s/]fluency\s+rating',
        re.IGNORECASE,
    ),
    "post_editing": re.compile(
        r'post[\s-]?edit(ing|ed)?'
        r'|postedit(ing|ed)?'
        r'|human[\s-]?post[\s-]?edit(ing|ed)?'
        r'|\bHTER\b',
        re.IGNORECASE,
    ),
}

# has_auto_eval and has_human_eval override: LLM only sees extracted section (may be truncated).
# When regex found >= 2 autoeval signal matches, override a False to True.
# For humeval, override False to True if humeval_count > 2 AND METHOD_RES matches full text.
AUTOEVAL_OVERRIDE_THRESHOLD = 2
HUMEVAL_OVERRIDE_THRESHOLD = 2

_flt_counts = pd.read_csv(
    "../output/filtered_papers.csv",
    usecols=lambda c: c in {"id", "autoeval_count", "humeval_count"},
    encoding="utf-8", on_bad_lines="skip", low_memory=False,
).rename(columns={"id": "paper_id"})

_flt_counts["autoeval_count"] = pd.to_numeric(
    _flt_counts["autoeval_count"], errors="coerce"
).fillna(0)
_flt_counts["humeval_count"] = pd.to_numeric(
    _flt_counts["humeval_count"], errors="coerce"
).fillna(0)

df = df.merge(_flt_counts, on="paper_id", how="left")
df["autoeval_count"] = df["autoeval_count"].fillna(0)
df["humeval_count"] = df["humeval_count"].fillna(0)

df["has_auto_eval_original"] = df["has_auto_eval"].copy()
_auto_override = (~df["has_auto_eval"]) & (df["autoeval_count"] >= AUTOEVAL_OVERRIDE_THRESHOLD)
df.loc[_auto_override, "has_auto_eval"] = True
print(f"has_auto_eval override applied: {_auto_override.sum()} papers flipped to True")

df["has_human_eval_original"] = df["has_human_eval"].copy()
_hum_candidates = df[(~df["has_human_eval"]) & (df["humeval_count"] > HUMEVAL_OVERRIDE_THRESHOLD)]
_hum_override_count = 0
for idx, row in _hum_candidates.iterrows():
    text = _read_paper_text(row["paper_id"])
    if not text:
        continue
    if any(r.search(text) for r in METHOD_RES.values()):
        df.at[idx, "has_human_eval"] = True
        _hum_override_count += 1
print(f"has_human_eval override applied: {_hum_override_count} papers flipped to True")

# Merge in rescued papers (regex-only, no LLM) -- done AFTER override so their
# autoeval_count column doesn't conflict with the filtered_papers.csv merge above.
if os.path.exists(RESCUED_CSV):
    rescued = pd.read_csv(RESCUED_CSV, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    bool_cols = ["has_human_eval", "has_auto_eval", "has_llm_judge",
                 "parse_failed", "agreement_reported", "monolingual",
                 "haiku_spot_checked", "haiku_correction"]
    for col in bool_cols:
        if col in rescued.columns:
            rescued[col] = (
                rescued[col].astype(str).str.strip().str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )
    if "venue" not in rescued.columns or rescued["venue"].isna().all():
        from helpers import _extract_venue
        rescued["venue"] = rescued["paper_id"].apply(_extract_venue)
    rescued = rescued[rescued["year"] < 2026]
    rescued = rescued[~rescued["venue"].isin(NON_ENGLISH_VENUES)]
    # autoeval_count already correct for rescued papers; mark original same as final
    rescued["has_auto_eval_original"] = rescued["has_auto_eval"]
    rescued["has_human_eval_original"] = rescued["has_human_eval"]
    # Exclude paper IDs already in main df to avoid duplicates
    rescued = rescued[~rescued["paper_id"].isin(df["paper_id"])]
    df = pd.concat([df, rescued], ignore_index=True, sort=False)
    print(f"Merged {len(rescued):,} rescued papers. Total corpus: {len(df):,}")

# Venue groups
VENUE_GROUPS = {
    "generation": sorted(SIGGEN_VENUES),
    "core_nlp":   ["acl", "emnlp", "naacl", "eacl", "aacl", "ijcnlp"],
    "journals":   ["tacl", "cl"],
}

def get_venue_group(venue):
    v = str(venue).lower().strip()
    for grp, members in VENUE_GROUPS.items():
        if v in members:
            return grp
    return "other"

df["venue_group"] = df["venue"].apply(get_venue_group)

# Language classification
df["lang_group"] = df.apply(classify_language, axis=1)

# Task lists
df["task_list"]     = df["inferred_tasks"].fillna("").str.split("|")
df["num_tasks"]     = df["task_list"].apply(len)
df["is_multi_task"] = df["num_tasks"] > 1
df_tasks  = df.explode("task_list").rename(columns={"task_list": "task"})
df_tasks  = df_tasks[df_tasks["task"].str.strip() != ""].copy()
df_single = df[~df["is_multi_task"]].explode("task_list").rename(columns={"task_list": "task"})
df_single = df_single[df_single["task"].str.strip() != ""].copy()
df_single["task"] = df_single["task"].str.strip().str.lower()
df_tasks["task"]  = df_tasks["task"].str.strip().str.lower()

# LLM judge type
_INSTRUCTION_RE = re.compile(
    r"gpt-4|gpt4|gpt-3[.]5|gpt3[.]5|chatgpt|text-davinci-00[23]"
    r"|claude-?[1-9]|claude [1-9]|gemini|palm-?2|bard|command-?r"
    r"|llama-?[23]|llama [23]|vicuna|alpaca|mistral|falcon|instructgpt",
    re.IGNORECASE,
)
_ML_JUDGE_RE = re.compile(
    r"\bgpt-?2\b|discriminator|adversarial.*classif|\bclip\b"
    r"|roberta.*classif|bert.*classif|infersent|reward[- ]model"
    r"|perplexit|nli[- ]model|trained[- ]classif",
    re.IGNORECASE,
)

def classify_llm_judge_type(s):
    if pd.isna(s) or not str(s).strip():
        return "unknown"
    if _INSTRUCTION_RE.search(str(s)):
        return "instruction_llm"
    if _ML_JUDGE_RE.search(str(s)):
        return "ml_evaluator"
    return "unknown"

def _norm_manual_judge_type(v):
    """Normalise manual labels to {instruction_llm, ml_evaluator} or None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if not s or s in {"na", "n/a", "none", "?"}:
        return None
    if s in {"ml", "ml_evaluator", "mlevaluator"}:
        return "ml_evaluator"
    if s in {"instruct", "instruction", "instruction_llm", "instruction-llm"}:
        return "instruction_llm"
    if s in {"unknown", "unk"}:
        return None
    return None

# Keep the original auto classification for reproducibility/curation.
df["llm_judge_type_auto"] = df["llm_judge_model"].apply(classify_llm_judge_type)

# Working type used for analysis.
df["llm_judge_type"] = df["llm_judge_type_auto"]

# Optional manual overrides (paper-level) from the curation CSV.
manual_judge_path = "tables/rq2_unknown_llm_judge_papers.csv"
if os.path.exists(manual_judge_path):
    try:
        manual_tbl = pd.read_csv(manual_judge_path)
        manual_type_col = None
        if "type" in manual_tbl.columns:
            manual_type_col = "type"
        elif "llm_judge_type" in manual_tbl.columns:
            manual_type_col = "llm_judge_type"
        if manual_type_col and "paper_id" in manual_tbl.columns:
            manual_map = (
                manual_tbl[["paper_id", manual_type_col]]
                .dropna(subset=["paper_id"])
                .assign(_manual_type=lambda x: x[manual_type_col].apply(_norm_manual_judge_type))
                .dropna(subset=["_manual_type"])
                .set_index("paper_id")["_manual_type"]
                .to_dict()
            )
            if manual_map:
                manual_series = df["paper_id"].map(manual_map)
                df.loc[manual_series.notna(), "llm_judge_type"] = manual_series[manual_series.notna()]
    except Exception:
        pass

# Policy: treat any remaining unknown LLM-judge types as instruction-tuned.
df.loc[df["has_llm_judge"] & (df["llm_judge_type"] == "unknown"), "llm_judge_type"] = "instruction_llm"

df["has_true_llm_judge"] = df["has_llm_judge"] & (df["llm_judge_type"] == "instruction_llm")
df["has_ml_judge"]       = df["has_llm_judge"] & (df["llm_judge_type"] == "ml_evaluator")

for col in ("llm_judge_type_auto", "llm_judge_type", "has_true_llm_judge", "has_ml_judge"):
    _m = df.set_index("paper_id")[col]
    df_tasks[col]  = df_tasks["paper_id"].map(_m)
    df_single[col] = df_single["paper_id"].map(_m)

# Normalised criteria
CRITERIA_NORM = {
    "fluency": "fluency",           "flu": "fluency",
    "adequacy": "adequacy",         "adq": "adequacy",
    "relevance": "relevance",
    "faithfulness": "faithfulness", "factual consistency": "faithfulness",
    "factuality": "faithfulness",   "factual": "faithfulness",
    "coherence": "coherence",
    "grammaticality": "grammaticality", "grammatical": "grammaticality",
    "naturalness": "naturalness",
    "informativeness": "informativeness",
    "accuracy": "accuracy",         "correctness": "accuracy",
    "consistency": "consistency",
    "readability": "readability",
    "quality": "overall_quality",   "overall quality": "overall_quality",
    "overall": "overall_quality",
    "simplicity": "simplicity",
    "meaning preservation": "meaning_preservation",
}
df["criteria_norm"] = apply_norm(df["human_eval_criteria"], CRITERIA_NORM)
df_single["criteria_norm"] = df_single["paper_id"].map(df.set_index("paper_id")["criteria_norm"])

print(f"Analysis corpus: {len(df):,} papers  ({df['year'].min():.0f}-{df['year'].max():.0f})")
print(f"  with human eval  : {df['has_human_eval'].sum():,}")
print(f"  with auto eval   : {df['has_auto_eval'].sum():,}")
print(f"  with LLM judge   : {df['has_llm_judge'].sum():,}  "
      f"(instruction: {df['has_true_llm_judge'].sum()}, ml: {df['has_ml_judge'].sum()})")

SEP = "=" * 70


# Quality CRITERIA patterns (what dimension of output quality is assessed).
# Noun/nominal forms are preferred to reduce false positives over adjective forms.
# Informed by Howcroft et al. (2020) taxonomy and LLM-extracted values in this corpus.
# Notes on potential false-positive risk are included per criterion.
CRITERIA_RES = {
    # --- Core criteria: high specificity, stable evaluation terminology ---
    "fluency": re.compile(
        # Howcroft: "degree to which text flows well"
        # Appears as-is in papers; one of the most common criterion names.
        r'\bfluency\b',
        re.IGNORECASE,
    ),
    "adequacy": re.compile(
        # MT-specific: how much meaning is preserved from source.
        # Highly specific term, very low false-positive risk.
        r'\badequacy\b',
        re.IGNORECASE,
    ),
    "relevance": re.compile(
        # Common in summarisation / QA evaluation.
        # Some false-positive risk ("relevance of our approach"),
        # mitigated by restricting to HE papers (has_human_eval=True).
        r'\brelevance\b',
        re.IGNORECASE,
    ),
    "faithfulness": re.compile(
        # Howcroft: "Correctness of outputs relative to external frame of reference"
        # Covers: faithfulness, factual consistency, factuality, hallucination,
        # fidelity (common synonym in older papers), veracity.
        # "hallucination" is a strong signal for hallucination-focused HE.
        r'\bfaithfulness\b'
        r'|\bfactual[\s-]consistency\b'
        r'|\bfactuality\b'
        r'|\bfidelity\b'
        r'|\bveracity\b',
        re.IGNORECASE,
    ),
    "coherence": re.compile(
        # Howcroft: "degree to which content is presented in well-structured,
        # logical and meaningful way".  Noun form only.
        r'\bcoherence\b',
        re.IGNORECASE,
    ),
    "grammaticality": re.compile(
        # Howcroft: "degree to which output is free of grammatical errors".
        # Compound forms added to catch "grammatical correctness", "grammatical error".
        r'\bgrammaticality\b'
        r'|\bgrammatical\s+(error|correctness|quality|acceptability)\b',
        re.IGNORECASE,
    ),
    "naturalness": re.compile(
        # Howcroft: "degree to which output is likely to be used by native speaker".
        # Noun form only; "natural" excluded (too common in other contexts).
        r'\bnaturalness\b',
        re.IGNORECASE,
    ),
    "informativeness": re.compile(
        # Howcroft: "Information content of outputs".
        # "information content" added as a synonym phrase.
        r'\binformativeness\b'
        r'|\binformation\s+content\b',
        re.IGNORECASE,
    ),
    # --- Moderately specific criteria: some false-positive risk ---
    "accuracy": re.compile(
        # Howcroft: "Correctness of outputs in their own right (content)".
        # Appears in papers as factual accuracy or as a general metric name;
        # false positives likely for "model accuracy", "classification accuracy".
        r'\baccuracy\b',
        re.IGNORECASE,
    ),
    "consistency": re.compile(
        # Includes factual consistency (= faithfulness in many papers)
        # and self-consistency across turns.  False positives possible.
        r'\bconsistency\b',
        re.IGNORECASE,
    ),
    "readability": re.compile(
        # Howcroft: "degree to which output is easy to read".
        # Noun form; fairly specific in NLG evaluation context.
        r'\breadability\b',
        re.IGNORECASE,
    ),
    "overall_quality": re.compile(
        # Howcroft: "Quality of outputs" (maximally underspecified criterion).
        # Compound forms only to reduce false positives.
        r'\boverall[\s-]quality\b'
        r'|\boverall[\s-]score\b',
        re.IGNORECASE,
    ),
    "simplicity": re.compile(
        # Howcroft: "Text Property [Complexity/simplicity]".
        # Mainly text simplification evaluation; fairly specific.
        r'\bsimplicity\b',
        re.IGNORECASE,
    ),
    "meaning_preservation": re.compile(
        # Howcroft: "Correctness of outputs relative to input (content)".
        # Very specific compound; low false-positive risk.
        r'\bmeaning[\s-]preserv(ation|ing)\b',
        re.IGNORECASE,
    ),
    "correctness": re.compile(
        # Howcroft: "Correctness of outputs in their own right".
        # Noun form only; false positives possible ("correctness of our approach").
        r'\bcorrectness\b',
        re.IGNORECASE,
    ),
    # --- Additional criteria from Howcroft (2020) taxonomy ---
    "understandability": re.compile(
        # Howcroft: "Understandability" and "Clarity".
        # Noun forms only; "clarity" excluded (too generic in non-eval text).
        r'\bunderstandability\b'
        r'|\bcomprehensibility\b',
        re.IGNORECASE,
    ),
    "engagingness": re.compile(
        # Common in dialogue and story evaluation.
        # Noun forms only; "engaging" excluded (too common as adjective).
        r'\bengagingness\b'
        r'|\binterestingness\b',
        re.IGNORECASE,
    ),
    "humanlikeness": re.compile(
        # Howcroft: "Humanlikeness" -- degree to which output could be from a human.
        # Common in dialogue system evaluation.
        r'\bhumanlikeness\b'
        r'|\bhuman[\s-]likeness\b',
        re.IGNORECASE,
    ),
    "appropriateness": re.compile(
        # Howcroft: "Appropriateness" -- suitable in given context/situation.
        # Noun form only.
        r'\bappropriateness\b',
        re.IGNORECASE,
    ),
}

# ---------------------------------------------------------------------------
# Combined keyword detection pass over all HE papers (single file read each)
# ---------------------------------------------------------------------------
he_df = df[df["has_human_eval"]].copy()

print(f"\nKeyword detection pass (methods + criteria) over {len(he_df):,} HE papers ...")
_kw_rows = []
_missing_txt_kw = []
for _, row in he_df.iterrows():
    pid  = row["paper_id"]
    text = _read_paper_text(pid)

    # Methods: fall back to LLM-extracted field if full text unavailable
    method_text = text
    if not method_text:
        method_text = str(row.get("human_eval_methods") or "")

    source = ("full_text" if text
              else ("llm_fallback" if method_text.strip() else "missing"))
    if source == "missing":
        _missing_txt_kw.append(pid)

    entry = {"paper_id": pid, "year": row["year"], "source": source}
    # methods: use method_text (may include LLM fallback)
    for m, r in METHOD_RES.items():
        entry[f"m_{m}"] = int(bool(r.search(method_text)))
    # criteria: full text only (no LLM fallback -- that is what we are replacing)
    crit_text = text or ""
    for c, r in CRITERIA_RES.items():
        entry[f"c_{c}"] = int(bool(r.search(crit_text)))
    _kw_rows.append(entry)

he_kw = pd.DataFrame(_kw_rows)

_method_cols = [f"m_{m}" for m in METHOD_RES]
_crit_cols   = [f"c_{c}" for c in CRITERIA_RES]

# Convenience DataFrames with clean column names (no prefix)
he_methods = he_kw[["paper_id", "year", "source"] + _method_cols].copy()
he_methods.columns = ["paper_id", "year", "source"] + list(METHOD_RES.keys())

he_criteria = he_kw[["paper_id", "year"] + _crit_cols].copy()
he_criteria.columns = ["paper_id", "year"] + list(CRITERIA_RES.keys())

n_full    = (he_kw["source"] == "full_text").sum()
n_llmfb   = (he_kw["source"] == "llm_fallback").sum()
n_missing = (he_kw["source"] == "missing").sum()
print(f"  full text: {n_full:,}  |  llm fallback: {n_llmfb:,}  |  missing: {n_missing:,}")
if _missing_txt_kw:
    print(f"  Missing txt (first 10): {_missing_txt_kw[:10]}")

crit_cols   = list(CRITERIA_RES.keys())
method_cols = list(METHOD_RES.keys())

# ===========================================================================
# RQ1  Prevalence and trajectory of human evaluation
# ===========================================================================
print(f"\n{SEP}")
print("RQ1: Prevalence and trajectory of human evaluation")
print(SEP)

n_he = df["has_human_eval"].sum()
print(f"Papers with human eval: {n_he:,} / {len(df):,} ({n_he/len(df)*100:.1f}%)")

# 1.1 Overall time trend
fig, ax = plt.subplots(figsize=(10, 5))
dual_axis_trend(df, "year", "has_human_eval",
                "RQ1: Human evaluation prevalence over time", ax=ax)
plt.tight_layout()
save_fig(fig, "rq1_human_eval_timeline")
plt.close()

# 1.2 Logistic regression (statsmodels if available)
try:
    import statsmodels.formula.api as smf
    
    # HE Trend (overall)
    df_all = df.copy()
    df_all["has_human_eval"] = df_all["has_human_eval"].astype(int)
    m = smf.logit("has_human_eval ~ year", data=df_all).fit(disp=False)
    print(f"Logistic reg overall coef(year)={m.params['year']:.4f}  p={m.pvalues['year']:.4e}")
    
    # HE Trend (2015-2025)
    df_recent = df[df["year"].between(2015, 2025)].copy()
    df_recent["has_human_eval"] = df_recent["has_human_eval"].astype(int)
    m_he = smf.logit("has_human_eval ~ year", data=df_recent).fit(disp=False)
    print(f"Logistic reg HE (2015-2025) coef(year)={m_he.params['year']:.4f}  p={m_he.pvalues['year']:.4e}")
    
    # LLM judge Trend (2020-2025)
    df_llm = df[df["year"].between(2020, 2025)].copy()
    df_llm["has_llm_judge"] = df_llm["has_llm_judge"].astype(int)
    m_llm = smf.logit("has_llm_judge ~ year", data=df_llm).fit(disp=False)
    print(f"Logistic reg LLM-judge (2020-2025) coef(year)={m_llm.params['year']:.4f}  p={m_llm.pvalues['year']:.4e}")
    
    # Faithfulness Trend (2015-2025, HE only)
    df_he = he_criteria[he_criteria["year"].between(2015, 2025)].copy()
    df_he["faithfulness"] = df_he["faithfulness"].astype(int)
    m_faith = smf.logit("faithfulness ~ year", data=df_he).fit(disp=False)
    print(f"Logistic reg Faithfulness (2015-2025) coef(year)={m_faith.params['year']:.4f}  p={m_faith.pvalues['year']:.4e}")
    
except Exception as e:
    print(f"Logistic regression skipped: {e}")

# 1.3 By venue group
vg_data = df.groupby(["venue_group", "year"])["has_human_eval"].agg(["sum","count"])
fig, axes = plt.subplots(1, len(VENUE_GROUPS), figsize=(16, 5), sharey=True)
for ax, (vg, _) in zip(axes, VENUE_GROUPS.items()):
    sub = df[df["venue_group"] == vg]
    dual_axis_trend(sub, "year", "has_human_eval",
                    f"{vg} (n={len(sub):,})", ax=ax)
plt.suptitle("RQ1: Human evaluation rate by venue group")
plt.tight_layout()
save_fig(fig, "rq1_by_venue")
plt.close()

# 1.4 Post-2018 detailed trend (highlight the recent decline)
recent = df[df["year"] >= 2018]
fig, ax = plt.subplots(figsize=(10, 5))
dual_axis_trend(recent, "year", "has_human_eval",
                "RQ1: Human evaluation -- recent trend (2018-2025)", ax=ax)
plt.tight_layout()
save_fig(fig, "rq1_recent_decline")
plt.close()

yr_tbl = (
    df[df["year"] >= 2018]
    .groupby("year")["has_human_eval"]
    .agg(["sum","count","mean"])
    .rename(columns={"sum": "n_he", "count": "total", "mean": "rate"})
)
yr_tbl.to_csv("tables/rq1_yearly_he_rate.csv")
print("\nHuman eval rate 2018-2025:")
print(yr_tbl.to_string())

# 1.4b Main conf+journals+SIGGEN vs all venues (controls for workshop coverage)
MAIN_VENUE_GROUPS = {"generation", "core_nlp", "journals"}
df["is_main_conf"] = df["venue_group"].isin(MAIN_VENUE_GROUPS)

main_yr = (
    df[df["is_main_conf"] & (df["year"] >= 2018)]
    .groupby("year")["has_human_eval"]
    .agg(["sum", "count", "mean"])
    .rename(columns={"sum": "n_he", "count": "total", "mean": "rate"})
)
all_yr = yr_tbl.copy()
combined_venue = pd.DataFrame({
    "main_conf_rate": main_yr["rate"],
    "main_conf_n":   main_yr["total"],
    "all_rate":      all_yr["rate"],
    "all_n":         all_yr["total"],
})
print("\nHuman eval rate: main conf+journals+SIGGEN vs ALL venues (2018-2025):")
print(combined_venue.to_string())
combined_venue.to_csv("tables/rq1_main_vs_all_yearly.csv")

# 1.4c INLG series baseline analysis
print("\n=== INLG Series Baseline Analysis ===")
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
    print(inlg_yr.to_string())
    
    # Compare with non-INLG generation venues
    gen_other_df = df[(df["venue_group"] == "generation") & (df["venue"].astype(str).str.lower() != "inlg")]
    print(f"\nOther generation venues (n={len(gen_other_df)}): HE rate={gen_other_df['has_human_eval'].mean():.1%}")
    core_nlp_df = df[df["venue_group"] == "core_nlp"]
    print(f"Core NLP venues (n={len(core_nlp_df)}): HE rate={core_nlp_df['has_human_eval'].mean():.1%}")

    inlg_yr.to_csv("tables/rq1_inlg_yearly.csv")

# 1.5 By language group (excluding unknown)
lang_he = (
    df[df["lang_group"] != "unknown"]
    .groupby("lang_group")["has_human_eval"]
    .agg(["sum","count","mean"])
    .rename(columns={"sum":"n_he","count":"total","mean":"rate"})
    .sort_values("rate", ascending=False)
)
print("\nHuman eval rate by language group:")
print(lang_he.to_string())

# 1.5b By language group excluding MT papers (MT is multilingual by default)
non_mt = df[~df["inferred_tasks"].fillna("").str.contains("machine_translation", case=False, regex=False)].copy()
lang_he_non_mt = (
    non_mt[non_mt["lang_group"] != "unknown"]
    .groupby("lang_group")["has_human_eval"]
    .agg(["sum", "count", "mean"])
    .rename(columns={"sum": "n_he", "count": "total", "mean": "rate"})
    .sort_values("rate", ascending=False)
)
lang_he_non_mt.to_csv("tables/rq1_lang_he_rate_excluding_mt.csv")
print("\nHuman eval rate by language group (excluding MT papers):")
print(lang_he_non_mt.to_string())

# ===========================================================================
# RQ2  LLM-as-a-judge adoption
# ===========================================================================
print(f"\n{SEP}")
print("RQ2: LLM-as-a-judge adoption")
print(SEP)

print(f"has_llm_judge=True   : {df['has_llm_judge'].sum():,}")
print(f"  instruction_llm    : {df['has_true_llm_judge'].sum():,}")
print(f"  ml_evaluator       : {df['has_ml_judge'].sum():,}")

_auto_unknown_mask = df["has_llm_judge"] & (df["llm_judge_type_auto"] == "unknown")
print(f"  auto-unknown type  : {_auto_unknown_mask.sum():,}  (curation set)")
print(f"    assumed instruct : {( _auto_unknown_mask & (df['llm_judge_type']=='instruction_llm') ).sum():,}")
print(f"    manual ML        : {( _auto_unknown_mask & (df['llm_judge_type']=='ml_evaluator') ).sum():,}")

# 2.0 List all auto-unknown LLM judge models (for post-hoc categorisation)
unknown_j = df[_auto_unknown_mask].copy()
unknown_j["llm_judge_model"] = unknown_j["llm_judge_model"].fillna("").astype(str).str.strip()
unknown_j["llm_judge_model_clean"] = unknown_j["llm_judge_model"].replace({"": np.nan})

unknown_papers_out_path = "tables/rq2_unknown_llm_judge_papers.csv"

unknown_papers_base_cols = [
    "paper_id",
    "year",
    "venue",
    "venue_group",
    "inferred_tasks",
    "llm_judge_model",
    "llm_judge_type_auto",
]
if "llm_judge_criteria" in unknown_j.columns:
    unknown_papers_base_cols.append("llm_judge_criteria")

unknown_papers_out = unknown_j[[c for c in unknown_papers_base_cols if c in unknown_j.columns]].copy()
unknown_papers_out["llm_judge_type_final"] = unknown_j["llm_judge_type"]

# Preserve any user-added annotation columns (including manual type labels).
if os.path.exists(unknown_papers_out_path):
    try:
        existing = pd.read_csv(unknown_papers_out_path)
        if "paper_id" in existing.columns:
            preserve_cols = [c for c in existing.columns if c not in set(unknown_papers_out.columns) | {"paper_id"}]
            if preserve_cols:
                unknown_papers_out = unknown_papers_out.merge(
                    existing[["paper_id"] + preserve_cols],
                    on="paper_id",
                    how="left",
                )
    except Exception:
        pass

# If the manual type column doesn't exist yet, create it for annotation.
if ("llm_judge_type" not in unknown_papers_out.columns) and ("type" not in unknown_papers_out.columns):
    unknown_papers_out["llm_judge_type"] = ""

unknown_papers_out = unknown_papers_out.sort_values(["year", "venue", "paper_id"])
unknown_papers_out.to_csv(unknown_papers_out_path, index=False)

unknown_models = (
    unknown_j.dropna(subset=["llm_judge_model_clean"])
    ["llm_judge_model_clean"].value_counts()
)

# Preserve any user-added annotation columns (e.g., manual categorisation).
unknown_models_out_path = "tables/rq2_unknown_llm_judge_models.csv"
unknown_models_df = (
    unknown_models
    .rename("count")
    .reset_index()
    .rename(columns={"index": "llm_judge_model_clean"})
)
if os.path.exists(unknown_models_out_path):
    try:
        existing = pd.read_csv(unknown_models_out_path)
        if "llm_judge_model_clean" in existing.columns:
            extra_cols = [c for c in existing.columns if c not in {"llm_judge_model_clean", "count"}]
            if extra_cols:
                unknown_models_df = unknown_models_df.merge(
                    existing[["llm_judge_model_clean"] + extra_cols],
                    on="llm_judge_model_clean",
                    how="left",
                )
    except Exception:
        # If the existing file is malformed, fall back to writing fresh counts.
        pass

unknown_models_df.to_csv(unknown_models_out_path, index=False)

print("\nUnknown LLM-judge models (regex type=unknown):")
print(f"  papers: {len(unknown_j):,}")
print(f"  papers with non-empty model string: {int(unknown_j['llm_judge_model_clean'].notna().sum()):,}")
print(f"  unique model strings: {len(unknown_models):,}")
if len(unknown_models):
    # Full list also written to tables/rq2_unknown_llm_judge_models.csv
    print(unknown_models.to_string())

# 2.1 Dual timeline: instruction LLM vs ML evaluator
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
dual_axis_trend(df, "year", "has_true_llm_judge",
                "RQ2a: Instruction-LLM-as-judge over time",
                ax=axes[0], color_bar="#55A868", color_line="#C44E52")
dual_axis_trend(df, "year", "has_ml_judge",
                "RQ2b: ML-model-as-evaluator over time",
                ax=axes[1], color_bar="#4C72B0", color_line="#DD8452")
plt.tight_layout()
save_fig(fig, "rq2_judge_timeline_by_type")
plt.close()

# 2.2 Is LLM judge displacing human eval? Co-occurrence over time
llm_yr = (
    df[df["has_true_llm_judge"]]
    .groupby("year")
    .apply(lambda x: pd.Series({
        "n_llm": len(x),
        "pct_also_human": x["has_human_eval"].mean() * 100,
        "pct_auto_only":  ((~x["has_human_eval"]) & x["has_auto_eval"]).mean() * 100,
    }))
    .reset_index()
)
llm_yr.to_csv("tables/rq2_llm_judge_cooccurrence.csv", index=False)
print("\nLLM-judge + human eval co-occurrence over time:")
print(llm_yr.to_string(index=False))

# 2.3 Top judge models (instruction LLM only)
models = (
    df[df["has_true_llm_judge"] & df["llm_judge_model"].notna()]
    ["llm_judge_model"].value_counts().head(12)
)
fig, ax = plt.subplots(figsize=(8, 5))
models.plot(kind="barh", ax=ax, color="#55A868", alpha=0.8)
ax.set_title("RQ2: Instruction-LLM judge models (top 12)")
ax.set_xlabel("Count")
ax.invert_yaxis()
_style_axes(ax)
plt.tight_layout()
save_fig(fig, "rq2_judge_models")
plt.close()

# ===========================================================================
# RQ3  Human evaluation criteria
# ===========================================================================
print(f"\n{SEP}")
print("RQ3: Human evaluation criteria")
print(SEP)

# 3.1 Overall frequency (keyword-based, all HE papers with full text)
he_criteria.to_csv("tables/rq3_criteria_kw.csv", index=False)
crit_totals = he_criteria[crit_cols].sum().sort_values(ascending=False)

print(f"Criteria prevalence across {len(he_criteria):,} HE papers (keyword-based):")
for c, cnt in crit_totals.items():
    print(f"  {c:<25s}: {cnt:5,}  ({cnt/len(he_criteria)*100:.1f}%)")

# 3.2 Top criteria over time
he_per_yr  = he_df.groupby("year").size().rename("total")
crit_per_yr = he_criteria.groupby("year")[crit_cols].sum()
crit_rate_yr = crit_per_yr.div(he_per_yr, axis=0).fillna(0)
crit_rate_yr.to_csv("tables/rq3_criteria_kw_trend.csv")

TOP_CRITERIA = crit_totals.head(8).index.tolist()

fig, ax = plt.subplots(figsize=(12, 5))
for crit in TOP_CRITERIA:
    if crit in crit_rate_yr.columns:
        s = crit_rate_yr[crit].rolling(3, min_periods=1).mean()
        ax.plot(s.index, s.values, marker="o", markersize=3, label=crit)
ax.set_title("RQ3: Top human evaluation criteria over time (keyword-based, 3-year rolling)")
ax.set_xlabel("Year")
ax.set_ylabel("Fraction of HE papers")
ax.legend(fontsize=8, ncol=2)
_style_axes(ax)
_style_year_axis(ax)
plt.tight_layout()
save_fig(fig, "rq3_criteria_over_time")
plt.close()

# 3.2b Small multiples: top criteria trends (counts + proportions)
TOP_CRITERIA_10 = crit_totals.head(10).index.tolist()
years = crit_rate_yr.index.sort_values().tolist()
x = np.arange(len(years))

fig, axes = plt.subplots(2, 5, figsize=(13, 5), sharey=False)
for ax, crit in zip(axes.flatten(), TOP_CRITERIA_10):
    counts = crit_per_yr[crit].reindex(years, fill_value=0).values
    rate = crit_rate_yr[crit].reindex(years, fill_value=0).values
    rate_smooth = pd.Series(rate, index=years).rolling(3, min_periods=1, center=True).mean().values

    ax.bar(x, counts, color="#4C72B0", alpha=0.45)
    ax2 = ax.twinx()
    ax2.plot(x, rate_smooth, color="#DD8452", linewidth=2.5)
    ax2.plot(x, rate, color="#DD8452", alpha=0.22, linewidth=1.25)
    ax2.set_ylim(0, 1)

    ax.set_title(crit, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) if y % 5 == 0 else "" for y in years], rotation=45, ha="right", fontsize=8)

    _style_axes(ax)
    ax2.grid(False)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)

plt.tight_layout()
save_fig(fig, "rq3_criteria_trends")
plt.close()

# 3.2b Criteria by task heatmap (using df_single)
he_s = df_single[df_single['has_human_eval'] & df_single['criteria_norm'].fillna("").str.strip().astype(bool)].copy()
he_s_exp = (
    he_s[['task', 'criteria_norm']]
    .assign(criterion=he_s['criteria_norm'].str.split('|'))
    .explode('criterion')
    .assign(criterion=lambda x: x['criterion'].str.strip())
)
he_s_exp = he_s_exp[he_s_exp['criterion'].isin(TOP_CRITERIA_10)]
he_per_task = he_s.groupby('task').size().rename('he_total')
crit_by_task = (
    he_s_exp.groupby(['task', 'criterion']).size()
    .reset_index(name='count')
    .merge(he_per_task, on='task')
    .assign(rate=lambda x: x['count'] / x['he_total'])
)
top_tasks_10 = df_single['task'].value_counts().head(10).index
pivot_ct = (
    crit_by_task.pivot(index='criterion', columns='task', values='rate')
    .fillna(0)
    .reindex(columns=[t for t in top_tasks_10 if t in crit_by_task['task'].unique()])
)

pivot_ct.index = [i.replace('_', ' ').capitalize() for i in pivot_ct.index]
pivot_ct.columns = [c.replace('_', ' ').capitalize() for c in pivot_ct.columns]
pivot_ct.index.name = None
pivot_ct.columns.name = None

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot_ct, annot=True, fmt='.0%', cmap='YlOrRd', linewidths=0.5, ax=ax)
ax.set_title('')
ax.set_xlabel('')
ax.set_ylabel('')
plt.xticks(rotation=35, ha='right')
plt.tight_layout()
save_fig(fig, 'rq3_criteria_by_task')
plt.close()


# 3.3 Faithfulness trajectory (post-2015 focus -- summarization era)
print("\nFaithfulness (keyword) mentions by year (2015+):")
if "faithfulness" in crit_rate_yr.columns:
    print(crit_rate_yr["faithfulness"][crit_rate_yr.index >= 2015].round(3).to_string())

# 3.4 Multiple criteria per paper (shows keyword detection captures co-occurrence)
n_crit_per_paper = he_criteria[crit_cols].sum(axis=1)
print(f"\nCriteria per HE paper: mean={n_crit_per_paper.mean():.2f} "
      f"median={n_crit_per_paper.median():.1f} "
      f"max={n_crit_per_paper.max():.0f}")

# ===========================================================================
# RQ4  Evaluation modality mix (human vs. auto vs. LLM-judge)
# ===========================================================================
print(f"\n{SEP}")
print("RQ4: Evaluation modality mix")
print(SEP)

def eval_modality(row):
    h = bool(row["has_human_eval"])
    a = bool(row["has_auto_eval"])
    j = bool(row["has_true_llm_judge"])
    if not h and not a and not j:
        return "none_reported"
    if h and not a and not j:
        return "human_only"
    if a and not h and not j:
        return "auto_only"
    if j and not h and not a:
        return "llm_judge_only"
    if h and a and not j:
        return "human_and_auto"
    if j and h:
        return "llm_judge_and_human"
    if j and a:
        return "llm_judge_and_auto"
    return "all_three"

df["eval_modality"] = df.apply(eval_modality, axis=1)
ORDER = ["human_only","human_and_auto","auto_only","llm_judge_and_human",
         "llm_judge_and_auto","llm_judge_only","none_reported"]

print("Evaluation modality distribution:")
print(df["eval_modality"].value_counts().to_string())

# 4.1 Stacked bar: modality mix over time (5-year buckets for readability)
df["period"] = ((df["year"] // 5) * 5).astype(int)
mod_yr = (
    df.groupby(["period","eval_modality"])
    .size().unstack(fill_value=0)
    .reindex(columns=[c for c in ORDER if c in df["eval_modality"].unique()], fill_value=0)
)
mod_yr_pct = mod_yr.div(mod_yr.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 5))
mod_yr_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", legend=False)
ax.set_xlabel("Period")
ax.set_ylabel("")

legend_labels = {
    "human_only": "Human only",
    "human_and_auto": "Human & automatic",
    "auto_only": "Automatic only",
    "llm_judge_and_human": "LLM-judge & human",
    "llm_judge_and_auto": "LLM-judge & automatic",
    "llm_judge_only": "LLM-judge only",
    "none_reported": "None reported"
}
handles, labels = ax.get_legend_handles_labels()
clean_labels = [legend_labels.get(l, l.replace("_", " ").capitalize()) for l in labels]
ax.legend(handles, clean_labels, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=7, fontsize=9, frameon=True)

plt.xticks(rotation=45, ha="right")
_style_axes(ax, y_grid=False)
plt.tight_layout()
save_fig(fig, "rq4_modality_mix")
plt.close()
mod_yr_pct.to_csv("tables/rq4_modality_mix.csv")

# 4.1b Simplified modality profile by venue group (%), matching the older notebook figure rq4_by_venue
def eval_type4(row):
    h = bool(row["has_human_eval"])
    a = bool(row["has_auto_eval"])
    j = bool(row["has_true_llm_judge"])
    if j and not h:
        return "llm_judge_no_human"
    if h and a:
        return "both_human_auto"
    if h and not a:
        return "human_only"
    if a and not h:
        return "auto_only"
    return "none"

ET4_ORDER = ["human_only", "both_human_auto", "auto_only", "llm_judge_no_human", "none"]
ET4_COLORS = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#CCCCCC"]

df["eval_type4"] = df.apply(eval_type4, axis=1)
vg_et = (
    df.groupby(["venue_group", "eval_type4"]).size()
    .unstack(fill_value=0)
    .reindex(columns=ET4_ORDER, fill_value=0)
)
vg_et_pct = vg_et.div(vg_et.sum(axis=1), axis=0) * 100
vg_et_pct = vg_et_pct.rename(index={
    "core_nlp": "*ACL",
    "generation": "SIGGEN",
    "journals": "journals",
    "other": "other"
})
vg_et_pct.to_csv("tables/rq4_by_venue_eval_type4_pct.csv")

print("\nRQ4: Simplified modality profile by venue group (%):")
print(vg_et_pct.round(1).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
vg_et_pct.plot(kind="bar", stacked=True, ax=ax, color=ET4_COLORS, alpha=0.85, width=0.65)
ax.set_ylabel("")
ax.set_xlabel("")

et4_legend_labels = {
    "human_only": "Human only",
    "both_human_auto": "Human & automatic",
    "auto_only": "Automatic only",
    "llm_judge_no_human": "LLM-judge (no human)",
    "none": "None reported"
}
handles, labels = ax.get_legend_handles_labels()
clean_labels = [et4_legend_labels.get(l, l.replace("_", " ").capitalize()) for l in labels]
ax.legend(handles, clean_labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=9, frameon=True)

plt.xticks(rotation=0, ha="center")
_style_axes(ax, y_grid=False)
plt.tight_layout()
save_fig(fig, "rq4_by_venue")
plt.close()

# 4.1c Early-period reporting: how common is none_reported?
none_by_period = (
    df.groupby("period")["eval_modality"]
    .apply(lambda s: (s == "none_reported").mean() * 100)
)
none_by_period.to_csv("tables/rq4_none_reported_by_period.csv", header=["pct_none_reported"])
print("\nRQ4: % papers with no reported evaluation signal by 5-year period (selected early periods):")
print(none_by_period[none_by_period.index <= 1980].round(1).to_string())

# 4.2 Fraction skipping human eval entirely (no human, no LLM-judge)
df["no_human_signal"] = ~df["has_human_eval"] & ~df["has_true_llm_judge"]
fig, ax = plt.subplots(figsize=(10, 5))
dual_axis_trend(df, "year", "no_human_signal",
                "RQ4: Papers with no human evaluation signal (auto-only or none)",
                ax=ax, color_bar="#C44E52", color_line="#4C72B0")
plt.tight_layout()
save_fig(fig, "rq4_no_human_signal")
plt.close()

# ===========================================================================
# RQ5  Task-based differences in evaluation practice
# ===========================================================================
print(f"\n{SEP}")
print("RQ5: Task-based differences")
print(SEP)

TOP_TASKS = df_single["task"].value_counts().head(8).index.tolist()

# 5.1 HE rate by task
task_eval = (
    df_single[df_single["task"].isin(TOP_TASKS)]
    .groupby("task")
    .agg(
        total=("has_human_eval","count"),
        n_he=("has_human_eval","sum"),
        he_rate=("has_human_eval","mean"),
        n_auto=("has_auto_eval","sum"),
        auto_rate=("has_auto_eval","mean"),
        n_llmj=("has_true_llm_judge","sum"),
    )
    .sort_values("he_rate", ascending=False)
)
task_eval.to_csv("tables/rq5_task_eval_profile.csv")
print("Evaluation profile by task:")
print(task_eval.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
task_eval["he_rate"].sort_values().plot(kind="barh", ax=axes[0],
                                         color="#4C72B0", alpha=0.8)
axes[0].set_title("RQ5: Human eval rate by task")
axes[0].set_xlabel("Proportion")
_style_axes(axes[0])

task_eval["auto_rate"].sort_values().plot(kind="barh", ax=axes[1],
                                           color="#DD8452", alpha=0.8)
axes[1].set_title("RQ5: Automatic eval rate by task")
axes[1].set_xlabel("Proportion")
_style_axes(axes[1])
plt.tight_layout()
save_fig(fig, "rq5_eval_by_task")
plt.close()

# 5.2 HE rate over time for MT vs non-MT
df["is_mt"] = df["inferred_tasks"].fillna("").str.contains("machine_translation")
mt_trend = df.groupby(["year","is_mt"])["has_human_eval"].agg(["sum","count","mean"]).reset_index()
mt_trend.to_csv("tables/rq5_mt_vs_nonmt_trend.csv", index=False)
print("\nMT vs non-MT human eval rate (recent years):")
print(mt_trend[mt_trend["year"] >= 2015].pivot(index="year", columns="is_mt", values="mean").round(3).to_string())

# ===========================================================================
# RQ6  Multilingual and non-English evaluation practices
# ===========================================================================
print(f"\n{SEP}")
print("RQ6: Multilingual and non-English evaluation")
print(SEP)

lang_profile = (
    df[df["lang_group"] != "unknown"]
    .groupby("lang_group")
    .agg(
        n=("has_human_eval","count"),
        he_rate=("has_human_eval","mean"),
        auto_rate=("has_auto_eval","mean"),
        llmj_rate=("has_true_llm_judge","mean"),
    )
    .sort_values("he_rate", ascending=False)
)
lang_profile.to_csv("tables/rq6_lang_eval_profile.csv")
print("Evaluation profile by language group:")
print(lang_profile.to_string())

# 6.1 Task profile of multilingual papers
multi_tasks = (
    df_tasks[df_tasks["lang_group"] == "multilingual"]["task"]
    .value_counts(normalize=True)
    .head(8) * 100
)
print("\nTop tasks in multilingual papers (%):")
print(multi_tasks.round(1).to_string())

# 6.2 HE rate for non-MT multilingual papers vs MT multilingual
multi = df_tasks[df_tasks["lang_group"] == "multilingual"].copy()
multi["is_mt"] = multi["task"] == "machine_translation"
print("\nMultilingual HE rate: MT vs non-MT:")
print(multi.groupby("is_mt")["has_human_eval"].agg(["sum","count","mean"]).to_string())

# 6.3 Human eval rate by task over time (single-task papers) -- refreshed rq6_human_eval_by_task
TOP_TASKS_RQ6 = df_single["task"].value_counts().head(8).index.tolist()
fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=False)
for ax, task in zip(axes.flatten(), TOP_TASKS_RQ6):
    sub = df_single[df_single["task"] == task]
    if len(sub) == 0:
        ax.set_title(task)
        continue
    dual_axis_trend(sub, "year", "has_human_eval", task, ax=ax)
    ax.set_xlabel("")

plt.suptitle("RQ6: Human eval rate per task over time (single-task papers)", y=1.01)
plt.tight_layout()
save_fig(fig, "rq6_human_eval_by_task")
plt.close()

# 6.3 Non-English monolingual trend
fig, ax = plt.subplots(figsize=(10, 5))
sub_lg = df[df["lang_group"] != "unknown"]
year_list = sorted(sub_lg["year"].dropna().unique())
_, ax_bar, ax_rel = dual_axis_trend(
    sub_lg,
    "year", "has_human_eval",
    "RQ6: Human eval rate over time by language group",
    ax=ax,
)
# Overlay non-English mono line
sub_ne = df[df["lang_group"] == "non_english_mono"]
ne_yr = sub_ne.groupby("year")["has_human_eval"].agg(["sum","count","mean"])
ne_yr = ne_yr[ne_yr["count"] >= 5]
if len(ne_yr):
    x_ne = np.array([year_list.index(int(y)) for y in ne_yr.index if int(y) in year_list], dtype=float)
    y_ne = np.array([ne_yr.loc[int(y), "mean"] for y in ne_yr.index if int(y) in year_list], dtype=float)
    ax_rel.plot(x_ne, y_ne, color="purple", linewidth=1.5, linestyle="--", label="non-English mono")
plt.tight_layout()
save_fig(fig, "rq6_lang_group_trend")
plt.close()

lang_yr = (
    df[df["lang_group"].isin(["multilingual","english_only","non_english_mono"])
       & df["year"].between(2010, 2025)]
    .groupby(["year","lang_group"])["has_human_eval"]
    .agg(["sum","count","mean"])
    .reset_index()
)
lang_yr.to_csv("tables/rq6_lang_trend.csv", index=False)

# ===========================================================================
# RQ7  Annotation methodology types (keyword-based detection)
# NOTE: num_annotators / num_items_rated / agreement_reported were extracted
# by the LLM but recall against gold annotations was too low (~18%) to support
# quantitative claims. We report only the *rate* of reporting and use
# keyword-based regex on full paper text for methodology types instead.
# ===========================================================================
print(f"\n{SEP}")
print("RQ7: Annotation methodology types")
print(SEP)

# he_df, he_methods, he_criteria already computed in the shared detection pass above.

# 7.0 Reporting-rate-only summary (LLM-extracted counts; recall too low for values)
for col in ("num_annotators", "num_items_rated"):
    he_df[col] = pd.to_numeric(he_df[col], errors="coerce")
    he_df.loc[he_df[col] <= 0, col] = np.nan
print(f"Papers reporting annotator count : "
      f"{he_df['num_annotators'].notna().sum():,} / {len(he_df):,} "
      f"({he_df['num_annotators'].notna().mean()*100:.1f}%)")
print(f"Papers reporting item count      : "
      f"{he_df['num_items_rated'].notna().sum():,} / {len(he_df):,} "
      f"({he_df['num_items_rated'].notna().mean()*100:.1f}%)")
print(f"Papers reporting IAA             : "
      f"{he_df['agreement_reported'].sum():,} / {len(he_df):,} "
      f"({he_df['agreement_reported'].mean()*100:.1f}%)")
print("(Counts not reported -- LLM extraction recall too low for quantitative claims)")

# 7.1 Method prevalence (from combined detection pass)
he_methods.to_csv("tables/rq7_annotation_methods.csv", index=False)
print("\nAnnotation method prevalence across HE papers (keyword-based):")
method_totals = he_methods[method_cols].sum().sort_values(ascending=False)
for m, cnt in method_totals.items():
    print(f"  {m:<25s}: {cnt:5,}  ({cnt/len(he_methods)*100:.1f}%)")
print(f"\nMethods per HE paper: mean={he_methods[method_cols].sum(axis=1).mean():.2f} "
      f"(confirms multiple methods per paper are captured)")

# 7.2 Method type over time (smoothed)
yr_methods = (
    he_methods[he_methods["year"].between(2000, 2025)]
    .groupby("year")[method_cols]
    .mean()
    .rolling(3, min_periods=1)
    .mean()
)
yr_methods.to_csv("tables/rq7_method_trend.csv")

fig, ax = plt.subplots(figsize=(12, 5))
COLORS = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860","#DA8BC3","#8C8C8C","#4C4C4C"]
for col, color in zip(method_cols, COLORS):
    if col in yr_methods.columns and yr_methods[col].sum() > 0:
        ax.plot(yr_methods.index, yr_methods[col], label=col, color=color,
                linewidth=1.8, marker="o", markersize=3)
ax.set_title("RQ7: Annotation method types in HE papers over time (3-year rolling)")
ax.set_xlabel("Year")
ax.set_ylabel("Fraction of HE papers")
ax.legend(fontsize=8, ncol=2)
_style_axes(ax)
_style_year_axis(ax)
plt.tight_layout()
save_fig(fig, "rq7_method_trend")
plt.close()

# Also save under the older notebook figure name for cross-referencing
fig, ax = plt.subplots(figsize=(6, 3.5))
top6 = method_totals.head(6).index.tolist()
yr_methods_top = (
    he_methods[he_methods["year"].between(2000, 2025)]
    .groupby("year")[top6]
    .mean()
    .rolling(3, min_periods=1)
    .mean()
)
for col, color in zip(top6, COLORS):
    ax.plot(yr_methods_top.index, yr_methods_top[col], label=col, color=color,
            linewidth=2, marker="o", markersize=3)
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis='x', rotation=0, labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.legend(fontsize=14, ncol=2)
_style_axes(ax)
_style_year_axis(ax)
plt.tight_layout()
save_fig(fig, "rq3b_methods_trends")
plt.close()

# MT-specific method trends (single-task MT HE papers)
mt_he = df_single[(df_single["task"] == "machine_translation") & (df_single["has_human_eval"])].copy()
mt_he = mt_he.merge(he_methods[["paper_id"] + method_cols], on="paper_id", how="left")
mt_he[method_cols] = mt_he[method_cols].fillna(0)

mt_yr = (
    mt_he.groupby("year")[method_cols]
    .mean()
    .rolling(3, min_periods=1)
    .mean()
)
mt_yr.to_csv("tables/rq3b_mt_methods_trend_kw.csv")

MT_KEY = [c for c in ["direct_assessment", "span", "post_editing"] if c in mt_yr.columns]
fig, ax = plt.subplots(figsize=(10, 5))
mt_colors = {"direct_assessment": "#4C72B0", "span": "#55A868", "post_editing": "#C44E52"}
for m in MT_KEY:
    ax.plot(mt_yr.index, mt_yr[m], marker="o", label=m,
            color=mt_colors.get(m, "#8172B3"), linewidth=2)
ax.set_title("RQ3b: MT-specific evaluation methods over time (keyword-based)")
ax.set_xlabel("Year")
ax.set_ylabel("Fraction of MT HE papers")
ax.set_ylim(0, 1)
ax.legend(fontsize=9)
_style_axes(ax)
_style_year_axis(ax)
plt.tight_layout()
save_fig(fig, "rq3b_mt_methods")
plt.close()

if "post_editing" in mt_yr.columns and mt_yr["post_editing"].notna().any():
    pe_mid_2010s = mt_yr.loc[(mt_yr.index >= 2014) & (mt_yr.index <= 2016), "post_editing"].mean()
    pe_recent = mt_yr.loc[(mt_yr.index >= 2023) & (mt_yr.index <= 2025), "post_editing"].mean()
    print("\nMT post-editing trend (3-year-rolled average rates):")
    if 2015 in mt_yr.index:
        pe_2015 = float(np.asarray(mt_yr.loc[2015, "post_editing"], dtype=float))
        print(f"  2015: {pe_2015*100:.1f}%")
    if 2025 in mt_yr.index:
        pe_2025 = float(np.asarray(mt_yr.loc[2025, "post_editing"], dtype=float))
        print(f"  2025: {pe_2025*100:.1f}%")
    print(f"  2014-2016 mean: {pe_mid_2010s*100:.1f}%")
    print(f"  2023-2025 mean: {pe_recent*100:.1f}%")

# 7.3 IAA reporting rate trend (separate from counts -- this is reliable)
fig, ax = plt.subplots(figsize=(10, 5))
dual_axis_trend(he_df, "year", "agreement_reported",
                "RQ7: Inter-annotator agreement reporting rate over time",
                ax=ax, color_bar="#8172B3", color_line="#C44E52")
plt.tight_layout()
save_fig(fig, "rq7_agreement_reporting")
plt.close()

he_df["period"] = ((he_df["year"] // 5) * 5).astype(int)
agr_period = (
    he_df.groupby("period")["agreement_reported"]
    .agg(["sum", "count", "mean"])
    .rename(columns={"sum": "n_agr", "count": "total", "mean": "rate"})
)
agr_period.to_csv("tables/rq7_agreement_rate.csv")
print("\nIAA reporting rate by 5-year period (HE papers only):")
print(agr_period.to_string())

# ===========================================================================
# Summary: notable findings
# ===========================================================================
print(f"\n{SEP}")
print("NOTABLE FINDINGS SUMMARY")
print(SEP)

he_2021 = df[df["year"]==2021]["has_human_eval"].mean()
he_2024 = df[df["year"]==2024]["has_human_eval"].mean()
print(f"HE rate: 2021={he_2021*100:.1f}%  2024={he_2024*100:.1f}%  "
      f"(drop: {(he_2021-he_2024)*100:.1f} pp)")

print(f"Agreement ever reported: {he_df['agreement_reported'].mean()*100:.1f}% of HE papers")

print(f"\nEnglish-only HE rate: "
      f"{df[df['lang_group']=='english_only']['has_human_eval'].mean()*100:.1f}%")
print(f"Multilingual HE rate: "
      f"{df[df['lang_group']=='multilingual']['has_human_eval'].mean()*100:.1f}%")
print(f"Multilingual papers that are MT: "
      f"{(df_tasks[df_tasks['lang_group']=='multilingual']['task']=='machine_translation').mean()*100:.1f}%")

print("\nFigures -> figures/   Tables -> tables/")

# --- Recall Check ---
print(f"\n{SEP}")
print("Recall Check on Human Eval of Hallucinations")
print(SEP)
try:
    gold_df = pd.read_excel("../Human Eval of Hallucinations Overview.xlsx", sheet_name="Detailed Annotations")
    gold_ids = set(gold_df["id"].dropna().astype(str))
    
    # We only care about papers that are in our final df
    gold_in_corpus = gold_ids.intersection(set(df["paper_id"].astype(str)))
    print(f"Total annotations in sheet: {len(gold_ids)}")
    print(f"Total annotations mapped to corpus: {len(gold_in_corpus)}")
    
    # Old recall (before override)
    old_tp = df[df["paper_id"].astype(str).isin(gold_in_corpus)]["has_human_eval_original"].sum()
    old_recall = old_tp / len(gold_in_corpus) if len(gold_in_corpus) > 0 else 0
    
    # New recall (after override)
    new_tp = df[df["paper_id"].astype(str).isin(gold_in_corpus)]["has_human_eval"].sum()
    new_recall = new_tp / len(gold_in_corpus) if len(gold_in_corpus) > 0 else 0
    
    print(f"Old Recall (before override): {old_recall*100:.1f}% ({old_tp}/{len(gold_in_corpus)})")
    print(f"New Recall (after override):  {new_recall*100:.1f}% ({new_tp}/{len(gold_in_corpus)})")
    print(f"Improvement: +{(new_recall - old_recall)*100:.1f} pp")
except Exception as e:
    print(f"Failed to check recall: {e}")

# Copy generated figures to paper_latex
import shutil
latex_dir = "../paper_latex"
if os.path.exists(latex_dir):
    figures_to_copy = [
        "rq3_criteria_trends.pdf",
        "rq4_modality_mix.pdf",
        "rq3b_methods_trends.pdf",
        "rq3_criteria_by_task.pdf"
    ]
    for fig_name in figures_to_copy:
        src = os.path.join(FIGURES_DIR, fig_name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(latex_dir, fig_name))
            print(f"Copied {fig_name} to paper_latex")
        else:
            print(f"Warning: {src} not found, couldn't copy to paper_latex")

