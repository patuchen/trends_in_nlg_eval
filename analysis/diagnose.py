"""
Diagnostic script for data quality issues.
Run from the analysis/ directory:  python diagnose.py
"""

import re
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RESULTS_CSV = "../output/results.csv"
PAPERS_CSV  = "../papers.csv"

df = pd.read_csv(RESULTS_CSV, encoding="utf-8", on_bad_lines="skip")
bool_cols = [
    "parse_failed", "haiku_spot_checked", "haiku_correction",
    "has_human_eval", "has_auto_eval", "has_llm_judge",
    "agreement_reported", "monolingual",
]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )


# ── helpers ────────────────────────────────────────────────────────────────
def _extract_venue(paper_id: str) -> str:
    m = re.match(r"\d{4}\.([a-z][a-z0-9-]+)-", str(paper_id))
    if m:
        return m.group(1)
    m = re.match(r"([A-Z])(\d{2})-", str(paper_id))
    if m:
        code = m.group(1)
        legacy = {
            "J": "cl", "Q": "tacl", "P": "acl", "N": "naacl",
            "D": "emnlp", "E": "eacl", "W": "workshop",
            "C": "coling", "L": "lrec", "S": "semeval",
        }
        return legacy.get(code, code.lower())
    return "unknown"

# Confirmed INLG/ENLG W-codes (from title inspection above)
INLG_WCODES = {
    "W07-23", "W08-11", "W09-06", "W10-42",
    "W11-28", "W13-21", "W14-44", "W16-66",
    "W17-35", "W18-65", "W19-86",
}

def extract_venue_fixed(paper_id: str) -> str:
    pid = str(paper_id)
    # W-prefix: check for INLG workshop codes before falling back to 'workshop'
    m = re.match(r"(W\d{2}-\d{2})", pid)
    if m and m.group(1) in INLG_WCODES:
        return "inlg"
    return _extract_venue(pid)

df["venue"] = df["paper_id"].apply(extract_venue_fixed)


# ═══════════════════════════════════════════════════════════════════════════
# 1. 1952 papers
# ═══════════════════════════════════════════════════════════════════════════
sep = "=" * 70
print(sep)
print("1.  PAPERS FROM 1952")
print(sep)
p52 = df[df["year"] == 1952][["paper_id", "title", "has_human_eval", "inferred_tasks"]]
print(f"Count: {len(p52)}")
print()
for _, r in p52.iterrows():
    url = f"https://aclanthology.org/{r['paper_id']}"
    print(f"  ID:    {r['paper_id']}")
    print(f"  URL:   {url}")
    print(f"  Title: {r['title']}")
    print(f"  human_eval={r['has_human_eval']}  tasks={r['inferred_tasks']}")
    print()

print("Assessment: all 7 are from the ACL 'earlymt' collection (1952 MIT")
print("mechanical translation conference). Correctly has_human_eval=False —")
print("there was no systematic output evaluation in that era.")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Multilingual bug
# ═══════════════════════════════════════════════════════════════════════════
print(sep)
print("2.  MULTILINGUAL CLASSIFICATION BUG")
print(sep)
print("Root cause:")
print("  str(np.nan) → 'nan' (truthy), so `str(val) or ''` never returns ''.")
print("  classify_language then adds 'nan' to all_langs, which is not 'english',")
print("  so any paper with missing src/tgt languages and monolingual=False")
print("  (the CSV default) gets classified as 'multilingual'.")
print()

non_mt = df[~df["inferred_tasks"].fillna("").str.contains("machine_translation")]
bug = non_mt[
    non_mt["source_languages"].isna()
    & non_mt["target_languages"].isna()
    & (non_mt["monolingual"] == False)
]
print(f"Non-MT papers:                           {len(non_mt):,}")
print(f"  → misclassified as multilingual:       {len(bug):,}  ({len(bug)/len(non_mt)*100:.1f}%)")
print()
print("Fix (in helpers.py classify_language):")
print("  Replace  `str(row.get(...) or '').strip()`")
print("  With     `'' if pd.isna(v) else str(v).strip()`")
print("  Also filter 'nan' strings from all_langs.")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Venue mapping: INLG before 2020
# ═══════════════════════════════════════════════════════════════════════════
print(sep)
print("3.  INLG VENUE MAPPING (pre-2020 W-prefix IDs)")
print(sep)
inlg_papers = df[df["venue"] == "inlg"]
print(f"INLG papers after fix: {len(inlg_papers)}")
print()
print("Confirmed INLG/ENLG W-code mappings:")
for code in sorted(INLG_WCODES):
    sub = df[df["paper_id"].str.startswith(code)]
    if len(sub):
        yr = sub["year"].iloc[0]
        ex = sub["title"].iloc[0][:60]
        print(f"  {code} → inlg  ({yr}, {len(sub)} papers) e.g. '{ex}...'")
print()
print("Note: W12-15 excluded — 'Rich Morphology Generation via SMT' is MT-adjacent,")
print("not clearly an INLG proceedings paper.")
print()
print("INLG counts by year (after fix):")
print(df[df["venue"] == "inlg"].groupby("year").size().sort_index().to_string())
print()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Early LLM-as-judge false positives
# ═══════════════════════════════════════════════════════════════════════════
print(sep)
print("4.  EARLY LLM-AS-JUDGE FALSE POSITIVES")
print(sep)
early = df[df["has_llm_judge"] & (df["year"] < 2023)].sort_values("year")
print(f"Papers flagged has_llm_judge=True before 2023: {len(early)}")
print()
print("These are almost certainly extraction errors — pre-2023 'judges' are")
print("adversarial discriminators, GPT-2-small perplexity filters, CLIP")
print("retrievers, etc., NOT instruction-tuned LLMs used as evaluators.")
print()
print(f"{'paper_id':30s} {'year':5s} {'llm_judge_model':35s} title")
print("-" * 120)
for _, r in early.iterrows():
    model = str(r["llm_judge_model"])[:33] if pd.notna(r["llm_judge_model"]) else "NaN"
    title = str(r["title"])[:45]
    print(f"  {r['paper_id']:28s} {r['year']:5d} {model:35s} {title}")

print()
print("Where to edit:")
print("  File:   output/results.csv")
print("  Column: has_llm_judge  →  set to False for each paper_id above")
print("  Column: llm_judge_model → clear the value")
print()
print("  You can also edit output/results.jsonl: find the paper by paper_id,")
print("  change  llm_as_judge.conducted  from true to false.")
print()
print("Suggested threshold: treat has_llm_judge=True as credible only from")
print("2023 onwards (ChatGPT/GPT-4 era). Papers from 2022 using 'GPT-2 small'")
print("or 'Codex' as an evaluator are borderline — your call.")
print()
print("Papers from 2022 that might be legitimate (using GPT-3/Codex):")
borderline = early[early["year"] == 2022]
for _, r in borderline.iterrows():
    model = str(r["llm_judge_model"]) if pd.notna(r["llm_judge_model"]) else "NaN"
    print(f"  {r['paper_id']}  model={model}  {str(r['title'])[:60]}")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Year label overlap fix (reminder)
# ═══════════════════════════════════════════════════════════════════════════
print(sep)
print("5.  YEAR LABEL READABILITY — proposed fix")
print(sep)
print("In any plot over the full year range, use:")
print()
print("  ax.set_xticks(x)")
print("  ax.set_xticklabels(")
print("      [str(y) if y % 5 == 0 else '' for y in years],")
print("      rotation=45, ha='right'")
print("  )")
print()
print("This labels only 1950, 1955, 1960, … 2005, 2010, 2015, 2020, 2025.")
print()


# ═══════════════════════════════════════════════════════════════════════════
# 6. RQ1 heatmap readability
# ═══════════════════════════════════════════════════════════════════════════
print(sep)
print("6.  RQ1 TASK×YEAR HEATMAP — readability issues")
print(sep)
yrs = sorted(df["year"].unique())
print(f"Full year range in data: {yrs[0]}–{yrs[-1]} ({len(yrs)} columns)")
print()
print("With 70+ year columns the heatmap is illegible.")
print("Recommended approach:")
print("  • Restrict heatmap to 2010+ (or 2015+) where task variety is rich")
print("  • OR group into 5-year periods")
print("  • OR show two heatmaps: pre-2010 (grouped) and 2010-2025 (annual)")
print()
print("Most task-diverse years:")
top_yrs = df[df["year"] >= 2010].groupby("year")["inferred_tasks"].nunique().sort_values(ascending=False).head(10)
print(top_yrs.to_string())
print()


# ═══════════════════════════════════════════════════════════════════════════
# Summary of fixes needed
# ═══════════════════════════════════════════════════════════════════════════
print(sep)
print("SUMMARY OF FIXES")
print(sep)
print("helpers.py:")
print("  [A] classify_language: fix NaN handling (str(nan)='nan' bug)")
print("  [B] _extract_venue / INLG_WCODES: map pre-2020 W-codes to 'inlg'")
print()
print("analysis.ipynb / notebook:")
print("  [C] dual_axis_trend / all year-axis plots: label every 5th year only")
print("  [D] RQ1 heatmap: restrict to 2010+ or use 5-year buckets")
print()
print("output/results.csv  (manual edits):")
print("  [E] has_llm_judge = False for the", len(early), "pre-2023 papers listed above")
print("      (especially the 2017 adversarial-discriminator papers)")
