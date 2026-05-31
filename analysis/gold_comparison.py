"""
gold_comparison.py -- evaluate pipeline against the hallucination HE gold set.

Gold set: "Human Eval of Hallucinations Overview.xlsx", sheet "Detailed Annotations".
  - 70 rows (some duplicate paper IDs = same paper annotated twice e.g. two evaluations)
  - AnnType: annotation method type used in the paper's hallucination evaluation
  - All papers evaluate hallucination -> we expect our "faithfulness" criterion detected

Questions answered:
  1. How many gold papers are in our corpus?
  2. Of those, how many have has_human_eval=True?
  3. For found HE papers: do our detected method types cover what the gold says?
  4. For found HE papers: is faithfulness/hallucination detected?

Run from the analysis/ directory:  python gold_comparison.py
"""

import os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))
from helpers import load_results, _extract_venue

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GOLD_PATH    = r"C:\Users\Lenovo\Downloads\Human Eval of Hallucinations Overview.xlsx"
RESULTS_JSONL = "../output/results.jsonl"
RESULTS_CSV   = "../output/results.csv"
RESCUED_CSV   = "../output/rescued_results.csv"

NON_ENGLISH_VENUES = {"jep", "recital", "tal", "taln", "rocling", "ijclclp", "ccl"}

# Pre-computed tables from rq_analysis.py (run that first)
METHODS_CSV  = "tables/rq7_annotation_methods.csv"
CRITERIA_CSV = "tables/rq3_criteria_kw.csv"

SEP = "=" * 70

# ---------------------------------------------------------------------------
# Load gold spreadsheet
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("Loading gold spreadsheet ...")
print(SEP)

gold_raw = pd.read_excel(GOLD_PATH, sheet_name="Detailed Annotations")
print(f"Gold rows (total): {len(gold_raw)}")
print(f"Gold unique paper IDs: {gold_raw['id'].nunique()}")
print(f"Applicable column: {gold_raw['Applicable'].value_counts(dropna=False).to_dict()}")

# Keep all rows (Applicable=NaN means applicable; only 1 "Maybe")
gold = gold_raw.copy()
# Normalise paper IDs
gold["paper_id"] = gold["id"].astype(str).str.strip()

print(f"\nAnnType distribution in gold:")
print(gold["AnnType"].value_counts().to_string())

# ---------------------------------------------------------------------------
# Map gold AnnType -> set of expected method labels (our taxonomy)
# Uses fuzzy matching so multi-word compound types are covered.
# ---------------------------------------------------------------------------
def anntype_to_methods(anntype):
    """Map a gold AnnType string to a set of method names from our taxonomy."""
    if pd.isna(anntype):
        return set()
    s = str(anntype).lower()
    if s in ("nm", "n/m", "not mentioned"):
        return set()
    methods = set()
    # likert / scale / continuous rating
    if re.search(r'likert|continuous\s+score|\d[\s-]?point|scale\s+of|\bscale\b', s):
        methods.add("likert")
    # span annotation
    if re.search(r'\bspan\b|\bmarking\b', s):
        methods.add("span")
    # pairwise / preference
    if re.search(r'pairwise|preference|a/b\s+test', s):
        methods.add("pairwise")
    # binary / categorization
    if re.search(r'binary|categoriz|categoris', s):
        methods.add("binary")
    # best-worst scaling
    if re.search(r'best[\s-]worst|bws', s):
        methods.add("best_worst_scaling")
    # ranking
    if re.search(r'\branking\b', s):
        methods.add("ranking")
    # percentage / count = none of our standard types
    return methods

gold["expected_methods"] = gold["AnnType"].apply(anntype_to_methods)
# Show mappings for verification
print("\nSample AnnType -> expected_methods mapping:")
sample_map = gold[["AnnType"]].assign(
    expected_methods=gold["expected_methods"].apply(str)
).drop_duplicates().head(20)
for _, r in sample_map.iterrows():
    print(f"  {str(r['AnnType']):<60s} -> {r['expected_methods']}")

# ---------------------------------------------------------------------------
# Load our analysis corpus
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("Loading analysis corpus ...")
print(SEP)

df, _ = load_results(RESULTS_JSONL, RESULTS_CSV)
df = df[df["year"] < 2026].copy()
df = df[~df["venue"].isin(NON_ENGLISH_VENUES)].copy()

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
        rescued["venue"] = rescued["paper_id"].apply(_extract_venue)
    rescued = rescued[rescued["year"] < 2026]
    rescued = rescued[~rescued["venue"].isin(NON_ENGLISH_VENUES)]
    rescued["has_auto_eval_original"] = rescued["has_auto_eval"]
    rescued = rescued[~rescued["paper_id"].isin(df["paper_id"])]
    df = pd.concat([df, rescued], ignore_index=True, sort=False)

print(f"Analysis corpus: {len(df):,} papers")

# Load pre-computed keyword tables
if not os.path.exists(METHODS_CSV) or not os.path.exists(CRITERIA_CSV):
    print("ERROR: keyword tables not found. Run rq_analysis.py first.")
    sys.exit(1)

he_methods  = pd.read_csv(METHODS_CSV)
he_criteria = pd.read_csv(CRITERIA_CSV)

method_cols = [c for c in he_methods.columns
               if c not in ("paper_id", "year", "source")]
crit_cols   = [c for c in he_criteria.columns
               if c not in ("paper_id", "year")]

# ---------------------------------------------------------------------------
# 1. Recall: how many gold papers are in our corpus?
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("1. Corpus recall for gold papers")
print(SEP)

gold_ids = gold["paper_id"].unique()
print(f"Unique gold paper IDs: {len(gold_ids)}")

in_corpus = set(gold_ids) & set(df["paper_id"])
print(f"Found in our corpus  : {len(in_corpus)} / {len(gold_ids)} "
      f"({len(in_corpus)/len(gold_ids)*100:.1f}%)")

missing = set(gold_ids) - in_corpus
if missing:
    print(f"Missing from corpus  : {sorted(missing)}")

# Of those in corpus, how many have has_human_eval=True?
df_gold = df[df["paper_id"].isin(in_corpus)].copy()
n_he = df_gold["has_human_eval"].sum()
print(f"\nOf {len(df_gold)} found papers, has_human_eval=True: {n_he} "
      f"({n_he/max(len(df_gold),1)*100:.1f}%)")

n_not_he = len(df_gold) - n_he
if n_not_he:
    print(f"has_human_eval=False : {n_not_he}")
    print("  (These gold papers do human evaluation but our pipeline missed them)")

# ---------------------------------------------------------------------------
# 2. Method matching: do our detected methods cover gold's AnnType?
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("2. Method detection recall vs gold AnnType")
print(SEP)

# Merge gold with our detected methods (paper level -- take union across rows)
gold_methods_expected = (
    gold.groupby("paper_id")["expected_methods"]
    .apply(lambda x: set().union(*x))
    .reset_index()
    .rename(columns={"expected_methods": "gold_methods"})
)

# Only look at HE papers found in corpus
methods_found = he_methods[he_methods["paper_id"].isin(in_corpus)].copy()

# For each paper: set of detected methods
methods_found["detected_methods"] = methods_found[method_cols].apply(
    lambda row: {m for m in method_cols if row[m] == 1}, axis=1
)

merged = gold_methods_expected.merge(
    methods_found[["paper_id", "detected_methods"]], on="paper_id", how="left"
)
merged["detected_methods"] = merged["detected_methods"].apply(
    lambda x: x if isinstance(x, set) else set()
)

# Exclude rows where gold has no mappable expected method (NM or unknown types)
has_expected = merged["gold_methods"].apply(len) > 0
print(f"Gold papers with mappable AnnType: {has_expected.sum()} / {len(merged)}")

comp = merged[has_expected].copy()
comp["covered"] = comp.apply(
    lambda r: len(r["gold_methods"] & r["detected_methods"]) > 0, axis=1
)
comp["exact"]   = comp.apply(
    lambda r: r["gold_methods"] == r["detected_methods"], axis=1
)
comp["over_detect"] = comp.apply(
    lambda r: r["detected_methods"] - r["gold_methods"], axis=1
)

n = len(comp)
n_covered = comp["covered"].sum()
print(f"\nRecall (gold method covered by ours): {n_covered} / {n} "
      f"({n_covered/max(n,1)*100:.1f}%)")
print(f"Exact match                         : {comp['exact'].sum()} / {n} "
      f"({comp['exact'].sum()/max(n,1)*100:.1f}%)")

print("\nPer-paper breakdown (gold_methods | detected_methods | covered):")
for _, row in comp.iterrows():
    mark = "OK" if row["covered"] else "MISS"
    over = f"  +{row['over_detect']}" if row["over_detect"] else ""
    print(f"  [{mark}] {row['paper_id']:<35s} "
          f"gold={row['gold_methods']}  det={row['detected_methods']}{over}")

# Cases where we detect MORE than gold (expected, since gold is hallucination-specific)
n_over = (comp["over_detect"].apply(len) > 0).sum()
print(f"\nPapers where we detect EXTRA methods beyond gold: {n_over} / {n}")
print("  (Expected -- gold annotated only hallucination HE; paper may have more)")

# ---------------------------------------------------------------------------
# 3. Faithfulness criterion detection for gold papers
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("3. Faithfulness criterion detection for gold papers")
print(SEP)
print("(All gold papers evaluate hallucination, so faithfulness should fire)")

crit_found = he_criteria[he_criteria["paper_id"].isin(in_corpus)].copy()
if "faithfulness" in crit_found.columns:
    n_faith = crit_found["faithfulness"].sum()
    print(f"faithfulness detected: {n_faith} / {len(crit_found)} "
          f"({n_faith/max(len(crit_found),1)*100:.1f}%)")
    # Papers where faithfulness NOT detected
    not_faith = crit_found[crit_found["faithfulness"] == 0]["paper_id"].tolist()
    if not_faith:
        print(f"  Not detected in: {not_faith}")
        # Show all criteria that WERE detected for those papers
        for pid in not_faith[:5]:
            row = crit_found[crit_found["paper_id"] == pid].iloc[0]
            detected = [c for c in crit_cols if row[c] == 1]
            print(f"    {pid}: detected={detected}")

# Summary table: gold papers, HE found, faithfulness detected
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
print(f"Gold papers (unique IDs)           : {len(gold_ids)}")
print(f"Found in corpus                    : {len(in_corpus)} "
      f"({len(in_corpus)/len(gold_ids)*100:.1f}%)")
print(f"Has HE flagged (of found)          : {n_he} "
      f"({n_he/max(len(df_gold),1)*100:.1f}%)")
if "faithfulness" in crit_found.columns and len(crit_found) > 0:
    print(f"Faithfulness criterion detected    : {n_faith} "
          f"({n_faith/max(len(crit_found),1)*100:.1f}% of found papers)")
if n > 0:
    print(f"Method type covered (recall)       : {n_covered}/{n} "
          f"({n_covered/n*100:.1f}%)")
