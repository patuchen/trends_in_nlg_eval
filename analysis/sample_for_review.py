"""
Produce a semicolon-delimited CSV of papers sampled for manual review.

Categories
----------
1  - Older papers from the earliest 30 years in the dataset
2  - Ruled out as surveys, proceedings, or overview papers
3  - Excluded because no generative task was detected
4  - Generative task detected but no evaluation regex matched
5  - Regex matched but LLM found no human or automatic evaluation
6  - LLM extracted evaluation information

Run from the analysis/ directory:
    python sample_for_review.py
"""

import re
import sys
import random
import json
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

random.seed(42)
N = 10  # target per category

FILTERED_CSV = "../output/filtered_papers.csv"
RESULTS_CSV  = "../output/results.csv"

# --------------------------------------------------------------------------
# Load data (read only needed columns to save memory)
# --------------------------------------------------------------------------
print("Loading filtered_papers.csv (needed columns only)...")
flt = pd.read_csv(
    FILTERED_CSV,
    encoding="utf-8",
    on_bad_lines="skip",
    usecols=lambda c: c in {
        "id", "URL", "name", "year",
        "passed_filter", "filter_drop_reason",
        "inferred_tasks", "humeval_count", "autoeval_count", "llm_judge_count",
    },
    low_memory=False,
)
flt = flt.rename(columns={"id": "paper_id", "name": "title", "URL": "url"})
flt["year"] = pd.to_numeric(flt["year"], errors="coerce")
flt["passed_filter"] = flt["passed_filter"].astype(str).str.lower().map(
    {"true": True, "false": False, "1": True, "0": False}
)
for col in ("humeval_count", "autoeval_count", "llm_judge_count"):
    if col in flt.columns:
        flt[col] = pd.to_numeric(flt[col], errors="coerce").fillna(0)

print(f"  {len(flt):,} rows loaded from filtered_papers.csv")

print("Loading results.csv...")
res = pd.read_csv(RESULTS_CSV, encoding="utf-8", on_bad_lines="skip", low_memory=False)
for col in ("has_human_eval", "has_auto_eval", "has_llm_judge", "parse_failed"):
    if col in res.columns:
        res[col] = res[col].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
for col in ("num_annotators", "num_items_rated"):
    if col in res.columns:
        res[col] = pd.to_numeric(res[col], errors="coerce")
print(f"  {len(res):,} rows loaded from results.csv")


# --------------------------------------------------------------------------
# Helper: build a URL from paper_id if the URL column is missing/NaN
# --------------------------------------------------------------------------
def make_url(row):
    url = str(row.get("url", "") or "").strip()
    if url and url.lower() not in ("nan", "none", ""):
        return url
    pid = str(row.get("paper_id", "")).strip()
    if pid:
        return f"https://aclanthology.org/{pid}"
    return ""


def sample_rows(df, n=N):
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=42)


def to_records(df, category_id, category_label, note_col=None, extra_cols=None):
    rows = []
    for _, r in df.iterrows():
        note_parts = []
        if note_col and note_col in r.index and pd.notna(r[note_col]):
            note_parts.append(f"{note_col}={r[note_col]}")
        if extra_cols:
            for c in extra_cols:
                if c in r.index and pd.notna(r[c]) and str(r[c]).strip():
                    note_parts.append(f"{c}={r[c]}")
        rows.append({
            "category_id": category_id,
            "category": category_label,
            "paper_id": r.get("paper_id", ""),
            "url": make_url(r),
            "title": str(r.get("title", "")).replace(";", ",").replace("\n", " "),
            "year": r.get("year", ""),
            "notes": " | ".join(note_parts),
        })
    return rows


# --------------------------------------------------------------------------
# Category 1: Oldest papers (earliest 30 unique years in filtered_papers)
# --------------------------------------------------------------------------
print("\n[1] Oldest papers...")
all_years = sorted(flt["year"].dropna().unique())
earliest_30_years = set(all_years[:30])
cat1_pool = flt[flt["year"].isin(earliest_30_years)].copy()
cat1 = sample_rows(cat1_pool, N)
records = to_records(cat1, 1, "oldest papers (earliest 30 years)", extra_cols=["filter_drop_reason"])
print(f"    pool={len(cat1_pool)}, sampled={len(cat1)}  years: {sorted(earliest_30_years)[:5]}...{sorted(earliest_30_years)[-3:]}")


# --------------------------------------------------------------------------
# Category 2: Ruled out as surveys / proceedings / meta-papers
# --------------------------------------------------------------------------
print("[2] Surveys and proceedings...")
cat2_pool = flt[
    flt["passed_filter"] == False
].copy()
# Keep only those with a 'meta_paper' drop reason
meta_mask = cat2_pool["filter_drop_reason"].astype(str).str.contains(
    "meta_paper|proceeding|survey|overview", case=False, na=False
)
cat2_pool = cat2_pool[meta_mask]
cat2 = sample_rows(cat2_pool, N)
records += to_records(cat2, 2, "ruled out: survey/proceedings/meta-paper", note_col="filter_drop_reason")
print(f"    pool={len(cat2_pool)}, sampled={len(cat2)}")


# --------------------------------------------------------------------------
# Category 3: Excluded because no generative task was detected
# --------------------------------------------------------------------------
print("[3] No generative task...")
cat3_pool = flt[
    flt["filter_drop_reason"].astype(str).str.contains("no_task", case=False, na=False)
].copy()
cat3 = sample_rows(cat3_pool, N)
records += to_records(cat3, 3, "excluded: no generative task detected", note_col="filter_drop_reason")
print(f"    pool={len(cat3_pool)}, sampled={len(cat3)}")


# --------------------------------------------------------------------------
# Category 4: Generative task found but no evaluation regex matched
# --------------------------------------------------------------------------
print("[4] Generative task but no eval regex...")
cat4_pool = flt[
    flt["filter_drop_reason"].astype(str).str.contains("no_eval_signal", case=False, na=False)
].copy()
cat4 = sample_rows(cat4_pool, N)
records += to_records(
    cat4, 4,
    "excluded: generative task found, no eval regex match",
    note_col="filter_drop_reason",
    extra_cols=["inferred_tasks"],
)
print(f"    pool={len(cat4_pool)}, sampled={len(cat4)}")


# --------------------------------------------------------------------------
# Category 5: Regex matched but LLM found no evaluation at all
# --------------------------------------------------------------------------
print("[5] Regex matched, LLM found no evaluation...")
cat5_pool = res[
    ~res["has_human_eval"] & ~res["has_auto_eval"] & ~res["has_llm_judge"]
].copy()
# Merge URL from filtered_papers if available
if "url" not in cat5_pool.columns:
    cat5_pool = cat5_pool.merge(
        flt[["paper_id", "url"]].drop_duplicates(), on="paper_id", how="left"
    )
cat5 = sample_rows(cat5_pool, N)
records += to_records(
    cat5, 5,
    "regex matched; LLM: no evaluation found",
    extra_cols=["inferred_tasks", "extraction_method", "parse_failed"],
)
print(f"    pool={len(cat5_pool)}, sampled={len(cat5)}")


# --------------------------------------------------------------------------
# Category 6: LLM extracted evaluation information
# Stratify across has_human_eval / has_auto_eval / has_llm_judge combinations
# --------------------------------------------------------------------------
print("[6] LLM extracted evaluation info (stratified)...")
cat6_pool = res[
    res["has_human_eval"] | res["has_auto_eval"] | res["has_llm_judge"]
].copy()
if "url" not in cat6_pool.columns:
    cat6_pool = cat6_pool.merge(
        flt[["paper_id", "url"]].drop_duplicates(), on="paper_id", how="left"
    )

# Build eval type label for stratification
def _etype(r):
    parts = []
    if r.get("has_human_eval"): parts.append("H")
    if r.get("has_auto_eval"):  parts.append("A")
    if r.get("has_llm_judge"):  parts.append("L")
    return "+".join(parts)

cat6_pool["_etype"] = cat6_pool.apply(_etype, axis=1)

# Sample ~2-3 per stratum up to N total
strata = cat6_pool["_etype"].unique()
n_per = max(1, N // len(strata))
parts = []
for s in strata:
    sub = cat6_pool[cat6_pool["_etype"] == s]
    parts.append(sample_rows(sub, n_per))
cat6 = pd.concat(parts).drop_duplicates("paper_id").head(N)

records += to_records(
    cat6, 6,
    "LLM extracted evaluation info",
    extra_cols=["has_human_eval", "has_auto_eval", "has_llm_judge", "inferred_tasks"],
)
print(f"    pool={len(cat6_pool)}, sampled={len(cat6)}  strata: {dict(cat6_pool['_etype'].value_counts())}")


# --------------------------------------------------------------------------
# Write output
# --------------------------------------------------------------------------
out_df = pd.DataFrame(records, columns=[
    "category_id", "category", "paper_id", "url", "title", "year", "notes"
])

output_path = "manual_review_sample.csv"
out_df.to_csv(output_path, sep=";", index=False, encoding="utf-8")
print(f"\nWritten {len(out_df)} rows to {output_path}")
print(out_df.groupby(["category_id", "category"]).size().to_string())
