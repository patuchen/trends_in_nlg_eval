import os
import re
import pandas as pd
from pathlib import Path

# Path setups
analysis_dir = os.path.dirname(os.path.abspath(__file__))
TXT_DIR = Path(os.path.join(analysis_dir, "../txt_papers"))
_REFS_RE = re.compile(r'\n(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY|Appendix|APPENDIX|Appendices|APPENDICES)\n')

def _read_paper_text(paper_id):
    p = TXT_DIR / f"{paper_id}.pdf"
    if p.exists() and p.stat().st_size > 0:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            m = _REFS_RE.search(text)
            return text[:m.start()] if m else text
        except Exception:
            return ""
    return ""

RESULTS_CSV = os.path.join(analysis_dir, "../output/results.csv")
RESCUED_CSV = os.path.join(analysis_dir, "../output/rescued_results.csv")

# 1. Load data
df = pd.read_csv(RESULTS_CSV, encoding="utf-8", on_bad_lines="skip", low_memory=False)

# Convert boolean cols
bool_cols = [
    "parse_failed", "haiku_spot_checked", "haiku_correction",
    "has_human_eval", "has_auto_eval", "has_llm_judge",
    "agreement_reported", "monolingual",
]
for col in bool_cols:
    if col in df.columns:
        df[col] = (
            df[col].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )

# Filter 2026 out
df = df[df["year"] < 2026].copy()

# Filter non-english
NON_ENGLISH_VENUES = {"jep", "recital", "tal", "taln", "rocling", "ijclclp", "ccl"}
df = df[~df["venue"].isin(NON_ENGLISH_VENUES)].copy()

# Incorporate filtered counts overrides
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

# Human eval override using METHOD_RES
METHOD_RES = {
    "likert": re.compile(r'\blikert\b|\d[\s-]?point\s+(scale|rating|likert)|scale\s+of\s+\d[^\.]*?to\s+\d|\d[\s-]to[\s-]\d\s+scale|\b\d(?:\.\d+)?\s*/\s*[57]\b', re.IGNORECASE),
    "ranking": re.compile(r'\branking\s+(evaluat|annot|study)|human\s+ranking|rank\s+(order\s+)?(the\s+output|outputs|the\s+response|the\s+translation|the\s+summar)|relative\s+ranking\s+of|pairwise\s+(comparison|preference|evaluat|judg)|side[\s-]by[\s-]side|forced[\s-]choice|\bA/B\s+test(ing)?|preference\s+(study|test|judg)', re.IGNORECASE),
    "categories": re.compile(r'binary\s+(annotation|evaluat|judg|rating|choice)|yes[\s/]no\s+(evaluat|judg|rating)|acceptability\s+judg|fluent\s+or\s+not|\bcategories:\s+[A-Z]|annotat\w*.{0,50}\bcategor\w*|annotat\w*\s+(?:in|into|as|with)\s+["\u201c\u2018]', re.IGNORECASE),
    "span": re.compile(r'span\s+annotation|error\s+(span|mark|annot)|span[\s-]?(level|based)\s+(evaluat|annot)|error\s+(highlight|tag)|\bMQM\b|Error\s+Span\s+Annotation', re.IGNORECASE),
    "best_worst_scaling": re.compile(r'best[\s-]worst\s+scal|\bBWS\b', re.IGNORECASE),
    "direct_assessment": re.compile(r'direct\s+assessment|\bDA\b\s+(evaluat|score|protocol|rating)|adequacy[\s/]fluency\s+rating', re.IGNORECASE),
    "post_editing": re.compile(r'post[\s-]?edit(ing|ed)?|postedit(ing|ed)?|human[\s-]?post[\s-]?edit(ing|ed)?|\bHTER\b', re.IGNORECASE),
}

_hum_candidates = df[(~df["has_human_eval"]) & (df["humeval_count"] > 2)]
for idx, row in _hum_candidates.iterrows():
    text = _read_paper_text(row["paper_id"])
    if text and any(r.search(text) for r in METHOD_RES.values()):
        df.at[idx, "has_human_eval"] = True

# Incorporate rescued results
if os.path.exists(RESCUED_CSV):
    rescued = pd.read_csv(RESCUED_CSV, encoding="utf-8", on_bad_lines="skip", low_memory=False)
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
    rescued = rescued[~rescued["paper_id"].isin(df["paper_id"])]
    df = pd.concat([df, rescued], ignore_index=True, sort=False)

# Classify LLM judge type
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
    if pd.isna(s) or not str(s).strip(): return "unknown"
    if _INSTRUCTION_RE.search(str(s)): return "instruction_llm"
    if _ML_JUDGE_RE.search(str(s)): return "ml_evaluator"
    return "unknown"

df["llm_judge_type_auto"] = df["llm_judge_model"].apply(classify_llm_judge_type)
df["llm_judge_type"] = df["llm_judge_type_auto"]

# Manual overrides if available
manual_judge_path = os.path.join(analysis_dir, "tables/rq2_unknown_llm_judge_papers.csv")
if os.path.exists(manual_judge_path):
    try:
        manual_tbl = pd.read_csv(manual_judge_path)
        manual_type_col = "type" if "type" in manual_tbl.columns else ("llm_judge_type" if "llm_judge_type" in manual_tbl.columns else None)
        if manual_type_col and "paper_id" in manual_tbl.columns:
            def _norm_manual_judge_type(v):
                if v is None or (isinstance(v, float) and np.isnan(v)): return None
                s = str(v).strip().lower()
                if not s or s in {"na", "n/a", "none", "?"}: return None
                if s in {"ml", "ml_evaluator", "mlevaluator"}: return "ml_evaluator"
                if s in {"instruct", "instruction", "instruction_llm", "instruction-llm"}: return "instruction_llm"
                return None
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
    except Exception as e:
        print("Manual override load failed:", e)

df.loc[df["has_llm_judge"] & (df["llm_judge_type"] == "unknown"), "llm_judge_type"] = "instruction_llm"
df["has_true_llm_judge"] = df["has_llm_judge"] & (df["llm_judge_type"] == "instruction_llm")

print("=== Validation of LLM Judges on Full Merged Corpus ===")
llm_papers = df[df['has_true_llm_judge']].copy()
val_re = re.compile(
    r'validat|manual(?:ly)?\s+check|human\s+check|manual(?:ly)?\s+eval|human(?:ly)?\s+eval|align\w*\s+with\s+human',
    re.IGNORECASE
)

validated_count = 0
validated_also_he_count = 0
for idx, row in llm_papers.iterrows():
    text = _read_paper_text(row["paper_id"])
    has_val = False
    if text and val_re.search(text):
        has_val = True
        validated_count += 1
        if row["has_human_eval"]:
            validated_also_he_count += 1

print(f"Total papers using instruction-LLM judge: {len(llm_papers)}")
print(f"Number of those explicitly mentioning validation/manual checking: {validated_count} ({validated_count/len(llm_papers)*100:.2f}%)")
print(f"Number also reporting human evaluation: {llm_papers['has_human_eval'].sum()} ({llm_papers['has_human_eval'].mean()*100:.2f}%)")

llm_papers_with_he = llm_papers[llm_papers["has_human_eval"]]
print(f"Number of those that also report human evaluation that mention validation: {validated_also_he_count} ({validated_also_he_count/len(llm_papers_with_he)*100:.2f}%)")
