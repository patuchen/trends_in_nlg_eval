import pandas as pd
import numpy as np
import os
import re
import ast

RESULTS_CSV = "../output/results.csv"

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

# Explode tasks
df["task_list"] = df["inferred_tasks"].fillna("").str.split("|")
df_tasks = df.explode("task_list").rename(columns={"task_list": "task"})
df_tasks = df_tasks[df_tasks["task"].str.strip() != ""].copy()
df_tasks["task"] = df_tasks["task"].str.strip().str.lower()

# LLM judge type detection
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

df["llm_judge_type_auto"] = df["llm_judge_model"].apply(classify_llm_judge_type)
df["llm_judge_type"] = df["llm_judge_type_auto"]

# Manual overrides
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
            def _norm_manual_judge_type(v):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return None
                s = str(v).strip().lower()
                if not s or s in {"na", "n/a", "none", "?"}:
                    return None
                if s in {"ml", "ml_evaluator", "mlevaluator"}:
                    return "ml_evaluator"
                if s in {"instruct", "instruction", "instruction_llm", "instruction-llm"}:
                    return "instruction_llm"
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
        print("Manual override error:", e)

# Treat remaining unknown as instruction_llm
df.loc[df["has_llm_judge"] & (df["llm_judge_type"] == "unknown"), "llm_judge_type"] = "instruction_llm"

df["has_true_llm_judge"] = df["has_llm_judge"] & (df["llm_judge_type"] == "instruction_llm")

# Map to exploded tasks
_m = df.set_index("paper_id")["has_true_llm_judge"]
df_tasks["has_true_llm_judge"] = df_tasks["paper_id"].map(_m)

# Count
g = df_tasks.groupby('task')['has_true_llm_judge'].agg(['sum', 'count'])
g['pct'] = g['sum'] / g['count'] * 100
print("\n=== True LLM Judge adoption by task ===")
print(g.sort_values(by='sum', ascending=False).head(20).to_string())
