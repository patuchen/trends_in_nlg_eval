"""
rescue_false_negatives.py

Re-scans papers filtered out as 'no_task' using expanded task keywords,
to identify false negatives missed by the original pipeline.

Run from REPO ROOT:
    python rescue_false_negatives.py
"""
import sys
import re
import pandas as pd
from typing import Dict, List

sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------------------
# EXPANDED_TASKS: all original patterns PLUS new patterns for known gaps
# ---------------------------------------------------------------------------

EXPANDED_TASKS: Dict[str, List[str]] = {
    # ---- original categories (verbatim from pipeline/filters.py) ----
    "machine_translation": [
        r'machine\s+translation', r'\bMT\b', r'neural\s+translation',
        r'\btranslat(ion|ing|e)\b',
    ],
    "summarization": [
        r'(text\s+)?summarization', r'abstractive\s+summar',
        r'document\s+summar',
    ],
    "data_to_text": [
        r'data[\s-]to[\s-]text', r'table[\s-]to[\s-]text',
        r'AMR[\s-]to[\s-]text', r'graph[\s-]to[\s-]text',
        r'knowledge[\s-]graph.*generat',
    ],
    "dialogue": [
        r'dialogue\s+(generation|system|response)',
        r'response\s+generation', r'open[\s-]domain\s+dialogue',
        r'conversational\s+(AI|model|system)',
    ],
    "question_generation": [
        r'question\s+generation', r'answer\s+generation',
    ],
    "story_generation": [
        r'story\s+generation', r'narrative\s+generation',
        r'creative\s+(text\s+)?generation',
    ],
    "simplification": [
        r'text\s+simplification', r'sentence\s+simplification',
    ],
    "paraphrase": [
        r'paraphrase\s+generation', r'paraphras(ing|e)',
    ],
    "captioning": [
        r'(image\s+)?caption(ing|\s+generation)',
    ],
    # general_nlg: original patterns + new open/long/controlled/constrained
    "general_nlg": [
        r'natural\s+language\s+generation', r'\bNLG\b',
        r'\btext\s+generation\b', r'\blanguage\s+generation\b',
        r'open[\s-]ended\s+generat',
        r'long[\s-]form\s+generat',
        r'controlled\s+text\s+generat',
        r'constrained\s+generat',
    ],
    "code_generation": [
        r'code\s+generation',
    ],
    "style_transfer": [
        r'style\s+transfer',
    ],

    # ---- NEW categories for gaps identified by manual review ----

    # Open-domain QA, free-form QA, long-form QA, abstractive QA, etc.
    "question_answering": [
        r'open[\s-]domain\s+(question\s+answer|QA)',
        r'free[\s-]form\s+(question\s+answer|QA)',
        r'long[\s-]form\s+(question\s+answer|QA)',
        r'open[\s-]ended\s+(question\s+answer|QA)',
        r'abstractive\s+(question\s+answer|QA)',
        r'reading\s+comprehension',
        r'generative\s+(question\s+answer|QA)',
    ],

    # Open-ended instruction following, instruction-tuned generation
    "instruction_following": [
        r'instruction[\s-]follow(ing)?',
        r'open[\s-]ended\s+instruction',
        r'instruction[\s-]tuned\s+generat',
        r'instruction[\s-]back[\s-]translat',
    ],

    # Counterspeech generation, counter-narrative generation, hate speech response
    "counterspeech": [
        r'counter[\s-]?speech(\s+generation)?',
        r'counter[\s-]narrative(\s+generation)?',
        r'hate\s+speech\s+response',
    ],

    # Biography generation (also partially covered by general_nlg)
    "biography_generation": [
        r'biography\s+generation',
        r'biographical\s+text',
        r'bio(graphy|graphical)\s+generat',
    ],

    # Highlight generation, key point generation, aspect-based generation
    "highlight_generation": [
        r'highlight\s+generation',
        r'key[\s-]?point\s+generation',
        r'aspect[\s-]based\s+generation',
        r'extractive[\s-]abstractive\s+generation',
    ],
}

# Gold false-negative paper IDs identified by manual review
GOLD_IDS = {
    "2024.emnlp-main.395",
    "2024.inlg-main.23",
    "2024.eacl-long.57",
    "2024.naacl-long.62",
    "2024.naacl-long.230",
    "2024.emnlp-main.451",
    "2022.inlg-main.7",
    "2024.emnlp-main.255",
    "2024.inlg-main.35",
    "2023.acl-long.307",
    "2022.acl-long.586",
}

# ---------------------------------------------------------------------------
# Task matching with EXPANDED_TASKS
# ---------------------------------------------------------------------------

def match_expanded_tasks(text: str) -> List[str]:
    """Return list of task names matched by EXPANDED_TASKS patterns."""
    matched = []
    for task_name, patterns in EXPANDED_TASKS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matched.append(task_name)
                break
    return matched


def match_new_tasks_only(text: str, original_tasks: List[str]) -> List[str]:
    """Return tasks matched by EXPANDED_TASKS that were NOT in original GENERATIVE_TASKS."""
    ORIGINAL_TASK_NAMES = {
        "machine_translation", "summarization", "data_to_text", "dialogue",
        "question_generation", "story_generation", "simplification", "paraphrase",
        "captioning", "general_nlg", "code_generation", "style_transfer",
    }
    all_matched = match_expanded_tasks(text)
    # A paper is rescued if it matches at least one category that is either:
    # - a brand new category (not in original set), OR
    # - general_nlg matched now but was NOT matched before
    # Since these are no_task papers, original_tasks is always empty, so
    # any match counts. We just report which tasks were newly matched.
    new_cats = [t for t in all_matched if t not in ORIGINAL_TASK_NAMES]
    orig_cats = [t for t in all_matched if t in ORIGINAL_TASK_NAMES]
    return all_matched, new_cats, orig_cats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("RESCUE FALSE NEGATIVES -- expanded task keyword scan")
    print("=" * 70)

    # Load filtered_papers.csv (only need id and filter_drop_reason)
    print("\nLoading output/filtered_papers.csv ...")
    filtered_df = pd.read_csv(
        "output/filtered_papers.csv",
        usecols=["id", "filter_drop_reason"],
        low_memory=False,
    )
    no_task_ids = set(
        filtered_df.loc[filtered_df["filter_drop_reason"] == "no_task", "id"]
    )
    print(f"  Total papers in filtered_papers.csv : {len(filtered_df)}")
    print(f"  Papers dropped as 'no_task'         : {len(no_task_ids)}")

    # Load papers.csv (id, name, year, abstract only -- file can be large)
    print("\nLoading papers.csv (id, name, year, abstract) ...")
    papers_df = pd.read_csv(
        "papers.csv",
        usecols=["id", "name", "year", "abstract"],
        low_memory=False,
    )
    print(f"  Total rows in papers.csv            : {len(papers_df)}")

    # Restrict to no_task papers
    no_task_df = papers_df[papers_df["id"].isin(no_task_ids)].copy()
    print(f"  no_task papers found in papers.csv  : {len(no_task_df)}")

    # Run expanded keyword matching
    print("\nRunning expanded keyword matching ...")
    rescued_rows = []

    for _, row in no_task_df.iterrows():
        paper_id = row["id"]
        title = str(row["name"]) if pd.notna(row["name"]) else ""
        abstract = str(row["abstract"]) if pd.notna(row["abstract"]) else ""
        year = row.get("year", None)
        text = f"{title} {abstract}"

        all_matched, new_cats, orig_cats = match_new_tasks_only(text, [])

        if all_matched:
            rescued_rows.append({
                "paper_id": paper_id,
                "title": title,
                "year": year,
                "new_tasks": "|".join(all_matched),
                "original_reason": "no_task",
            })

    rescued_df = pd.DataFrame(rescued_rows)

    # Save output
    output_path = "output/rescued_papers.csv"
    rescued_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nSaved rescued papers to {output_path}")

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Papers scanned (no_task)              : {len(no_task_df)}")
    print(f"Papers newly rescued                  : {len(rescued_df)}")

    if len(rescued_df) > 0:
        # Breakdown by task category
        from collections import Counter
        task_counts: Counter = Counter()
        for tasks_str in rescued_df["new_tasks"]:
            for t in tasks_str.split("|"):
                task_counts[t] += 1

        # Separate new vs original categories
        ORIGINAL_TASK_NAMES = {
            "machine_translation", "summarization", "data_to_text", "dialogue",
            "question_generation", "story_generation", "simplification", "paraphrase",
            "captioning", "general_nlg", "code_generation", "style_transfer",
        }
        NEW_TASK_NAMES = set(EXPANDED_TASKS.keys()) - ORIGINAL_TASK_NAMES

        print("\nBreakdown by task category (papers rescued, can overlap):")
        print(f"  {'Category':<30}  {'Count':>6}  {'Type'}")
        print(f"  {'-'*30}  {'-'*6}  {'-'*8}")
        for task in sorted(task_counts.keys()):
            cat_type = "NEW" if task in NEW_TASK_NAMES else "original-expanded"
            print(f"  {task:<30}  {task_counts[task]:>6}  {cat_type}")

        print("\nBreakdown for NEW categories only:")
        new_total = 0
        for task in sorted(NEW_TASK_NAMES):
            cnt = task_counts.get(task, 0)
            new_total += cnt
            print(f"  {task:<30}  {cnt:>6}")
        print(f"\n  Total rescues from NEW categories     : {new_total}")

    # ---------------------------------------------------------------------------
    # Gold false-negative check
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GOLD FALSE-NEGATIVE CHECK")
    print("=" * 70)
    print(f"Gold false-negative paper IDs ({len(GOLD_IDS)} total):")

    rescued_ids = set(rescued_df["paper_id"]) if len(rescued_df) > 0 else set()
    found_count = 0

    for gid in sorted(GOLD_IDS):
        if gid in rescued_ids:
            tasks_str = rescued_df.loc[rescued_df["paper_id"] == gid, "new_tasks"].values[0]
            status = f"RESCUED  [{tasks_str}]"
            found_count += 1
        elif gid in no_task_ids:
            status = "MISSED   (still not rescued by expanded patterns)"
        else:
            status = "NOT in no_task set (was not dropped as no_task)"
        print(f"  {gid:<35}  {status}")

    print(f"\nGold IDs rescued by expanded patterns : {found_count} / {len(GOLD_IDS)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
