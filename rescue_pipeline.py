"""
rescue_pipeline.py

Full-text task keyword scan + eval-signal regex pass for papers that were
dropped as no_task by the original pipeline.

Stages:
  1. For each no_task paper that has a txt file, run EXPANDED_TASKS keywords
     on the FULL paper body (before References section).
  2. For newly matched papers, run the eval-signal regexes (humeval, autoeval,
     llm_judge) -- same regexes as pipeline/filters.py.
  3. Keep only papers with at least one eval signal.
  4. Write output/rescued_results.csv in the same column layout as results.csv
     (boolean flags only; fine-grained LLM fields left as NaN).
  5. Print progress and summary statistics.

NOTE: Does NOT run LLM extraction. Fine-grained fields
      (human_eval_criteria, human_eval_methods, etc.) are left empty.
      extraction_method is set to "regex_only".

Run from REPO ROOT:
    python rescue_pipeline.py
"""

import re
import sys
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

TXT_DIR      = Path("txt_papers")
FILTERED_CSV = Path("output/filtered_papers.csv")
PAPERS_CSV   = Path("papers.csv")
RESULTS_CSV  = Path("output/results.csv")
OUT_CSV      = Path("output/rescued_results.csv")
PROGRESS_LOG = Path("output/rescue_pipeline_progress.log")

# ---------------------------------------------------------------------------
# Expanded task patterns (same as rescue_false_negatives.py but used on
# full text, so broader patterns are safer here too)
# ---------------------------------------------------------------------------
EXPANDED_TASKS = {
    "machine_translation": [
        r'machine\s+translation', r'\bMT\b', r'neural\s+translation',
        r'\btranslat(ion|ing|e)\b',
    ],
    "summarization": [
        r'(text\s+)?summarization', r'abstractive\s+summar',
        r'document\s+summar', r'extractive\s+summar',
    ],
    "data_to_text": [
        r'data[\s-]to[\s-]text', r'table[\s-]to[\s-]text',
        r'AMR[\s-]to[\s-]text', r'graph[\s-]to[\s-]text',
        r'knowledge[\s-]graph.*generat',
        r'structured\s+data.*generat', r'report\s+generat',
    ],
    "dialogue": [
        r'dialogue\s+(generation|system|response)',
        r'response\s+generation', r'open[\s-]domain\s+dialogue',
        r'conversational\s+(AI|model|system)',
        r'chatbot', r'task[\s-]oriented\s+dialogue',
    ],
    "question_generation": [
        r'question\s+generation', r'answer\s+generation',
        r'question[\s-]answer\s+pair\s+generat',
    ],
    "story_generation": [
        r'story\s+generation', r'narrative\s+generation',
        r'creative\s+(text\s+)?generation',
        r'(plot|story|fable|poem)\s+generat',
    ],
    "simplification": [
        r'text\s+simplification', r'sentence\s+simplification',
        r'lexical\s+simplification',
    ],
    "paraphrase": [
        r'paraphrase\s+generation', r'paraphras(ing|e)',
        r'back[\s-]translat',
    ],
    "captioning": [
        r'(image|video|chart|figure)\s+caption(ing|\s+generation)',
    ],
    "general_nlg": [
        r'natural\s+language\s+generation', r'\bNLG\b',
        r'\btext\s+generation\b', r'\blanguage\s+generation\b',
        r'open[\s-]ended\s+generat',
        r'long[\s-]form\s+generat',
        r'controlled\s+text\s+generat',
        r'constrained\s+generat',
    ],
    "code_generation": [
        r'code\s+generation', r'program\s+synthes',
        r'(source\s+)?code\s+synthes',
    ],
    "style_transfer": [
        r'style\s+transfer', r'sentiment\s+transfer',
        r'formality\s+(transfer|style)',
    ],
    # ---- new categories ----
    "question_answering": [
        r'open[\s-]domain\s+(question\s+answer|QA)',
        r'free[\s-]form\s+(question\s+answer|QA)',
        r'long[\s-]form\s+(question\s+answer|QA)',
        r'open[\s-]ended\s+(question\s+answer|QA)',
        r'abstractive\s+(question\s+answer|QA)',
        r'generative\s+(question\s+answer|QA)',
        r'reading\s+comprehension.*generat',
    ],
    "instruction_following": [
        r'instruction[\s-]follow(ing|er)',
        r'open[\s-]ended\s+instruction',
        r'instruction[\s-]tuned\s+generat',
        r'instruction[\s-]back[\s-]translat',
        r'following\s+(natural\s+language\s+)?instructions',
    ],
    "counterspeech": [
        r'counter[\s-]?speech(\s+generation)?',
        r'counter[\s-]narrative(\s+generation)?',
        r'hate\s+speech\s+response',
        r'counter(ing)?\s+hate\s+speech',
    ],
    "biography_generation": [
        r'biograph(y|ical)\s+generat',
        r'person\s+description\s+generat',
        r'bio\s+generat',
    ],
    "highlight_generation": [
        r'highlight\s+generat',
        r'key\s+point\s+generat',
        r'aspect[\s-]based\s+generat',
        r'review\s+highlight',
    ],
}

# Compile all patterns
_COMPILED = {
    task: [re.compile(p, re.IGNORECASE) for p in patterns]
    for task, patterns in EXPANDED_TASKS.items()
}


def match_tasks_fulltext(text: str) -> list:
    matched = []
    for task, patterns in _COMPILED.items():
        if any(p.search(text) for p in patterns):
            matched.append(task)
    return matched


# ---------------------------------------------------------------------------
# Eval signal regexes (from pipeline/filters.py)
# ---------------------------------------------------------------------------
humeval_regex = re.compile(
    r'('
    r'(human|manual)\s+evaluation|qualitative\s+analysis|'
    r'human\s+(rater|annotator|judge|evaluator|judgment|judgement)|'
    r'(rater|annotator)\s+agreement|'
    r'inter[\s-]?(annotator|rater)\s+agreement|'
    r'crowd[\s-]?sourc\w+\s+(annotation|evaluation)|'
    r'\bMTurk\b|Mechanical\s+Turk|Amazon\s+Turk|'
    r'preference\s+(study|evaluation|test|judgment|judgement)|'
    r'side[\s-]by[\s-]side\s+(evaluation|comparison)|'
    r'pairwise\s+(human\s+)?(evaluation|comparison|ranking)|'
    r'annotation\s+(study|campaign|task)|'
    r'(manual|human)\s+(error\s+)?analysis|'
    r'blind\s+annotation|error\s+annotation|'
    r'Direct\s+Assessment|\bDA\s+(evaluation|score|rating|protocol)\b|'
    r'Multidimensional\s+Quality\s+Metric|\bMQM\b|'
    r'Error\s+Span\s+Annotation|\bESA\b\s+(evaluation|protocol|annotation)|'
    r'Scalar\s+Quality\s+Metric|\bSQM\b|'
    r'Human[\s-]?Targeted\s+(Translation\s+)?Error\s+Rate|\bHTER\b|'
    r'post[\s-]edit(ing|ed)(\s+evaluation)?|\bMTPE\b|'
    r'adequacy[\s/]fluency\s+(evaluation|rating)|'
    r'\d[\s-]?point\s+(likert|scale|rating)|'
    r'likert[\s-]?(scale|type)|'
    r'binary\s+(evaluation|judgment|judgement|annotation)|'
    r'(best[\s-]worst|BWS)\s+scal'
    r')',
    re.IGNORECASE,
)

autoeval_regex = re.compile(
    r'('
    r'automatic\s+(evaluation|metric|scoring|assessment)|'
    r'reference[\s-]based\s+metric|'
    r'\b(BLEU[\s-]?\d*|SacreBLEU|ROUGE[\s-]?\d*L?|METEOR|chrF[\+\d]*|TER|WER|CIDEr|SPICE|NIST|RIBES)\b|'
    r'\b(BERTScore|BLEURT|MoverScore|BARTScore|COMET|COMET[\s-]?DA|'
    r'UniTE|PRISM|YiSi|NUBIA|BLANC|SummaQA|QuestEval|UniEval|'
    r'GPTScore|FED|USR|GRADE)\b|'
    r'\b(perplexity|bits[\s-]per[\s-](character|word))\b|'
    r'\b(SER|slot\s+error\s+rate)\b|'
    r'\b(accuracy|F1[\s-]score|exact\s+match|PARENT[\s-]?T?)\b'
    r')',
    re.IGNORECASE,
)

llm_judge_regex = re.compile(
    r'\bllm[-\s]*(as[-\s]*a[-\s]*judge|as[-\s]*an?[-\s]*evaluator|'
    r'based[-\s]*metric|metric|judge|evaluator)\b|'
    r'(GPT-?4|ChatGPT|claude|gemini)[\s-]*(as[\s-]*(a[\s-]*)?judge|'
    r'evaluation|evaluator|based\s+metric)|'
    r'model[\s-]based\s+(judge|evaluator)',
    re.IGNORECASE,
)

_REFS_RE = re.compile(
    r'\n(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY|'
    r'Appendix|APPENDIX|Appendices|APPENDICES)\n',
)


def read_paper_body(path: Path) -> str:
    """Read text file and strip from References onward."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        m = _REFS_RE.search(text)
        return text[:m.start()] if m else text
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 70)
    log.info("RESCUE PIPELINE -- full-text scan (no LLM)")
    log.info("=" * 70)

    # Load no_task paper IDs
    log.info("Loading output/filtered_papers.csv ...")
    flt = pd.read_csv(
        FILTERED_CSV,
        usecols=lambda c: c in {"id", "filter_drop_reason"},
        encoding="utf-8", on_bad_lines="skip", low_memory=False,
    )
    no_task_ids = set(
        flt[flt["filter_drop_reason"].astype(str).str.contains(
            "no_task", case=False, na=False
        )]["id"].astype(str)
    )
    log.info(f"  no_task papers: {len(no_task_ids):,}")

    # Load paper metadata (title + year) for output
    log.info("Loading papers.csv metadata ...")
    meta = pd.read_csv(
        PAPERS_CSV,
        usecols=lambda c: c in {"id", "name", "year"},
        encoding="utf-8", on_bad_lines="skip", low_memory=False,
    ).rename(columns={"name": "title"})
    meta_dict = meta.set_index("id")[["title", "year"]].to_dict("index")

    # Find available txt files for no_task papers
    available = [
        TXT_DIR / f"{pid}.pdf"
        for pid in no_task_ids
        if (TXT_DIR / f"{pid}.pdf").exists()
           and (TXT_DIR / f"{pid}.pdf").stat().st_size > 0
    ]
    log.info(f"  no_task papers with txt file: {len(available):,}")

    # Also load paper IDs already in results.csv to avoid duplicates
    existing_ids = set()
    if RESULTS_CSV.exists():
        existing = pd.read_csv(
            RESULTS_CSV, usecols=["paper_id"],
            encoding="utf-8", on_bad_lines="skip", low_memory=False,
        )
        existing_ids = set(existing["paper_id"].astype(str))
    log.info(f"  papers already in results.csv: {len(existing_ids):,}")
    if OUT_CSV.exists():
        prev = pd.read_csv(
            OUT_CSV, usecols=["paper_id"],
            encoding="utf-8", on_bad_lines="skip", low_memory=False,
        )
        existing_ids |= set(prev["paper_id"].astype(str))
    log.info(f"  papers already in rescued_results.csv: {len(existing_ids):,}")

    rows = []
    stats = {"scanned": 0, "task_match": 0, "eval_signal": 0, "skipped_existing": 0}

    for txt_path in tqdm(available, desc="Scanning"):
        pid = txt_path.stem  # removes .pdf
        stats["scanned"] += 1

        if pid in existing_ids:
            stats["skipped_existing"] += 1
            continue

        body = read_paper_body(txt_path)
        if not body:
            continue

        tasks = match_tasks_fulltext(body)
        if not tasks:
            continue
        stats["task_match"] += 1

        hc = len(humeval_regex.findall(body))
        ac = len(autoeval_regex.findall(body))
        lc = len(llm_judge_regex.findall(body))

        if hc == 0 and ac == 0 and lc == 0:
            continue
        stats["eval_signal"] += 1

        m = meta_dict.get(pid, {})
        rows.append({
            "paper_id":            pid,
            "title":               m.get("title", ""),
            "year":                m.get("year", ""),
            "venue":               "",          # filled by helpers._extract_venue
            "inferred_tasks":      "|".join(tasks),
            "extraction_method":   "regex_only",
            "parse_failed":        False,
            "haiku_spot_checked":  False,
            "haiku_correction":    False,
            "has_human_eval":      hc > 0,
            "human_eval_methods":  "",
            "human_eval_criteria": "",
            "human_eval_mt_methods": "",
            "num_annotators":      None,
            "num_items_rated":     None,
            "agreement_reported":  False,
            "agreement_metric":    "",
            "has_auto_eval":       ac > 0,
            "has_auto_eval_original": ac > 0,
            "autoeval_count":      ac,
            "auto_metrics":        "",
            "has_llm_judge":       lc > 0,
            "llm_judge_model":     "",
            "source_languages":    "",
            "target_languages":    "",
            "monolingual":         False,
            # raw counts for transparency
            "humeval_count":       hc,
            "llm_judge_count":     lc,
        })

    log.info("")
    log.info("=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)
    log.info(f"  Scanned txt files       : {stats['scanned']:,}")
    log.info(f"  Skipped (already in DB) : {stats['skipped_existing']:,}")
    log.info(f"  Task keyword match      : {stats['task_match']:,}")
    log.info(f"  Eval signal found       : {stats['eval_signal']:,}  -> added to output")

    if not rows:
        log.info("No new papers to add. Exiting.")
        return

    out_df = pd.DataFrame(rows)

    # Derive venue from paper_id
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent / "analysis"))
    try:
        from helpers import _extract_venue
        out_df["venue"] = out_df["paper_id"].apply(_extract_venue)
    except Exception:
        pass

    # Append to existing rescued_results.csv if present
    if OUT_CSV.exists():
        prev_df = pd.read_csv(OUT_CSV, encoding="utf-8", on_bad_lines="skip", low_memory=False)
        out_df = pd.concat([prev_df, out_df], ignore_index=True).drop_duplicates("paper_id")

    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    log.info(f"\nWritten {len(out_df):,} rescued papers to {OUT_CSV}")
    PROGRESS_LOG.write_text("done", encoding="utf-8")
    log.info("Progress marker written to output/rescue_pipeline_progress.log")


if __name__ == "__main__":
    main()
