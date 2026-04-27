"""
Stage 5: Output assembly — flatten results.jsonl to results.csv with summary stats.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


def _to_pipe_string(value: Any) -> str:
    """Convert list-like/string/scalar values to a stable pipe-delimited string."""
    if value is None:
        return ''
    if isinstance(value, list):
        return '|'.join(str(item) for item in value)
    if isinstance(value, tuple):
        return '|'.join(str(item) for item in value)
    if isinstance(value, set):
        return '|'.join(str(item) for item in sorted(value))
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return ''
    return str(value)


def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a single record into flat CSV columns.
    
    Args:
        record: Single record from results.jsonl
    
    Returns:
        Flattened row dictionary
    """
    flatten = {
        'paper_id': record.get('paper_id', ''),
        'title': record.get('title', ''),
        'year': record.get('year', ''),
        'venue': record.get('venue', ''),
        'inferred_tasks': _to_pipe_string(record.get('inferred_tasks', [])),
        'extraction_method': record.get('pipeline', {}).get('extraction_method', ''),
        'parse_failed': record.get('pipeline', {}).get('parse_failed', False),
        'haiku_spot_checked': record.get('pipeline', {}).get('haiku_spot_checked', False),
        'haiku_correction': record.get('pipeline', {}).get('haiku_correction', False),
    }
    
    # Human evaluation
    human_eval = record.get('human_evaluation', {})
    if isinstance(human_eval, dict) and human_eval.get('conducted'):
        flatten['has_human_eval'] = True
        flatten['human_eval_methods'] = _to_pipe_string(human_eval.get('methods', []))
        flatten['human_eval_criteria'] = _to_pipe_string(human_eval.get('criteria', []))
        flatten['human_eval_mt_methods'] = _to_pipe_string(human_eval.get('mt_specific_methods', []))
        flatten['num_annotators'] = human_eval.get('num_annotators', '')
        flatten['num_items_rated'] = human_eval.get('num_items_rated', '')
        flatten['agreement_reported'] = human_eval.get('agreement_reported', False)
        flatten['agreement_metric'] = human_eval.get('agreement_metric', '')
    else:
        flatten['has_human_eval'] = False
        flatten['human_eval_methods'] = ''
        flatten['human_eval_criteria'] = ''
        flatten['human_eval_mt_methods'] = ''
        flatten['num_annotators'] = ''
        flatten['num_items_rated'] = ''
        flatten['agreement_reported'] = False
        flatten['agreement_metric'] = ''
    
    # Automatic evaluation
    auto_eval = record.get('automatic_evaluation', {})
    if isinstance(auto_eval, dict) and auto_eval.get('conducted'):
        flatten['has_auto_eval'] = True
        flatten['auto_metrics'] = _to_pipe_string(auto_eval.get('metrics', []))
    else:
        flatten['has_auto_eval'] = False
        flatten['auto_metrics'] = ''
    
    # LLM judge
    llm_judge = record.get('llm_as_judge', {})
    if isinstance(llm_judge, dict) and llm_judge.get('conducted'):
        flatten['has_llm_judge'] = True
        flatten['llm_judge_model'] = llm_judge.get('judge_model', '')
    else:
        flatten['has_llm_judge'] = False
        flatten['llm_judge_model'] = ''
    
    # Languages
    langs = record.get('languages', {})
    if isinstance(langs, dict):
        flatten['source_languages'] = _to_pipe_string(langs.get('source_languages', []))
        flatten['target_languages'] = _to_pipe_string(langs.get('target_languages', []))
        flatten['monolingual'] = langs.get('monolingual', False)
    else:
        flatten['source_languages'] = ''
        flatten['target_languages'] = ''
        flatten['monolingual'] = False
    
    return flatten


def assemble_outputs(results_jsonl: str, output_csv: str = "output/results.csv") -> Tuple[int, Dict[str, Any]]:
    """Assemble and flatten results into CSV with summary stats.
    
    Args:
        results_jsonl: Path to results.jsonl
        output_csv: Output CSV path
    
    Returns:
        Tuple of (num_records, summary_stats)
    """
    logger.info("=" * 80)
    logger.info("STAGE 5: OUTPUT ASSEMBLY")
    logger.info("=" * 80)
    
    # Load records
    logger.info(f"Loading results from {results_jsonl}...")
    records = []
    with open(results_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    logger.info(f"Loaded {len(records)} records")
    
    # Flatten records
    logger.info("Flattening records...")
    flat_rows = [flatten_record(r) for r in records]
    df = pd.DataFrame(flat_rows)
    
    # Save CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} records to {output_path}")
    
    # Compute summary statistics
    stats = compute_summary_stats(df, records)
    
    # Print summary
    print_summary_stats(stats, df)
    
    return len(records), stats


def compute_summary_stats(df: pd.DataFrame, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics.
    
    Args:
        df: Flattened dataframe
        records: Original records
    
    Returns:
        Statistics dictionary
    """
    stats = {}
    
    # Overall counts
    stats['total_papers'] = len(df)
    stats['papers_with_human_eval'] = df['has_human_eval'].sum()
    stats['papers_with_auto_eval'] = df['has_auto_eval'].sum()
    stats['papers_with_llm_judge'] = df['has_llm_judge'].sum()
    stats['papers_with_both_human_auto'] = (
        (df['has_human_eval']) & (df['has_auto_eval'])
    ).sum()
    stats['papers_with_neither'] = (
        (~df['has_human_eval']) & (~df['has_auto_eval']) & (~df['has_llm_judge'])
    ).sum()
    
    # Percentages
    total = stats['total_papers']
    if total > 0:
        stats['pct_human_eval'] = 100 * stats['papers_with_human_eval'] / total
        stats['pct_auto_eval'] = 100 * stats['papers_with_auto_eval'] / total
        stats['pct_llm_judge'] = 100 * stats['papers_with_llm_judge'] / total
        stats['pct_both_human_auto'] = 100 * stats['papers_with_both_human_auto'] / total
        stats['pct_neither'] = 100 * stats['papers_with_neither'] / total
    
    # Most common metrics
    metrics_count = defaultdict(int)
    for _, row in df[df['auto_metrics'] != ''].iterrows():
        for metric in str(row['auto_metrics']).split('|'):
            metrics_count[metric] += 1
    stats['top_metrics'] = sorted(metrics_count.items(), key=lambda x: -x[1])[:10]
    
    # Most common human eval criteria
    criteria_count = defaultdict(int)
    for _, row in df[df['human_eval_criteria'] != ''].iterrows():
        for crit in str(row['human_eval_criteria']).split('|'):
            criteria_count[crit] += 1
    stats['top_criteria'] = sorted(criteria_count.items(), key=lambda x: -x[1])[:10]
    
    # Breakdown by year
    year_stats = defaultdict(lambda: {'human': 0, 'auto': 0, 'llm': 0, 'both': 0, 'total': 0})
    for _, row in df.iterrows():
        year = int(row['year']) if pd.notna(row['year']) else None
        if year:
            year_stats[year]['total'] += 1
            if row['has_human_eval']:
                year_stats[year]['human'] += 1
            if row['has_auto_eval']:
                year_stats[year]['auto'] += 1
            if row['has_llm_judge']:
                year_stats[year]['llm'] += 1
            if row['has_human_eval'] and row['has_auto_eval']:
                year_stats[year]['both'] += 1
    stats['by_year'] = dict(year_stats)
    
    # Breakdown by venue
    venue_stats = defaultdict(lambda: {'human': 0, 'auto': 0, 'llm': 0, 'both': 0, 'total': 0})
    for _, row in df.iterrows():
        venue = str(row['venue']) if pd.notna(row['venue']) else 'unknown'
        venue_stats[venue]['total'] += 1
        if row['has_human_eval']:
            venue_stats[venue]['human'] += 1
        if row['has_auto_eval']:
            venue_stats[venue]['auto'] += 1
        if row['has_llm_judge']:
            venue_stats[venue]['llm'] += 1
        if row['has_human_eval'] and row['has_auto_eval']:
            venue_stats[venue]['both'] += 1
    stats['by_venue'] = dict(venue_stats)
    
    return stats


def print_summary_stats(stats: Dict[str, Any], df: pd.DataFrame):
    """Print summary statistics in formatted output.
    
    Args:
        stats: Statistics dictionary
        df: Flattened dataframe
    """
    total = stats['total_papers']
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\nOverall:")
    print(f"  Total papers:                {total}")
    print(f"  With human eval:             {stats['papers_with_human_eval']:4d} ({stats['pct_human_eval']:5.1f}%)")
    print(f"  With automatic metrics:      {stats['papers_with_auto_eval']:4d} ({stats['pct_auto_eval']:5.1f}%)")
    print(f"  With LLM judge:              {stats['papers_with_llm_judge']:4d} ({stats['pct_llm_judge']:5.1f}%)")
    print(f"  With both human + auto:      {stats['papers_with_both_human_auto']:4d} ({stats['pct_both_human_auto']:5.1f}%)")
    print(f"  With neither:                {stats['papers_with_neither']:4d} ({stats['pct_neither']:5.1f}%)")
    
    print(f"\nTop 10 Automatic Metrics:")
    for metric, count in stats['top_metrics']:
        print(f"  {metric:30s} {count:4d}")
    
    print(f"\nTop 10 Human Evaluation Criteria:")
    for crit, count in stats['top_criteria']:
        print(f"  {crit:30s} {count:4d}")
    
    print(f"\nBreakdown by Year:")
    for year in sorted(stats['by_year'].keys()):
        ys = stats['by_year'][year]
        print(f"  {year}: {ys['total']:4d} papers | "
              f"Human: {ys['human']:3d} | Auto: {ys['auto']:3d} | "
              f"LLM: {ys['llm']:3d} | Both: {ys['both']:3d}")
    
    print(f"\nBreakdown by Venue:")
    for venue in sorted(stats['by_venue'].keys()):
        vs = stats['by_venue'][venue]
        print(f"  {venue:15s}: {vs['total']:4d} papers | "
              f"Human: {vs['human']:3d} | Auto: {vs['auto']:3d} | "
              f"LLM: {vs['llm']:3d} | Both: {vs['both']:3d}")
    
    print("\n" + "=" * 80)
