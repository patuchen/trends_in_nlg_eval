"""
Stage 4: Interactive Haiku spot checks of LLM extraction results.
Agent manually reviews sampled papers and confirms or corrects extracted records.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Set
import pandas as pd
import random

logger = logging.getLogger(__name__)


def load_results_jsonl(results_jsonl: str) -> list[Dict[str, Any]]:
    """Load all records from JSONL file.
    
    Args:
        results_jsonl: Path to results.jsonl
    
    Returns:
        List of record dictionaries
    """
    records = []
    with open(results_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_results_jsonl(records: list[Dict[str, Any]], results_jsonl: str):
    """Save records back to JSONL file.
    
    Args:
        records: List of record dictionaries
        results_jsonl: Path to results.jsonl
    """
    with open(results_jsonl, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')


def get_spot_check_candidates(records: list[Dict[str, Any]], 
                              sample_fraction: float = 0.03) -> Set[int]:
    """Select papers for spot checking.
    
    Strategy:
    1. Random sample of sample_fraction of all successfully processed papers
    2. ALL papers where parse_failed: true or reasoning contains hedging
    
    Args:
        records: List of records
        sample_fraction: Fraction of papers to randomly sample
    
    Returns:
        Set of record indices to spot check
    """
    candidates = set()
    
    # Find all successfully processed papers
    successful_indices = [i for i, r in enumerate(records) 
                         if not r.get('pipeline', {}).get('parse_failed', False)]
    
    # Random sample
    n_to_sample = max(1, int(len(successful_indices) * sample_fraction))
    random_sample = set(random.sample(successful_indices, min(n_to_sample, len(successful_indices))))
    candidates.update(random_sample)
    
    # ALL parse_failed papers + papers with hedging reasoning
    for i, record in enumerate(records):
        if record.get('pipeline', {}).get('parse_failed', False):
            candidates.add(i)
        
        # Check for hedging language in confirmation reasoning
        reasoning = record.get('_confirmation_reasoning', '')
        hedging_words = ['unclear', 'possibly', 'not sure', 'cannot determine', 'ambiguous']
        if any(word in reasoning.lower() for word in hedging_words):
            candidates.add(i)
    
    return candidates


def format_spot_check_block(record: Dict[str, Any], record_idx: int) -> str:
    """Format a single spot check block for review.
    
    Args:
        record: Record to review
        record_idx: Index in results
    
    Returns:
        Formatted string for display
    """
    block = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SPOT CHECK: {record['paper_id']:<55} ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Title: {record['title'][:76]:<76} ║
║ Year: {record['year']} | Venue: {record.get('venue', 'unknown'):<65} ║
╠══════════════════════════════════════════════════════════════════════════════╣

--- EXTRACTION TEXT (first 1500 chars) ---
{record.get('extraction_text', '')[:1500]}

--- EXTRACTED RECORD ---
{json.dumps({k: v for k, v in record.items() if k not in ['extraction_text', '_confirmation_reasoning']}, indent=2)}

--- TASK ---
Review the extracted record against the extraction text.
1. Type "confirm" to mark as correct
2. Type corrected JSON to replace the record
3. Type "skip" to leave unchanged

Respond only with one of: "confirm", "skip", or corrected JSON object
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    return block


def run_spot_checks(results_jsonl: str, 
                    sample_fraction: float = 0.03,
                    interactive: bool = True) -> Dict[str, Any]:
    """Run interactive spot checks on sampled papers.
    
    Args:
        results_jsonl: Path to results.jsonl
        sample_fraction: Fraction of papers to randomly sample
        interactive: Whether to pause for input (set to False for testing)
    
    Returns:
        Statistics dictionary
    """
    logger.info("=" * 80)
    logger.info("STAGE 4: INTERACTIVE HAIKU SPOT CHECKS")
    logger.info("=" * 80)
    
    # Load records
    logger.info(f"Loading results from {results_jsonl}...")
    records = load_results_jsonl(results_jsonl)
    logger.info(f"Loaded {len(records)} records")
    
    # Get candidates for spot checking
    candidates = get_spot_check_candidates(records, sample_fraction=sample_fraction)
    logger.info(f"Selected {len(candidates)} papers for spot checking")
    
    # Track corrections
    stats = {
        'total_checked': 0,
        'confirmed': 0,
        'corrected': 0,
        'skipped': 0,
        'corrected_ids': [],
    }
    
    # Process each candidate (sorted for consistency)
    for idx in sorted(candidates):
        record = records[idx]
        
        # Skip if already spot checked
        if record.get('pipeline', {}).get('haiku_spot_checked', False):
            logger.info(f"Skipping {record['paper_id']} (already spot checked)")
            continue
        
        stats['total_checked'] += 1
        
        # Print spot check block
        block = format_spot_check_block(record, idx)
        print(block)
        
        if not interactive:
            # In non-interactive mode, just confirm everything
            response = "confirm"
        else:
            # Get user input
            response = input("\nYour response: ").strip()
        
        # Process response
        if response.lower() == "confirm":
            record['pipeline']['haiku_spot_checked'] = True
            record['pipeline']['haiku_correction'] = False
            stats['confirmed'] += 1
            logger.info(f"✓ Confirmed {record['paper_id']}")
        
        elif response.lower() == "skip":
            logger.info(f"↷ Skipped {record['paper_id']}")
            stats['skipped'] += 1
        
        else:
            # Try to parse as corrected JSON
            try:
                corrected = json.loads(response)
                
                # Merge with original (preserve pipeline metadata)
                original_pipeline = record['pipeline']
                record.update(corrected)
                record['pipeline'] = original_pipeline
                record['pipeline']['haiku_spot_checked'] = True
                record['pipeline']['haiku_correction'] = True
                
                stats['corrected'] += 1
                stats['corrected_ids'].append(record['paper_id'])
                logger.info(f"✓ Corrected {record['paper_id']}")
                
            except json.JSONDecodeError:
                logger.warning(f"Could not parse response as JSON for {record['paper_id']}. Skipping.")
                stats['skipped'] += 1
        
        # Save after each record (incremental save)
        save_results_jsonl(records, results_jsonl)
    
    # Generate summary report
    report_lines = [
        "# Spot Check Report\n",
        f"## Summary\n",
        f"- Total papers spot-checked: {stats['total_checked']}\n",
        f"- Confirmed: {stats['confirmed']}\n",
        f"- Corrected: {stats['corrected']}\n",
        f"- Skipped: {stats['skipped']}\n",
        f"\n",
    ]
    
    if stats['corrected_ids']:
        report_lines.append(f"## Corrected Papers\n")
        for paper_id in stats['corrected_ids']:
            # Find the correction notes (if any)
            corrected_record = next((r for r in records if r['paper_id'] == paper_id), None)
            if corrected_record:
                report_lines.append(f"- {paper_id}\n")
        report_lines.append("\n")
    
    # Write report
    report_path = Path('output') / 'spot_check_report.txt'
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    
    logger.info("=" * 80)
    logger.info(f"Spot checks complete. Report saved to {report_path}")
    logger.info(f"  Confirmed: {stats['confirmed']}")
    logger.info(f"  Corrected: {stats['corrected']}")
    logger.info(f"  Skipped: {stats['skipped']}")
    logger.info("=" * 80)
    
    return stats
