#!/usr/bin/env python3
"""Run Stage 4 interactive spot checks."""

import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.haiku_checker import run_spot_checks

if __name__ == "__main__":
    # Run spot checks with 5% sample for better coverage
    stats = run_spot_checks(
        results_jsonl="output/results.jsonl",
        sample_fraction=0.05,  # 5% sample instead of 3%
        interactive=True
    )
    
    print(f"\n\n📋 Spot Check Summary")
    print(f"=" * 50)
    print(f"Total checked: {stats['total_checked']}")
    print(f"Confirmed: {stats['confirmed']}")
    print(f"Corrected: {stats['corrected']}")
    print(f"Skipped: {stats['skipped']}")
    if stats['corrected_ids']:
        print(f"\nCorrected Papers:")
        for pid in stats['corrected_ids']:
            print(f"  - {pid}")
