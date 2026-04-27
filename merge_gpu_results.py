#!/usr/bin/env python3
"""Merge GPU 1 and GPU 2 results into single results.jsonl."""
import json
from pathlib import Path
from tqdm import tqdm

print("="*80)
print("Merging GPU 1 and GPU 2 results...")
print("="*80)

output_dir = Path('output')
gpu1_file = output_dir / 'results_gpu1.jsonl'
gpu2_file = output_dir / 'results_gpu2.jsonl'
merged_file = output_dir / 'results.jsonl'

# Count lines
gpu1_count = sum(1 for _ in open(gpu1_file))
gpu2_count = sum(1 for _ in open(gpu2_file))

print(f"\nMerging:")
print(f"  GPU 1 results: {gpu1_count:,} papers")
print(f"  GPU 2 results: {gpu2_count:,} papers")
print(f"  Total: {gpu1_count + gpu2_count:,} papers")

# Merge both files
with open(merged_file, 'w') as out_f:
    line_count = 0
    
    # Write GPU 1
    with open(gpu1_file, 'r') as f:
        for line in tqdm(f, total=gpu1_count, desc="GPU1"):
            out_f.write(line)
            line_count += 1
    
    # Write GPU 2
    with open(gpu2_file, 'r') as f:
        for line in tqdm(f, total=gpu2_count, desc="GPU2"):
            out_f.write(line)
            line_count += 1

print(f"\n✓ Merged {line_count:,} papers to output/results.jsonl")

# Verify
with open(merged_file, 'r') as f:
    verified = sum(1 for _ in f)

print(f"✓ Verified: {verified:,} records in merged file")

# Quick stats
eval_count = 0
with open(merged_file, 'r') as f:
    for line in f:
        record = json.loads(line)
        if (record.get('human_evaluation', {}).get('conducted') or 
            record.get('automatic_evaluation', {}).get('conducted') or
            record.get('llm_as_judge', {}).get('conducted')):
            eval_count += 1

print(f"✓ Papers with evaluation detected: {eval_count:,} ({eval_count*100//verified}%)")
print(f"\n✓✓ Merge complete! Results ready in output/results.jsonl")
