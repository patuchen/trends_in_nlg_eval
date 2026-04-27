#!/usr/bin/env python3
"""Run Stage 3 LLM extraction for GPU 2 (second half of papers)."""
import sys
sys.path.insert(0, '.')

from pipeline.llm_runner import run_extraction

print("="*80)
print("GPU 2: Processing second 8,294 papers (lower signal strength)")
print("="*80)

run_extraction(
    filtered_csv='output/gpu2_papers.csv',
    txt_dir='txt_papers',
    output_jsonl='output/results_gpu2.jsonl',
    api_url='http://localhost:8000/v1',
    model='Qwen/Qwen2.5-14B-Instruct',
    max_papers=None
)

print("\nGPU 2 extraction complete!")
