#!/usr/bin/env python3
"""Run Stage 3 LLM extraction for GPU 1 (first half of papers)."""
import sys
sys.path.insert(0, '.')

from pipeline.llm_runner import run_extraction

print("="*80)
print("GPU 1: Processing first 8,293 papers (high signal strength)")
print("="*80)

run_extraction(
    filtered_csv='output/gpu1_papers.csv',
    txt_dir='txt_papers',
    output_jsonl='output/results_gpu1.jsonl',
    api_url='http://localhost:8000/v1',
    model='Qwen/Qwen2.5-14B-Instruct',
    max_papers=None
)

print("\nGPU 1 extraction complete!")
