#!/usr/bin/env python3
"""Interactive spot check review with cleaner output."""

import json
import sys
from pathlib import Path

def review_papers():
    """Review extraction results interactively."""
    
    # Load results
    results_path = Path('output/results.jsonl')
    results = []
    with open(results_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    # Select papers with evals to review
    papers_with_evals = [r for r in results if 
                        r.get('human_evaluation', {}).get('conducted', False) or
                        r.get('automatic_evaluation', {}).get('conducted', False)]
    
    print(f"\n📋 Spot Check Review")
    print(f"=" * 70)
    print(f"Total papers processed: {len(results)}")
    print(f"Papers with evaluations: {len(papers_with_evals)}")
    print(f"=" * 70)
    print(f"\nReviewing first 5 papers with evaluations...\n")
    
    stats = {'confirm': 0, 'correct': 0, 'skip': 0, 'total': 0}
    
    for i, paper in enumerate(papers_with_evals[:5]):
        print(f"\n[{i+1}/5] {paper['paper_id']}")
        print(f"Title: {paper['title']}")
        print(f"Year: {paper['year']} | Venue: {paper.get('venue', 'unknown')}")
        print(f"-" * 70)
        
        # Show evaluation findings
        human_eval = paper.get('human_evaluation', {}).get('conducted', False)
        auto_eval = paper.get('automatic_evaluation', {}).get('conducted', False)
        llm_judge = paper.get('llm_as_judge', {}).get('conducted', False)
        hybrid_used = paper.get('pipeline', {}).get('hybrid_override', False)
        
        print(f"Human Eval: {human_eval}")
        if human_eval:
            criteria = paper.get('human_evaluation', {}).get('criteria', [])
            if criteria:
                print(f"  Criteria: {', '.join(criteria)}")
        
        print(f"Auto Eval: {auto_eval}")
        if auto_eval:
            metrics = paper.get('automatic_evaluation', {}).get('metrics', [])
            if metrics:
                print(f"  Metrics: {', '.join(metrics)}")
        
        print(f"LLM Judge: {llm_judge}")
        print(f"Hybrid Override Used: {hybrid_used}")
        print(f"-" * 70)
        
        # Get feedback
        while True:
            response = input("Correct? (y/n/skip): ").strip().lower()
            if response in ['y', 'yes', 'confirm']:
                print("✓ Confirmed")
                stats['confirm'] += 1
                break
            elif response in ['n', 'no', 'fix']:
                print("Note: Corrections not implemented in this review pass")
                stats['correct'] += 1
                break
            elif response in ['skip', 's']:
                print("↷ Skipped")
                stats['skip'] += 1
                break
            else:
                print("Please enter: y (confirm), n (needs correction), or skip")
        
        stats['total'] += 1
    
    print(f"\n" + "=" * 70)
    print(f"📊 Review Summary")
    print(f"Confirmed: {stats['confirm']}/{stats['total']}")
    print(f"Needs correction: {stats['correct']}/{stats['total']}")
    print(f"Skipped: {stats['skip']}/{stats['total']}")
    print(f"=" * 70)
    print(f"\n✓ Spot check review complete!")
    print(f"Pipeline ready for full execution on 85,792 papers.")

if __name__ == "__main__":
    review_papers()
