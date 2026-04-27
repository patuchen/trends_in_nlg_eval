"""
Spot check Stage 1 false negatives.
Sample papers that were DROPPED and manually inspect them to see if they should have passed.
"""
import pandas as pd
from pathlib import Path
import json

def spot_check_false_negatives(filtered_csv: str = "output/filtered_papers.csv", sample_size: int = 10):
    """Spot check papers that failed filtering to identify false negatives."""
    
    print("=" * 80)
    print("STAGE 1 FALSE NEGATIVE SPOT CHECK")
    print("=" * 80)
    print(f"Sampling {sample_size} papers that FAILED filtering")
    print()
    
    df = pd.read_csv(filtered_csv)
    
    # Papers that failed filter
    failed = df[df['passed_filter'] == False]
    print(f"Total papers sampled: {len(df)}")
    print(f"Papers that passed filter: {(df['passed_filter']==True).sum()}")
    print(f"Papers that failed filter: {len(failed)}")
    print()
    
    # Sample failures by drop reason
    print("Drop reasons:")
    for reason, count in failed['filter_drop_reason'].value_counts().items():
        print(f"  - {reason}: {count}")
    print()
    
    # Sample diverse failures
    sample_failed = failed.sample(min(sample_size, len(failed)), random_state=42)
    
    print(f"Sampled {len(sample_failed)} failed papers for inspection:")
    print()
    
    for idx, (_, row) in enumerate(sample_failed.iterrows(), 1):
        paper_id = row['id']
        title = row['name'][:70]
        reason = row['filter_drop_reason']
        tasks_str = row.get('inferred_tasks', '[]')
        
        try:
            tasks = json.loads(tasks_str)
        except:
            tasks = []
        
        print(f"{idx}. {paper_id}")
        print(f"   Title: {title}...")
        print(f"   Why dropped: {reason}")
        print(f"   Inferred tasks: {tasks}")
        
        # Load paper text to check manually
        txt_path = Path("txt_papers") / f"{paper_id}.pdf"
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Check for evaluation keywords manually
            has_eval_keywords = any([
                'evaluation' in text.lower(),
                'metric' in text.lower(),
                'human' in text.lower(),
                'benchmark' in text.lower(),
            ])
            
            text_size = len(text)
            print(f"   Text size: {text_size} chars, Has eval keywords: {has_eval_keywords}")
        
        print()


if __name__ == "__main__":
    spot_check_false_negatives(sample_size=10)
