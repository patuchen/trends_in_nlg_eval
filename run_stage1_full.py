"""
Run full Stage 1 filter on complete dataset.
This doesn't require GPU/LLM connection.
"""
import logging
import time
from datetime import datetime
from pathlib import Path

from pipeline.filters import load_papers, run_filter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("STAGE 1: FULL FILTER ON COMPLETE DATASET")
    logger.info(f"Started at {datetime.now()}")
    logger.info("=" * 80)
    
    stage_start = time.time()
    
    # Load all papers
    logger.info("Loading all papers from papers.csv...")
    papers_df = load_papers("papers.csv")
    
    # Run filter on complete dataset
    logger.info(f"Running filter on {len(papers_df)} papers...")
    papers_df = run_filter(papers_df, "txt_papers")
    
    # Save to output
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    filtered_csv = output_dir / "filtered_papers.csv"
    papers_df.to_csv(filtered_csv, index=False)
    
    elapsed = time.time() - stage_start
    logger.info(f"\n✓ Stage 1 completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    logger.info(f"✓ Saved filtered papers to {filtered_csv}")
    
    # Stats
    passed = (papers_df['passed_filter'] == True).sum()
    total = len(papers_df)
    logger.info(f"\nFinal stats:")
    logger.info(f"  Total papers: {total}")
    logger.info(f"  Papers passed filter: {passed} ({100*passed/total:.1f}%)")


if __name__ == "__main__":
    main()
