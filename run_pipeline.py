"""
Main pipeline entry point.
Wires all stages together with CLI and simple stage skipping.
"""
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime

# Wire up stages
from pipeline.filters import load_papers, run_filter
from pipeline.llm_runner import run_extraction
from pipeline.assembler import assemble_outputs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(papers_csv: str = "papers.csv",
                 txt_dir: str = "txt_papers",
                 output_dir: str = "output",
                 api_url: str = "http://localhost:8000/v1",
                 model: str = "Qwen/Qwen2.5-14B-Instruct",
                 max_papers: int = None):
    """Run complete pipeline.
    
    Args:
        papers_csv: Path to papers CSV
        txt_dir: Path to text directory
        output_dir: Output directory
        api_url: vLLM API URL
        model: Model name
        max_papers: Limit papers processed (for testing)
    """
    
    logger.info("=" * 80)
    logger.info("NLG EVALUATION TREND ANALYSIS PIPELINE")
    logger.info(f"Started at {datetime.now()}")
    logger.info("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # ========================================================================
    # STAGE 1: Filter and regex matching
    # ========================================================================
    
    stage_start = time.time()
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 1: FILTERING AND REGEX MATCHING")
    logger.info("=" * 80)
    
    filtered_csv = output_path / "filtered_papers.csv"
    
    if filtered_csv.exists():
        logger.info(f"✓ Stage 1 output already exists: {filtered_csv}")
    else:
        logger.info("Loading papers...")
        papers_df = load_papers(papers_csv)
        
        logger.info(f"Running filter on {len(papers_df)} papers...")
        papers_df = run_filter(papers_df, txt_dir)
        
        logger.info(f"Stage 1 completed in {time.time() - stage_start:.1f}s")
    
    # ========================================================================
    # STAGE 2: Evaluation section extraction
    # ========================================================================
    
    # Note: This is integrated into Stage 3 (llm_runner.py calls the extractor)
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: EVALUATION SECTION EXTRACTION")
    logger.info("(Integrated into Stage 3)")
    logger.info("=" * 80)
    
    # ========================================================================
    # STAGE 3: LLM extraction
    # ========================================================================
    
    stage_start = time.time()
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3: LLM EXTRACTION")
    logger.info("=" * 80)
    
    results_jsonl = output_path / "results.jsonl"
    
    if results_jsonl.exists():
        logger.info(f"✓ Stage 3 output already exists: {results_jsonl}")
    else:
        logger.info("Extracting evaluation details with LLM...")
        logger.info(f"vLLM API: {api_url}")
        logger.info(f"Model: {model}")
        
        try:
            papers_processed = run_extraction(
                str(filtered_csv),
                txt_dir,
                str(results_jsonl),
                api_url=api_url,
                model=model,
                max_papers=max_papers
            )
            logger.info(f"Stage 3 completed in {time.time() - stage_start:.1f}s")
        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            raise
    
    # ========================================================================
    # STAGE 5: Output assembly
    # ========================================================================
    
    stage_start = time.time()
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 5: OUTPUT ASSEMBLY")
    logger.info("=" * 80)
    
    try:
        num_records, stats = assemble_outputs(str(results_jsonl), 
                                              str(output_path / "results.csv"))
        logger.info(f"Stage 5 completed in {time.time() - stage_start:.1f}s")
    except Exception as e:
        logger.error(f"Stage 5 failed: {e}")
        raise
    
    # ========================================================================
    # Pipeline complete
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to {output_dir}/")
    logger.info(f"  - filtered_papers.csv: Stage 1 filter results")
    logger.info(f"  - results.jsonl: Stage 3 LLM extraction (structured)")
    logger.info(f"  - results.csv: Stage 5 flattened results")
    logger.info("=" * 80)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NLG Evaluation Trend Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run full pipeline
  python run_pipeline.py --max-papers 100  # Test on 100 papers
  python run_pipeline.py --api-url http://gpu-node:8000/v1  # Custom vLLM server
        """
    )
    
    parser.add_argument(
        '--papers',
        default='papers.csv',
        help='Path to papers CSV (default: papers.csv)'
    )
    parser.add_argument(
        '--txt-dir',
        default='txt_papers',
        help='Path to text directory (default: txt_papers)'
    )
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory (default: output)'
    )
    parser.add_argument(
        '--api-url',
        default='http://localhost:8000/v1',
        help='vLLM API URL (default: http://localhost:8000/v1)'
    )
    parser.add_argument(
        '--model',
        default='Qwen/Qwen2.5-14B-Instruct',
        help='Model name (default: Qwen/Qwen2.5-14B-Instruct)'
    )
    parser.add_argument(
        '--max-papers',
        type=int,
        default=None,
        help='Maximum papers to process (for testing)'
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline(
            papers_csv=args.papers,
            txt_dir=args.txt_dir,
            output_dir=args.output_dir,
            api_url=args.api_url,
            model=args.model,
            max_papers=args.max_papers
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
