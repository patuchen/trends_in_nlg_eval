# Trends in NLG Evaluation: A Longitudinal Study of 24,291 Papers (1952–2025)

This repository contains the extraction pipeline, analytical scripts, and data for our paper studying the longitudinal trends of evaluation modalities, criteria, and methodologies in Natural Language Generation (NLG) research.

## Directory Structure

*   `run_pipeline.py`: Main entry point for the regular-expression and metadata extraction pipeline.
*   `pipeline/`: Core modules for data loading, filtering, and assembly logic.
*   `analysis/`: Main data analysis scripts, regressions, and plotting tools.
    *   `rq_analysis.py`: Runs the complete research question (RQ) analysis, performs statistical significance tests (logistic regressions), and saves figures and tables.
    *   `helpers.py`: Shared utilities (language classification, confidence intervals).
    *   `plot_rq4.py`: Specific visualization script for venue-group modality mix.
*   `output/`: Directory where intermediate and final extraction results are written:
    *   `filtered_papers.csv`: Candidate papers after initial filters.
    *   `results.csv`: Main extracted dataset, one row per paper.
    *   `rescued_results.csv`: Regex-extracted papers rescued from task filter false negatives.
*   `papers.csv`: Raw ACL Anthology metadata (93MB).
*   `txt_papers/`: Directory containing full-text representations of anthology papers (e.g., `{paper_id}.pdf`).
*   `requirements.txt`: Python package dependencies.

## Installation & Setup

Set up a virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Code

### 1. Running the Data Extraction Pipeline
To run the regex-based signal extraction pipeline to load raw metadata, filter papers, extract evaluation indicators, and assemble the dataset:

```bash
python run_pipeline.py
```

This generates `output/results.csv` containing the extracted evaluation indicators (human evaluation, automatic evaluation, and LLM-as-a-judge) for each paper.

### 2. Generating Figures and Tables (Analysis)
To perform the trend analysis, execute the logistic regressions, and generate all plots and tables used in the paper:

```bash
python analysis/rq_analysis.py
```

This script outputs statistical summaries, writes CSV tables to `analysis/tables/`, and generates figures (like the modality mix bar chart and criteria heatmaps) to `analysis/figures/`.

### 3. Verifying LaTeX Numbers (Optional)
To verify that the compiled statistics exactly align with the claims made in the manuscript:

```bash
python analysis/verify_paper_stats.py
```
