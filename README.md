# Trends in NLG Evaluation: A Longitudinal Study of 24,291 Papers (1952–2025)

This repository contains the codebase, analytical scripts, and LaTeX manuscript files for our paper studying the longitudinal trends of evaluation modalities, criteria, and methodologies in Natural Language Generation (NLG) research.

## Directory Structure

*   `analysis/`: Main data analysis scripts, regressions, and verification tools.
    *   `rq_analysis.py`: Runs the complete research question (RQ) analysis pipeline, performs statistical significance tests (logistic regressions), and saves figures and tables.
    *   `verify_paper_stats.py`: Automated validation script checking the alignment between compiled statistics and LaTeX manuscript metrics.
    *   `topic_modeling_lda.py`: Template topic-modeling pipeline utilizing Latent Dirichlet Allocation (LDA) to group paper abstracts into thematic NLG tasks.
    *   `tables/`: Output directory for generated results in CSV format.
    *   `figures/`: Output directory for research figures (PDF format).
*   `pipeline/`: Raw regular-expression and metadata extraction scripts.
*   `paper_latex/`: LaTeX source files for the final camera-ready manuscript.
*   `output/`: Intermediate extraction artifacts and candidate text files.
*   `requirements.txt`: Python package dependencies.

## Installation & Setup

Set up a virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Analyses

### 1. Generating Figures and Tables
To run the full RQ analysis suite, execute:

```bash
python analysis/rq_analysis.py
```

This script will run logistic regressions, extract venue baseline comparison metrics, output statistics to `analysis/tables/`, and save figures to `analysis/figures/`.

### 2. Verifying LaTeX Numbers
To run the automated verification suite to confirm that the statistical claims made in the paper exactly align with the generated database tables:

```bash
python analysis/verify_paper_stats.py
```

### 3. Running Topic Modeling
To run Latent Dirichlet Allocation (LDA) over the abstracts of the corpus to explore the distribution of generative tasks:

```bash
python analysis/topic_modeling_lda.py
```

## Quality Assurance & Validation
Our regex-based extraction procedure was manually validated against a spreadsheet of 60 expert-annotated NLG papers. The extraction pipeline achieves **82.4% recall** after applying targeted override rules based on local citation text context, representing a reliable and conservative lower bound of actual NLG evaluation practices.
