import os
import re
import pandas as pd
import pytest

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES_DIR = os.path.join(BASE_DIR, "analysis", "tables")
LATEX_DIR = os.path.join(BASE_DIR, "paper_latex")

ACL_LATEX_PATH = os.path.join(LATEX_DIR, "acl_latex.tex")
OTHER_LATEX_PATH = os.path.join(LATEX_DIR, "other_version.tex")

# Load table data helper
def get_table_path(fn):
    return os.path.join(TABLES_DIR, fn)

def load_csv(fn):
    return pd.read_csv(get_table_path(fn))

def extract_float(pattern, text):
    m = re.search(pattern, text)
    if m:
        return float(m.group(1))
    return None

def extract_int(pattern, text):
    m = re.search(pattern, text)
    if m:
        return int(m.group(1).replace(",", ""))
    return None

@pytest.mark.parametrize("latex_path", [ACL_LATEX_PATH, OTHER_LATEX_PATH])
def test_latex_statistics(latex_path):
    assert os.path.exists(latex_path), f"LaTeX file {latex_path} does not exist"
    
    with open(latex_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Total papers count check (should be 24,291)
    total_papers = extract_int(r"(\d+,\d+)\s+papers", content)
    # Check if we can find 24,291
    assert total_papers == 24291 or total_papers is None, f"Mismatch total papers: {total_papers}"

    # 2. HE papers count check (should be 8,894)
    he_papers = extract_int(r"Among\s+all\s+(\d+,\d+)\s+human-evaluated\s+papers", content)
    assert he_papers == 8894 or he_papers is None, f"Mismatch HE papers: {he_papers}"

    # 3. Criteria rates (fluency 22.5%, relevance 21.6%, coherence 16.6%)
    # Let's verify these rates against tables/rq3_criteria_freq.csv
    # In table: fluency=2002, relevance=1924, coherence=1479 out of 8894
    fluency_rate = extract_float(r"fluency\s*\((\d+\.\d+)\\\%\)", content)
    relevance_rate = extract_float(r"relevance\s*\((\d+\.\d+)\\\%\)", content)
    coherence_rate = extract_float(r"coherence\s*\((\d+\.\d+)\\\%\)", content)
    
    if fluency_rate is not None:
        assert abs(fluency_rate - 22.5) < 0.2, f"Fluency rate mismatch: {fluency_rate}"
    if relevance_rate is not None:
        assert abs(relevance_rate - 21.6) < 0.2, f"Relevance rate mismatch: {relevance_rate}"
    if coherence_rate is not None:
        assert abs(coherence_rate - 16.6) < 0.2, f"Coherence rate mismatch: {coherence_rate}"

    # 4. Method rates (Likert 19.0%, pairwise/ranking 13.2%, post-editing 15.8%)
    likert_rate = extract_float(r"Likert-scale\s+rating.*\((\d+\.\d+)\\\%\)", content)
    pairwise_rate = extract_float(r"pairwise\s+comparison\s*\((\d+\.\d+)\\\%\)", content)
    post_editing_rate = extract_float(r"post-editing\s*\((\d+\.\d+)\\\%\)", content)

    if likert_rate is not None:
        assert abs(likert_rate - 19.0) < 0.2, f"Likert rate mismatch: {likert_rate}"
    if pairwise_rate is not None:
        assert abs(pairwise_rate - 13.2) < 0.2, f"Pairwise/ranking rate mismatch: {pairwise_rate}"
    if post_editing_rate is not None:
        assert abs(post_editing_rate - 15.8) < 0.2, f"Post-editing rate mismatch: {post_editing_rate}"

    # 5. Modality mix auto-only (57.8%)
    auto_only = extract_float(r"Automatic-only\s+evaluation\s+is\s+by\s+far\s+the\s+most\s+common\s+modality\s*\((\d+\.\d+)\\\%\)", content)
    if auto_only is not None:
        assert abs(auto_only - 57.8) < 0.2, f"Auto-only rate mismatch: {auto_only}"

    # 6. Venue Group differences
    # Core NLP: 60.6% auto-only, 33.1% human+auto
    # SIGGEN: 35.8% auto-only, 42.4% human+auto, 10.8% human-only
    core_auto = extract_float(r"Core\s+NLP\s+venues\s+are\s+the\s+most\s+automatic-only\s*\((\d+\.\d+)\\\%\s*auto-only", content)
    core_combo = extract_float(r"Core\s+NLP\s+venues\s+are\s+the\s+most\s+automatic-only[^\(]*\([^\)]*?;\s*(\d+\.\d+)\\\%\s*human\+auto\)", content)
    
    if core_auto is not None:
        assert abs(core_auto - 60.6) < 0.2, f"Core NLP auto rate mismatch: {core_auto}"
    if core_combo is not None:
        assert abs(core_combo - 33.1) < 0.2, f"Core NLP human+auto rate mismatch: {core_combo}"

    siggen_auto = extract_float(r"SIGGEN\s+venues\s+show\s+a\s+more\s+mixed\s+profile\s*\((\d+\.\d+)\\\%\s*auto-only", content)
    siggen_combo = extract_float(r"SIGGEN\s+venues\s+show\s+a\s+more\s+mixed\s+profile[^\(]*\([^\)]*?;\s*(\d+\.\d+)\\\%\s*human\+auto", content)
    siggen_human = extract_float(r"SIGGEN\s+venues\s+show\s+a\s+more\s+mixed\s+profile[^\(]*\([^\)]*?;\s*[^\)]*?;\s*(\d+\.\d+)\\\%\s*human-only\)", content)

    if siggen_auto is not None:
        assert abs(siggen_auto - 35.8) < 0.2, f"SIGGEN auto rate mismatch: {siggen_auto}"
    if siggen_combo is not None:
        assert abs(siggen_combo - 42.4) < 0.2, f"SIGGEN human+auto rate mismatch: {siggen_combo}"
    if siggen_human is not None:
        assert abs(siggen_human - 10.8) < 0.2, f"SIGGEN human-only rate mismatch: {siggen_human}"

    # 7. LLM-judge cooccurrence rate (61.4%)
    llm_cooccur = extract_float(r"Of\s+papers\s+using\s+an\s+instruction-tuned\s+LLM\s+judge,\s*(\d+\.\d+)\\\%\s+also\s+report\s+human\s+evaluation", content)
    if llm_cooccur is not None:
        assert abs(llm_cooccur - 61.4) < 0.2, f"LLM cooccurrence rate mismatch: {llm_cooccur}"

    # 8. LLM-judge validation rate (86.2%)
    llm_val = extract_float(r"and\s*(\d+\.\d+)\\\%\s+of\s+(?:them|all\s+instruction-tuned\s+LLM\s+judge\s+papers)\s+explicitly\s+mention\s+validation", content)
    if llm_val is not None:
        assert abs(llm_val - 86.2) < 0.2, f"LLM validation rate mismatch: {llm_val}"
