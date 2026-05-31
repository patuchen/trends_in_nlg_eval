import os
import re

analysis_dir = os.path.dirname(os.path.abspath(__file__))
tables_dir = os.path.join(analysis_dir, "tables")
latex_path = os.path.join(analysis_dir, "../paper_latex/acl_latex.tex")

print("Checking table files...")
for fn in os.listdir(tables_dir):
    if fn.endswith(".csv"):
        print(f"  Found table: {fn}")

# Load LaTeX content
with open(latex_path, "r", encoding="utf-8") as f:
    latex_content = f.read()

def check_match(pattern, name):
    m = re.search(pattern, latex_content)
    if m:
        print(f"[OK] Found pattern for {name}: {m.group(0)}")
    else:
        print(f"[WARNING] Did NOT find pattern for {name} (expected pattern: {pattern})")

print("\n==================================================")
print("VERIFYING LATEX STATS AGAINST KNOWN VALUES")
print("==================================================")

# 1. Total papers
# Paper has: We analyse 24,291 papers
# Our filtering procedure yields a final analysis corpus of \textbf{24,291~papers}
check_match(r"24,291\s*papers", "Total Papers")

# 2. Human evaluation papers count
# Paper has: Among all 7,339 human-evaluated papers (outdated!)
# Let's see what is currently in LaTeX:
check_match(r"Among\s+all\s+\d+,\d+\s+human-evaluated\s+papers", "HE Papers Count (Section 3)")
check_match(r"Among\s+all\s+\d+,\d+\s+human-evaluated\s+papers", "HE Papers Count (Annotation Methods)")

# 3. Criteria rates (outdated in LaTeX: fluency 20.0%, relevance 21.5%, coherence 15.9%)
# New correct rates (from 8894 HE papers):
# fluency: 2002 / 8894 = 22.5%
# relevance: 1924 / 8894 = 21.6%
# coherence: 1479 / 8894 = 16.6%
check_match(r"fluency\s*\(\d+\.\d+\\\%\)", "fluency rate")
check_match(r"relevance\s*\(\d+\.\d+\\\%\)", "relevance rate")
check_match(r"coherence\s*\(\d+\.\d+\\\%\)", "coherence rate")

# 4. Method rates (outdated in LaTeX: Likert 17.0%, pairwise 9.6%, post-editing 8.8%)
# New correct rates (from 8894 HE papers):
# likert: 1693 / 8894 = 19.0%
# post_editing: 1403 / 8894 = 15.8%
# categories: 1235 / 8894 = 13.9%
# ranking: 1173 / 8894 = 13.2%
check_match(r"Likert-scale\s+rating\s+is\s+the\s+most\s+frequently\s+detected\s+annotation\s+approach\s*\(\d+\.\d+\\\%\)", "Likert rate")
check_match(r"pairwise\s+comparison\s*\(\d+\.\d+\\\%\)", "pairwise rate")
check_match(r"post-editing\s*\(\d+\.\d+\\\%\)", "post-editing rate")

# 5. Modality mix rates (outdated in LaTeX: auto-only 56.9%)
# New: 14051 + 130 + 94 = 14275 (out of 24291 total) -> wait, auto-only count is 14051. Wait, let's see.
# Let's print out what is in LaTeX:
check_match(r"Automatic-only\s+evaluation\s+is\s+by\s+far\s+the\s+most\s+common\s+modality\s*\(\d+\.\d+\\\%\)", "Auto-only rate")

# 6. Venue Group differences (outdated in LaTeX: Core NLP 66.7% auto-only, 26.6% human+auto; SIGGEN 43.9% auto-only, 33.5% human+auto, 10.4% human-only)
# Let's see what is currently in LaTeX:
check_match(r"Core\s+NLP\s+venues\s+are\s+the\s+most\s+automatic-only\s*\(\d+\.\d+\\\%\s*auto-only;\s*\d+\.\d+\\\%\s*human\+auto\)", "Core NLP Venues")
check_match(r"SIGGEN\s+venues\s+show\s+a\s+more\s+mixed\s+profile\s*\(\d+\.\d+\\\%\s*auto-only;\s*\d+\.\d+\\\%\s*human\+auto;\s*\d+\.\d+\\\%\s*human-only\)", "SIGGEN Venues")
