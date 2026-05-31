import pandas as pd
import re
import sys
from pathlib import Path

_stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(_stdout_reconfigure):
    _stdout_reconfigure(encoding="utf-8")

TXT_DIR = Path('../txt_papers')
_REFS_RE = re.compile(r'\n(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY|Appendix|APPENDIX|Appendices|APPENDICES)\n')

def _read_paper_text(paper_id):
    p = TXT_DIR / f'{paper_id}.pdf'
    if p.exists() and p.stat().st_size > 0:
        try:
            text = p.read_text(encoding='utf-8', errors='replace')
            m = _REFS_RE.search(text)
            return text[:m.start()] if m else text
        except Exception: return ''
    return ''

df_res = pd.read_csv('../output/results.csv')
flt = pd.read_csv('../output/filtered_papers.csv', usecols=['id', 'humeval_count', 'autoeval_count'])
df = df_res.merge(flt.rename(columns={'id': 'paper_id'}), on='paper_id', how='left')

METHOD_RES = {
    "likert": re.compile(
        r'\blikert\b'
        r'|\d[\s-]?point\s+(scale|rating|likert)'
        r'|scale\s+of\s+\d[^\.]*?to\s+\d'
        r'|\d[\s-]to[\s-]\d\s+scale',
        re.IGNORECASE,
    ),
    "ranking": re.compile(
        r'\branking\s+(evaluat|annot|study)'
        r'|human\s+ranking'
        r'|rank\s+(order\s+)?(the\s+output|outputs|the\s+response|the\s+translation|the\s+summar)'
        r'|relative\s+ranking\s+of'
        r'|pairwise\s+(comparison|preference|evaluat|judg)'
        r'|side[\s-]by[\s-]side'
        r'|forced[\s-]choice'
        r'|\bA/B\s+test(ing)?'
        r'|preference\s+(study|test|judg)',
        re.IGNORECASE,
    ),
    "categories": re.compile(
        r'binary\s+(annotation|evaluat|judg|rating|choice)'
        r'|yes[\s/]no\s+(evaluat|judg|rating)'
        r'|acceptability\s+judg'
        r'|fluent\s+or\s+not'
        r'|\bcategories:\s+[A-Z]',
        re.IGNORECASE,
    ),
    "span": re.compile(
        r'span\s+annotation'
        r'|error\s+(span|mark|annot)'
        r'|span[\s-]?(level|based)\s+(evaluat|annot)'
        r'|error\s+(highlight|tag)'
        r'|\bMQM\b|Error\s+Span\s+Annotation',
        re.IGNORECASE,
    ),
    "best_worst_scaling": re.compile(
        r'best[\s-]worst\s+scal'
        r'|\bBWS\b',
        re.IGNORECASE,
    ),
    "direct_assessment": re.compile(
        r'direct\s+assessment'
        r'|\bDA\b\s+(evaluat|score|protocol|rating)'
        r'|adequacy[\s/]fluency\s+rating',
        re.IGNORECASE,
    ),
    "post_editing": re.compile(
        r'post[\s-]?edit(ing|ed)?'
        r'|postedit(ing|ed)?'
        r'|human[\s-]?post[\s-]?edit(ing|ed)?'
        r'|\bHTER\b',
        re.IGNORECASE,
    ),
}

# Load ground truth
df_gt = pd.read_excel('../Human Eval of Hallucinations Overview.xlsx', sheet_name='Detailed Annotations')
df_gt = df_gt.dropna(subset=['id'])
gt_ids = df_gt['id'].unique()

print(f"Number of annotated papers in gold set: {len(gt_ids)}")

# Find which ones are in our corpus
our_ids = df['paper_id'].unique()
gt_in_corpus = [pid for pid in gt_ids if pid in our_ids]
print(f"Number of gold papers in our corpus: {len(gt_in_corpus)}")

# Check recall
matched = 0
missed_papers = []

for pid in gt_in_corpus:
    # First check if humeval_count > 2. If not, it won't be overridden anyway.
    # Actually wait, the user's ground truth has SOME human evaluation.
    # If humeval_count <= 2, it would fail the override. Let's check both condition:
    row = df[df['paper_id'] == pid].iloc[0]
    
    # Let's check what the regexes find
    text = _read_paper_text(pid)
    text = text.replace('\n', ' ')
    
    found_any = False
    for method_name, method_re in METHOD_RES.items():
        if method_re.search(text):
            found_any = True
            break
            
    if found_any:
        matched += 1
    else:
        missed_papers.append(pid)

print(f"Regex Recall on Gold Set: {matched} / {len(gt_in_corpus)} ({matched/len(gt_in_corpus)*100:.1f}%)")

print("\nMissed Papers (No regex matched):")
for p in missed_papers[:10]:
    print(" -", p)
