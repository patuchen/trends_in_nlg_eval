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

METHOD_RES_OLD = {
    "likert": re.compile(r'\blikert\b|\d[\s-]?point\s+(scale|rating|likert)|scale\s+of\s+\d|\d[\s-]to[\s-]\d\s+scale', re.IGNORECASE),
    "pairwise": re.compile(r'pairwise\s+(comparison|ranking|preference|evaluat|judg)|side[\s-]by[\s-]side|forced[\s-]choice|\bA/B\s+test|preference\s+(study|test|judg)', re.IGNORECASE),
    "categories": re.compile(r'binary\s+(annotation|evaluat|judg|rating|choice)|yes[\s/]no\s+(evaluat|judg|rating)|acceptability\s+judg|fluent\s+or\s+not|categories', re.IGNORECASE),
    "span": re.compile(r'span\s+annotation|error\s+(span|mark|annot)|span[\s-]?(level|based)\s+(evaluat|annot)|error\s+(highlight|tag)|\bMQM\b|\bESA\b', re.IGNORECASE),
    "best_worst_scaling": re.compile(r'best[\s-]worst\s+scal|\bBWS\b', re.IGNORECASE),
    "ranking": re.compile(r'\branking\s+(evaluat|task|annot|study)|(system|output|model)\s+ranking|rank\s+(order|the\s+output)', re.IGNORECASE),
    "direct_assessment": re.compile(r'direct\s+assessment|\bDA\b\s+(evaluat|score|protocol|rating)|adequacy[\s/]fluency\s+rating', re.IGNORECASE),
    "post_editing": re.compile(r'post[\s-]?edit(ing|ed)?|postedit(ing|ed)?|human[\s-]?post[\s-]?edit(ing|ed)?|\bHTER\b', re.IGNORECASE),
}

METHOD_RES_NEW = {
    "likert": re.compile(
        r'\blikert\b'
        r'|\d[\s-]?point\s+(scale|rating|likert)'
        r'|scale\s+of\s+\d[^\.]*?to\s+\d'
        r'|\d[\s-]to[\s-]\d\s+scale'
        r'|\b\d(?:\.\d+)?\s*/\s*[57]\b',
        re.IGNORECASE,
    ),
    "ranking": re.compile( # pairwise merged here
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
        r'|\bcategories:\s+[A-Z]'
        r'|annotat\w*.{0,50}\bcategor\w*'
        r'|annotat\w*\s+(?:in|into|as|with)\s+["\u201c\u2018]',
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

our_ids = df['paper_id'].unique()
gt_in_corpus = [pid for pid in gt_ids if pid in our_ids]

old_matches = {}
new_matches = {}

for pid in gt_in_corpus:
    text = _read_paper_text(pid)
    if not text: continue
    text = text.replace('\n', ' ')
    
    old_found = []
    for method_name, method_re in METHOD_RES_OLD.items():
        if method_re.search(text):
            old_found.append(method_name)
    
    new_found = []
    for method_name, method_re in METHOD_RES_NEW.items():
        if method_re.search(text):
            new_found.append(method_name)
            
    old_matches[pid] = old_found
    new_matches[pid] = new_found

old_recall = sum(1 for v in old_matches.values() if len(v) > 0)
new_recall = sum(1 for v in new_matches.values() if len(v) > 0)

print(f"Old Recall: {old_recall} / {len(gt_in_corpus)} ({old_recall/len(gt_in_corpus)*100:.1f}%)")
print(f"New Recall: {new_recall} / {len(gt_in_corpus)} ({new_recall/len(gt_in_corpus)*100:.1f}%)")

print("\nPapers matched by OLD but NOT NEW:")
for pid in gt_in_corpus:
    if len(old_matches.get(pid, [])) > 0 and len(new_matches.get(pid, [])) == 0:
        print(f" - {pid} matched {old_matches[pid]}")
