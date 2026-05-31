import pandas as pd
import re
from pathlib import Path

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

METHOD_RES = {
    'likert': re.compile(r'\blikert\b|\d[\s-]?point\s+(scale|rating|likert)|scale\s+of\s+\d|\d[\s-]to[\s-]\d\s+scale', re.I),
    'pairwise': re.compile(r'pairwise\s+(comparison|ranking|preference|evaluat|judg)|side[\s-]by[\s-]side|forced[\s-]choice|\bA/B\s+test|preference\s+(study|test|judg)', re.I),
    'categories': re.compile(r'binary\s+(annotation|evaluat|judg|rating|choice)|yes[\s/]no\s+(evaluat|judg|rating)|acceptability\s+judg|fluent\s+or\s+not|categories', re.I),
    'span': re.compile(r'span\s+annotation|error\s+(span|mark|annot)|span[\s-]?(level|based)\s+(evaluat|annot)|error\s+(highlight|tag)|\bMQM\b|\bESA\b', re.I),
    'best_worst_scaling': re.compile(r'best[\s-]worst\s+scal|\bBWS\b', re.I),
    'ranking': re.compile(r'\branking\s+(evaluat|task|annot|study)|(system|output|model)\s+ranking|rank\s+(order|the\s+output)', re.I),
    'direct_assessment': re.compile(r'direct\s+assessment|\bDA\b\s+(evaluat|score|protocol|rating)|adequacy[\s/]fluency\s+rating', re.I),
    'post_editing': re.compile(r'post[\s-]?edit(ing|ed)?|postedit(ing|ed)?|human[\s-]?post[\s-]?edit(ing|ed)?|\bHTER\b', re.I)
}

df_res = pd.read_csv('../output/results.csv')
flt = pd.read_csv('../output/filtered_papers.csv', usecols=['id', 'humeval_count', 'name'])
df = df_res.merge(flt.rename(columns={'id': 'paper_id'}), on='paper_id', how='left')
df['humeval_count'] = pd.to_numeric(df['humeval_count'], errors='coerce').fillna(0)

cands = df[(~df['has_human_eval']) & (df['humeval_count'] > 2)]
flipped = []
for idx, row in cands.iterrows():
    txt = _read_paper_text(row['paper_id'])
    if txt and any(r.search(txt) for r in METHOD_RES.values()):
        flipped.append(row)
        if len(flipped) >= 20: break

for i, r in enumerate(flipped):
    print(f"{i+1}. [{r['paper_id']}] {r['name']} (Year: {r.get('year', 'Unknown')})")
