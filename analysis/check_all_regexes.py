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

df = pd.read_csv('../output/results.csv')

METHOD_RES = {
    "likert": re.compile(r'\blikert\b|\d[\s-]?point\s+(scale|rating|likert)|scale\s+of\s+\d|\d[\s-]to[\s-]\d\s+scale', re.IGNORECASE),
    "pairwise": re.compile(r'pairwise\s+(comparison|ranking|preference|evaluat|judg)|side[\s-]by[\s-]side|forced[\s-]choice|\bA/B\s+test|preference\s+(study|test|judg)', re.IGNORECASE),
    "categories": re.compile(r'binary\s+(annotation|evaluat|judg|rating|choice)|yes[\s/]no\s+(evaluat|judg|rating)|acceptability\s+judg|fluent\s+or\s+not|categories', re.IGNORECASE),
    "span": re.compile(r'span\s+annotation|error\s+(span|mark|annot)|span[\s-]?(level|based)\s+(evaluat|annot)|error\s+(highlight|tag)|\bMQM\b|\bESA\b', re.IGNORECASE),
    "best_worst_scaling": re.compile(r'best[\s-]worst\s+scal|\bBWS\b', re.IGNORECASE),
    "ranking": re.compile(r'\branking\s+(evaluat|task|annot|study)|(system|output|model)\s+ranking|rank\s+(order|the\s+output)', re.IGNORECASE),
    "direct_assessment": re.compile(r'direct\s+assessment|\bDA\b\s+(evaluat|score|protocol|rating)|adequacy[\s/]fluency\s+rating', re.IGNORECASE),
    "post_editing": re.compile(r'post[\s-]?edit(ing|ed)?|postedit(ing|ed)?|human[\s-]?post[\s-]?edit(ing|ed)?|\bHTER\b', re.IGNORECASE),
}

# Sample from HE papers for better chance of finding actual usage, but also some general to see overfiring.
# Since we want to see overfiring, we'll just sample randomly from the whole dataset.
df_sample = df.sample(frac=1, random_state=42)

for method_name, method_re in METHOD_RES.items():
    print(f"\n======================================")
    print(f"=== Sampling {method_name} ===")
    print(f"======================================")
    samples = []
    for idx, row in df_sample.iterrows():
        text = _read_paper_text(row["paper_id"])
        if text:
            sentences = [s.strip() for s in text.replace('\n', ' ').split('. ')]
            for s in sentences:
                if method_re.search(s):
                    # Clean up spacing for readability
                    s_clean = re.sub(r'\s+', ' ', s)
                    samples.append(f"[{row['paper_id']}] {s_clean}.")
                    break
        if len(samples) >= 100:
            break
    for i, s in enumerate(samples):
        print(f"{i+1}. {s}")
