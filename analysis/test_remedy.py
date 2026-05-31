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

# Let's test three new patterns
PATTERNS = {
    "likert_slash": re.compile(r'\b\d(?:\.\d+)?\s*/\s*[57]\b', re.IGNORECASE),
    "annotated_quotes": re.compile(r'annotated\s+(?:in|into|as|with)\s+["\u201c\u2018]', re.IGNORECASE),
    "annotated_categories": re.compile(r'annotat\w*.{0,50}\bcategor\w*', re.IGNORECASE)
}

df_sample = df.sample(frac=1, random_state=42)

for name, regex in PATTERNS.items():
    print(f"\n======================================")
    print(f"=== Sampling {name} ===")
    print(f"======================================")
    samples = []
    for idx, row in df_sample.iterrows():
        text = _read_paper_text(row["paper_id"])
        if text:
            sentences = [s.strip() for s in text.replace('\n', ' ').split('. ')]
            for s in sentences:
                if regex.search(s):
                    s_clean = re.sub(r'\s+', ' ', s)
                    samples.append(f"[{row['paper_id']}] {s_clean}.")
                    break
        if len(samples) >= 30:
            break
    for i, s in enumerate(samples):
        print(f"{i+1}. {s}")
