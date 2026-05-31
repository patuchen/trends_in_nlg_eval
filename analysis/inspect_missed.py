import re
from pathlib import Path

TXT_DIR = Path('../txt_papers')
missed = ['2024.emnlp-main.188', '2023.inlg-main.1', '2023.inlg-main.30', '2023.emnlp-main.1007', '2023.emnlp-main.770']

def _read_paper_text(paper_id):
    p = TXT_DIR / f'{paper_id}.pdf'
    if p.exists() and p.stat().st_size > 0:
        try:
            return p.read_text(encoding='utf-8', errors='replace').replace('\n', ' ')
        except Exception: return ''
    return ''

keywords = re.compile(r'(.{0,100}(?:human eval|annotator|evaluator|scale|binary|score|rating).{0,100})', re.IGNORECASE)

for pid in missed:
    print(f"\n--- {pid} ---")
    text = _read_paper_text(pid)
    matches = keywords.findall(text)
    # just print first 5 matches to avoid huge output
    for m in matches[:10]:
        print("  *", m.strip())

