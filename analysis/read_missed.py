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

keywords = re.compile(r'(.{0,60}(?:evaluat|annotat|scale|score|rating|human|judge).{0,60})', re.IGNORECASE)

with open('missed_snippets.txt', 'w', encoding='utf-8') as f:
    for pid in missed:
        f.write(f"\n--- {pid} ---\n")
        text = _read_paper_text(pid)
        matches = keywords.findall(text)
        for m in matches[:15]:
            f.write(f"  * {m.strip()}\n")
