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

# 1. Load data and apply the override
df_res = pd.read_csv('../output/results.csv')
flt = pd.read_csv('../output/filtered_papers.csv', usecols=['id', 'humeval_count', 'autoeval_count'])
df = df_res.merge(flt.rename(columns={'id': 'paper_id'}), on='paper_id', how='left')
df['humeval_count'] = pd.to_numeric(df['humeval_count'], errors='coerce').fillna(0)

# We must also do the LLM judge auto logic to get has_true_llm_judge
_INSTRUCTION_RE = re.compile(
    r"gpt-4|gpt4|gpt-3[.]5|gpt3[.]5|chatgpt|text-davinci-00[23]"
    r"|claude-?[1-9]|claude [1-9]|gemini|palm-?2|bard|command-?r"
    r"|llama-?[23]|llama [23]|vicuna|alpaca|mistral|falcon|instructgpt",
    re.IGNORECASE,
)
_ML_JUDGE_RE = re.compile(
    r"\bgpt-?2\b|discriminator|adversarial.*classif|\bclip\b"
    r"|roberta.*classif|bert.*classif|infersent|reward[- ]model"
    r"|perplexit|nli[- ]model|trained[- ]classif",
    re.IGNORECASE,
)
def classify_llm_judge_type(s):
    if pd.isna(s) or not str(s).strip(): return "unknown"
    if _INSTRUCTION_RE.search(str(s)): return "instruction_llm"
    if _ML_JUDGE_RE.search(str(s)): return "ml_evaluator"
    return "unknown"

df['llm_judge_type_auto'] = df['llm_judge_model'].apply(classify_llm_judge_type)
df['llm_judge_type'] = df['llm_judge_type_auto']
df.loc[df["has_llm_judge"] & (df["llm_judge_type"] == "unknown"), "llm_judge_type"] = "instruction_llm"
df['has_true_llm_judge'] = df['has_llm_judge'] & (df['llm_judge_type'] == "instruction_llm")


# Apply humeval override exactly as in rq_analysis
METHOD_RES = {
    "likert": re.compile(
        r'\blikert\b'
        r'|\d[\s-]?point\s+(scale|rating|likert)'
        r'|scale\s+of\s+\d[^\.]*?to\s+\d'
        r'|\d[\s-]to[\s-]\d\s+scale'
        r'|\b\d(?:\.\d+)?\s*/\s*[57]\b',
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

_hum_candidates = df[(~df["has_human_eval"]) & (df["humeval_count"] > 2)]
for idx, row in _hum_candidates.iterrows():
    text = _read_paper_text(row["paper_id"])
    if text and any(r.search(text) for r in METHOD_RES.values()):
        df.at[idx, "has_human_eval"] = True


print("=== 1. Validation of LLM Judges ===")
llm_papers = df[df['has_true_llm_judge']]
val_re = re.compile(r'validat|manual(?:ly)?\s+check|human\s+check|manual(?:ly)?\s+eval|human(?:ly)?\s+eval|align\w*\s+with\s+human', re.IGNORECASE)

validated_count = 0
for idx, row in llm_papers.iterrows():
    text = _read_paper_text(row["paper_id"])
    if text and val_re.search(text):
        validated_count += 1

print(f"Total papers using instruction-LLM judge: {len(llm_papers)}")
print(f"Number of those explicitly mentioning validation/manual checking: {validated_count} ({validated_count/len(llm_papers)*100:.1f}%)")
print(f"Number also reporting human evaluation: {llm_papers['has_human_eval'].sum()} ({llm_papers['has_human_eval'].mean()*100:.1f}%)\n")


print("=== 2. Trend of Human Evaluation over time ===")
# Let's print HE rate by year from 2010 to 2025
he_trend = df[df['year'] >= 2010].groupby('year')['has_human_eval'].agg(['sum', 'count', 'mean'])
print(he_trend.to_string())
print("\n")


print("=== 3. Categories Regex Overfiring Sample ===")
# Extract 50 samples of sentences matched by the categories regex
cat_re = METHOD_RES['categories']
samples = []
he_papers = df[df['has_human_eval']]

for idx, row in he_papers.iterrows():
    text = _read_paper_text(row["paper_id"])
    if text:
        # Simple sentence tokenizer
        sentences = [s.strip() for s in text.replace('\n', ' ').split('. ')]
        for s in sentences:
            if cat_re.search(s):
                samples.append(f"[{row['paper_id']}] {s}.")
                break
    if len(samples) >= 50:
        break

for i, s in enumerate(samples):
    print(f"{i+1}. {s}\n")

