# Methodological Decisions

This document records key decisions made during the analysis, with rationale.
It is intended as a reference for writing the paper.

---

## Scope

### What counts as an NLG task
In scope: machine translation, summarization, data-to-text, dialogue/response
generation, question generation, story/creative generation, text simplification,
paraphrase generation, image captioning, code generation, style transfer,
open-ended question answering (generative), instruction following,
counterspeech generation, highlight generation, biography generation.

Out of scope: text classification, information extraction, sentiment analysis,
named entity recognition, purely extractive QA, grammatical error correction,
factual error correction / text editing (the output is constrained to fix a
specific error, not generated freely), parsing tasks.

**Rationale for excluding text correction:** The task is to identify and fix a
specific error, not to generate novel text. The output space is tightly
constrained by the input in a way that distinguishes it from NLG.

### Venue coverage
- 1952-2021: ACL-OCL corpus (Hugging Face), all 215 ACL Anthology venues.
- 2022+: Own crawl, restricted to 9 venues: ACL, EMNLP, NAACL, EACL, AACL,
  IJCNLP, INLG, TACL, CL.
- Non-English venues excluded: JEP, RECITAL, TAL, TALN (French) and ROCLING,
  IJCLCLP, CCL (Chinese). The extraction prompt is English-language only.

**Implication for longitudinal claims:** Post-2021 data under-represents
smaller/specialised venues. Venue-level comparisons should note this asymmetry.

---

## Pipeline decisions

### Task filter: title+abstract vs full text
The original pipeline matched task keywords against title+abstract only.
A manual review of 70 expert-annotated papers found that ~69% of false
negatives had their task framing in the body text (e.g., hotel highlights,
data-to-text WebNLG, open-ended QA). A full-text rescue pass
(rescue_pipeline.py) was run on papers that have extracted text files,
using expanded keywords. The expanded set adds: question_answering,
instruction_following, counterspeech, biography_generation, highlight_generation.

**Rescued papers are included in the analysis with regex-only booleans (no LLM).**

### LLM extraction
Qwen2.5-14B-Instruct served locally via vLLM. Used for:
- Confirming whether each regex-matched eval category is genuinely present
  (filtering negated/prospective mentions).
- Extracting structured fields: criteria, methods, languages, judge model, etc.

Cross-check against 70 expert-annotated gold papers:
- Boolean fields (has_human_eval, has_auto_eval, has_llm_judge): reliable.
- Annotator counts (num_annotators, num_items_rated): recall ~18%. NOT USED
  for quantitative claims.
- Annotation method types: ~9% accuracy. NOT USED. Replaced by keyword-based
  detection from full text (see RQ7).

### has_auto_eval override
The LLM only sees the extracted evaluation section (potentially truncated).
If the regex found >= 2 autoeval matches in the full text but the LLM said
False, we override to True. Threshold: autoeval_count >= 2.
This corrects for truncation bias without re-running the LLM.
The original LLM judgment is preserved in has_auto_eval_original.

### LLM judge taxonomy
has_llm_judge=True in the raw extraction covers two distinct phenomena:
- instruction_llm: GPT-4, Claude, Gemini, Llama-2/3, Mistral, etc. used as
  evaluator via natural language prompting. This is the modern "LLM-as-judge"
  paradigm.
- ml_evaluator: GPT-2 perplexity scoring, RoBERTa/BERT discriminators, CLIP
  retrievers, adversarial classifiers. These use model internals, not
  instruction prompting.

For all trend claims about "LLM-as-judge adoption", we use has_true_llm_judge
(instruction_llm only), not has_llm_judge.

### Non-English venue exclusion
French and Chinese venue papers are excluded from the analysis corpus.
They represent ~41 papers. The English-language extraction prompt is not
appropriate for their content.

---

## Analysis decisions

### "other" criterion label
The criterion value "other" (and variants like "other -- please specify")
is excluded from criterion frequency counts and trend plots. It is a
survey template artifact, not a named evaluation criterion. We may note
the fraction of papers that use "other" as a catch-all, but do not treat
it as a substantive finding.

### Annotation method detection (RQ7)
Rather than relying on LLM-extracted human_eval_methods (low accuracy),
we use keyword-based regex on full paper text (from txt_papers/) with the
LLM-extracted field as a fallback when full text is unavailable.
Method categories: Likert, pairwise, binary, span annotation, best-worst
scaling, ranking, direct assessment, crowdsourcing.

### Scale fields dropped
num_annotators, num_items_rated, agreement_reported were extracted by LLM
but recall was ~18%. We do NOT report median/mean annotator counts or item
counts. We DO report the rate at which papers mention these numbers (i.e.,
how transparent they are), since the presence of a number is detectable
even when the number itself is wrong. IAA reporting rate is included as a
transparency indicator.

### Language classification
classify_language() in helpers.py classifies papers as:
english_only / non_english_mono / multilingual / unknown.

The original implementation had a NaN bug: str(np.nan) = 'nan' is truthy,
causing papers with missing language fields to be classified as multilingual.
Fixed by using pd.isna() and filtering 'nan' strings from all_langs.

Multilingual analysis focuses on non-MT multilingual papers, because MT is
by definition multilingual and dominates the multilingual bucket.

### Venue mapping for pre-2020 INLG/ENLG/SIG-GEN
Legacy W-prefix ACL Anthology IDs (e.g., W18-65) do not encode venue name.
~40 W-codes mapping to INLG/ENLG/SIG-GEN workshops are manually listed in
helpers._INLG_WCODES and mapped to the 'inlg' venue slug.
New-style IDs (2020+) encode venue directly.

### Year range
- Lower bound: no filter (full history from 1952 included).
- Upper bound: year < 2026 (exclude partial 2026 proceedings).
- For trend plots over the full range, x-axis labels only every-5th year to
  avoid overlap.

---

## Known limitations (for paper)

1. Task filter false negatives (~69% of gold false negatives missed by
   title/abstract keywords; full-text rescue partially addresses this).
2. Post-2021 venue coverage restricted to 9 venues.
3. LLM extraction of fine-grained fields (counts, method types) unreliable;
   only boolean fields used for quantitative analysis.
4. LLM judge taxonomy relies on model name regex; borderline cases (e.g.,
   GPT-2 used as evaluator in 2022) are ambiguous.
5. Language detection is based on LLM extraction of source/target language
   fields; errors propagate to multilingual analysis.
