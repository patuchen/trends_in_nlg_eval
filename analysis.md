# NLG Evaluation Trend Analysis — Analysis Plan (v2)

This plan picks up from `output/results.jsonl` and `output/results.csv` produced by the pipeline.
It specifies how to answer each research question, including data preparation,
statistical methods, visualisations, and caveats.

## File structure

```
analysis/
├── helpers.py              # shared utilities: loaders, normalisers, plot helpers
├── analysis.ipynb          # single notebook, sections correspond to this plan
├── figures/                # all saved plots (PDF + PNG)
└── tables/                 # machine-readable summary tables (CSV)
```

One notebook, executed top-to-bottom. Intermediate discoveries (e.g. building the criteria
normalisation dictionary in the RQ3 section) are expected to reshape later sections. The
notebook should be re-runnable end-to-end once the normalisation dicts are finalised.

Throughout: for every temporal trend, report **both** absolute counts and relative proportions
(divided by total papers in that year-stratum). Where appropriate, also report proportions
relative to task-specific denominators (e.g. "% of MT papers using BLEU").

---

## helpers.py — shared utilities

This file should contain:

```python
# --- Data loading ---
def load_results(jsonl_path, csv_path):
    """Load both formats, return (df, raw_records)."""

# --- Normalisation ---
def build_norm_dict(series, manual_overrides=None):
    """
    Given a pipe-separated column, explode, lowercase, count frequencies,
    and return (freq_df, norm_dict_skeleton) where the skeleton maps every
    unique value to itself. Manual overrides are merged in.
    Use this to bootstrap criteria and metric normalisation empirically.
    """

def apply_norm(series, norm_dict):
    """Apply normalisation dict to a pipe-separated column, return cleaned column."""

# --- Language classification ---
def classify_language(row):
    """Classify into english_only / non_english_mono / multilingual / unknown."""
    src = str(row.get('source_languages', '') or '').strip()
    tgt = str(row.get('target_languages', '') or '').strip()
    mono = row.get('monolingual', False)
    all_langs = set(l.strip().lower() for l in (src + '|' + tgt).split('|') if l.strip())

    if not all_langs:
        return 'unknown'
    if mono and all_langs == {'english'}:
        return 'english_only'
    if mono and 'english' not in all_langs:
        return 'non_english_mono'
    if len(all_langs) == 1 and 'english' in all_langs:
        return 'english_only'
    return 'multilingual'

# --- Plotting ---
def dual_axis_trend(df, year_col, bool_col, title, ax=None):
    """
    Plot absolute count (bars) + proportion with 95% Wilson CI (line) per year.
    Standard pattern reused across all RQs.
    """

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion."""

def save_fig(fig, name):
    """Save to analysis/figures/{name}.pdf and .png at 300 DPI."""
```

---

## Section 0 — Data preparation and sanity checks

### 0A — Load and validate

```python
df = load_results('output/results.jsonl', 'output/results.csv')
```

Check and print:
- Total records
- % with `parse_failed = True` (should be <5%)
- Distribution of `extraction_method` (heading_heuristic vs. context_fallback)
- Year distribution histogram
- Papers per venue per year — heatmap or table; this IS the denominator for all
  proportions and must be visible to the reader

Papers where all three LLM-confirmed booleans are False (regex matched but LLM rejected
as false positive) are kept in the denominator but excluded from evaluation-specific
breakdowns. Report how many papers fall into this category.

### 0B — Flag meta-evaluation papers

Flag but do not exclude. Their presence can drive artefactual trends.

```python
META_EVAL_RE = re.compile(
    r'\bmeta[\s-]?evaluat|\bevaluat\w+\s+(metric|evaluation method)|'
    r'\bcorrelation\s+with\s+human\b|\bbenchmark\w*\s+(metric|evaluation)\b|'
    r'\bmetric\s+(comparison|benchmark|analysis)\b',
    re.IGNORECASE
)
df['is_meta_eval'] = df['title'].str.contains(META_EVAL_RE) | \
                     df['abstract'].str.contains(META_EVAL_RE)
```

Where meta-evaluation papers exceed 10% of a year or task stratum, note as caveat.

### 0C — Multi-task paper handling

Papers matching multiple generative tasks may not be representative of single-task papers.
A paper benchmarking one model across summarisation, MT, and data-to-text will likely
report benchmark scores without deep evaluation. A single-task paper is more likely to
run a tailored human evaluation with specific criteria.

```python
df['task_list'] = df['inferred_tasks'].str.split('|')
df['num_tasks'] = df['task_list'].apply(len)
df['is_multi_task'] = df['num_tasks'] > 1
```

Report:
- % of papers that are multi-task and distribution of `num_tasks` (1, 2, 3, 4+)
- Quick contingency table: multi-task vs. single-task × has_human_eval / has_auto_eval

Prepare exploded DataFrames:
```python
df_tasks = df.explode('task_list').rename(columns={'task_list': 'task'})
df_single = df[~df.is_multi_task].explode('task_list').rename(columns={'task_list': 'task'})
```

For task-stratified analyses, always provide both versions. Lead with `df_single` in the
main text; `df_tasks` (all papers) goes in appendix or is discussed as a sensitivity check.
If multi-task papers behave systematically differently (hypothesis: less human eval, fewer
criteria, more benchmark-style metrics), that is itself a finding worth reporting.

### 0D — Language categorisation

```python
df['lang_group'] = df.apply(classify_language, axis=1)
```

Categories: `english_only`, `non_english_mono`, `multilingual`, `unknown`.
Report distribution — expect heavy English skew. Papers with `unknown` language stay
in overall analyses but are excluded from language-stratified plots.

### 0E — Year handling

Use individual years throughout (2019–2025). If 2026 data is present, mark as incomplete
(only early-year venues represented). Exclude 2026 from trend fitting or show as dashed
line with no confidence interval.

### 0F — Venue grouping

```python
VENUE_GROUPS = {
    'generation': ['inlg'],
    'core_nlp': ['acl', 'emnlp', 'naacl', 'eacl', 'aacl', 'ijcnlp'],
    'journals': ['tacl', 'cl'],
}
```

### 0G — Build normalisation dictionaries (empirical)

This step is critical and must happen before any criteria or metric analysis.

For both `human_eval_criteria` and `auto_metrics`:
1. Explode the pipe-separated column
2. Lowercase all values, strip whitespace
3. Print frequency counts (top 100)
4. **Manually review and cluster** — look for synonyms, spelling variants, abbreviations
5. Optionally use sentence-transformer cosine similarity on the long tail to find
   near-duplicates that manual review missed
6. Build the final `CRITERIA_NORM` and `METRIC_NORM` dictionaries from this review
7. All entries that cannot be confidently mapped go into "other"

For metrics: collapse all variants into the base metric (ROUGE-1/2/L → ROUGE,
SacreBLEU → BLEU, chrF++ → chrF, etc.). The point of the paper is not arguing for
specific reproducibility variants but tracking evaluation practice broadly.

Do the same for `human_eval_methods` and `human_eval_mt_methods` — likely fewer
unique values but still worth inspecting.

---

## Section 1 — RQ1: Prevalence of human evaluation over time and across venues

**Operationalisation:** `has_human_eval = True` (LLM-confirmed).

### Analyses

1. **Overall time trend**
   - `dual_axis_trend`: absolute count (bars) + proportion with Wilson CI (line)
   - Denominator = all papers passing generative task filter in that year
   - Logistic regression: has_human_eval ~ year; report odds ratio and p-value
   - Annotate key events: ChatGPT (Nov 2022), GPT-4 (Mar 2023)

2. **By venue**
   - Small multiples: one line per venue group, same y-axis, both absolute and relative
   - Table: per venue, mean % human eval across full period

3. **By task × year heatmap**
   - Rows = tasks, columns = years, cells = % papers with human eval
   - Use `df_single`; provide `df_tasks` version in appendix

4. **By language group**
   - Bar chart: % human eval for english_only vs. non_english_mono vs. multilingual
   - Same split over time if sample sizes permit (likely too sparse for non-English)
   - Hypothesis: non-English and multilingual papers do less human eval due to annotator cost

**Statistical tests:**
- Chi-squared on pooled pre-2023 vs. post-2023 (simple overview)
- Logistic regression: has_human_eval ~ year + venue_group (controlled version)
- Report both; the logistic regression is the more appropriate model

**Caveats:**
- Papers with `extraction_method = context_fallback` may have lower recall; check stability
- Meta-eval papers may inflate rates in certain strata; note share where relevant

---

## Section 2 — RQ2: Adoption of LLM-as-a-judge

**Operationalisation:** `has_llm_judge = True` (LLM-confirmed).

### Analyses

1. **Emergence timeline**
   - Bar chart of absolute counts per year (absolute only for early years; add
     proportions from 2024 onward when counts are large enough for stable %)
   - Annotate first known use per venue (retrieve paper titles)

2. **Venue breakdown**
   - Stacked bar: per venue, proportion using LLM judge, by year
   - Which venue adopted earliest?

3. **Task breakdown**
   - First year of LLM-judge use per task; total count per task
   - Bar chart of adoption rate by task

4. **Which LLMs are used as judges?**
   - Frequency table of `llm_judge_model` (normalised)
   - Bar chart; expect GPT-4 dominance

5. **Co-occurrence with human eval**
   - Among LLM-judge papers: what % also conduct human eval? What % also use auto metrics?
   - Is LLM-judge a replacement for, or complement to, human eval?
   - Note: this may change over time — early papers probably validated LLM judges against
     human eval; later papers may use them standalone

6. **By language group**
   - Is LLM-as-judge used almost exclusively for English? (likely yes)
   - Report counts; if non-English LLM judge usage exists, highlight those papers

**Note:** Small counts expected. Use absolute counts for pre-2024 data points.
Avoid % with denominators < 20.

---

## Section 3 — RQ3: Trends in human evaluation criteria

**Operationalisation:** `human_eval_criteria` column (pipe-separated, normalised via 0G).

### Analyses

1. **Overall frequency ranking**
   - Horizontal bar chart of top 15–20 normalised criteria, sorted by frequency
   - Report both raw counts and % of human-eval papers mentioning each criterion

2. **Criteria over time**
   - For top 10 criteria: dual-axis plot per year
   - 3-year rolling average on proportions to smooth noise
   - Separate rising vs. declining criteria into subplots
   - Key question: are fluency and adequacy declining? Is faithfulness/factuality rising?

3. **Criteria by task**
   - Heatmap: rows = criteria, columns = tasks, cells = % of task's human-eval papers
   - Use `df_single` to avoid multi-task dilution
   - Are MT-specific criteria declining as neural MT matures?

4. **Criteria grouping over time**
   - Higher-level categories:
     - *Output quality*: fluency, grammaticality, naturalness
     - *Semantic fidelity*: adequacy, faithfulness, accuracy, consistency
     - *Discourse*: coherence, cohesion, structure
     - *Task-specific*: relevance, informativeness, engagingness
   - Stacked area chart of category proportions over time

---

## Section 4 — RQ3b: Trends in human evaluation methods

**Operationalisation:** `human_eval_methods` and `human_eval_mt_methods` columns
(pipe-separated, normalised via 0G).

This section is a companion to RQ3 and asks: not just *what* do we measure, but *how*?

### Analyses

1. **Overall method frequency**
   - Horizontal bar chart: % of human-eval papers using each method
   - Expected: likert_scale, ranking, pairwise_comparison, error_span_annotation,
     binary_classification, free_text

2. **Methods over time**
   - Dual-axis plot for each major method
   - Key question: is Likert declining? Is pairwise/ranking rising (possibly driven by
     LLM-judge frameworks that default to pairwise comparison)?

3. **MT-specific methods over time**
   - Restrict to papers with `task = machine_translation` (single-task only)
   - Line plot: DA vs. MQM vs. ESA adoption over time
   - Hypothesis: DA dominant until ~2021, MQM rises 2021–2023, ESA appears 2023+
   - This MUST be per-task to avoid contamination from non-MT papers using different
     terminology

4. **Methods × criteria co-occurrence**
   - Heatmap: rows = methods, columns = criteria, cells = % co-occurrence
   - Do Likert studies tend to measure fluency/adequacy? Do error annotation studies
     measure accuracy/terminology?

5. **Methods by task**
   - Heatmap: rows = methods, columns = tasks
   - Does data-to-text favour classification-based evaluation? Does dialogue favour ranking?

---

## Section 5 — RQ4: Evaluation modality distribution over time

**Operationalisation:** Mutually exclusive categories per paper. Handle LLM-judge overlap
with human/auto by treating it as an additional layer:

```python
def assign_eval_type(row):
    h, a, l = row['has_human_eval'], row['has_auto_eval'], row['has_llm_judge']
    if not h and not a and not l:
        return 'none'  # regex FP, LLM rejected
    parts = []
    if h: parts.append('human')
    if a: parts.append('auto')
    if l: parts.append('llm_judge')
    return '+'.join(parts)

df['eval_type'] = df.apply(assign_eval_type, axis=1)
```

This gives fine-grained categories (e.g. `human+auto`, `auto+llm_judge`, `human+auto+llm_judge`).
For the main visualisation, simplify to four groups:
- `human_only`: has_human_eval and not has_auto_eval (LLM judge may or may not be present)
- `auto_only`: has_auto_eval and not has_human_eval and not has_llm_judge
- `both_human_auto`: has_human_eval and has_auto_eval
- `llm_judge_no_human`: has_llm_judge and not has_human_eval

### Analyses

1. **Stacked area chart over time** (primary visualisation for the paper)
   - x = year, y = proportion of papers in each category
   - Report absolute counts in a companion table
   - This is publication-ready — iterate on colour palette

2. **By task**
   - Grouped bar chart: for each task, four bars showing eval_type distribution
   - Use `df_single`; are some tasks more likely to use human eval than others?

3. **By venue**
   - Same grouped bar chart with venues instead of tasks
   - INLG hypothesised to have higher human eval rates than EMNLP

4. **By language group**
   - Same breakdown: does multilingual work lean more heavily on auto-only?

5. **Multi-task vs. single-task comparison**
   - Side-by-side bar: eval_type distribution for multi-task vs. single-task papers
   - Hypothesis: multi-task papers are overwhelmingly auto_only

**Statistical tests:**
- Pairwise chi-squared between venues for human_only + both_human_auto vs. auto_only
- Same between tasks; apply Bonferroni correction
- Logistic regression: has_human_eval ~ year + venue + is_multi_task

---

## Section 6 — RQ5: Decline of BLEU and metric succession

**Operationalisation:** `auto_metrics` column (pipe-separated, normalised via 0G).
Explode to one row per metric per paper.

### Analyses

1. **BLEU over time**
   - Dual-axis: absolute count + % of auto-eval papers reporting BLEU, per year
   - Same line for ROUGE as comparison
   - Annotate year of BERTScore paper (2019), COMET (2020), BLEURT (2020)

2. **Top metrics per year**
   - For each year, compute top 5 metrics by % of auto-eval papers
   - Bump chart (rank over time) for the top 10 metrics across all years
   - This directly visualises metric succession

3. **Neural vs. lexical metric ratio**
   - Classify each normalised metric as:
     - *Lexical*: BLEU, ROUGE, METEOR, chrF, TER, NIST, CIDEr
     - *Neural/embedding*: BERTScore, BLEURT, COMET, MoverScore, BARTScore, UniEval, ...
     - *Task-specific*: SER, PARENT, exact match, F1, ...
   - Stacked area chart over time
   - Hypothesis: neural metrics grow post-2020, lexical decline post-2022

4. **Metrics by task**
   - Heatmap: rows = metrics, columns = tasks, cells = % of task's auto-eval papers
   - Expected: BLEU in MT, ROUGE in summarisation, SER/PARENT in data-to-text
   - Use `df_single`

5. **Metric diversity**
   - Per paper: count distinct metrics used
   - Box plot or violin plot of metric count per paper, by year
   - Are papers reporting more metrics now?

6. **Metrics by language group**
   - Does non-English work use different metrics? (e.g. chrF may be more common for
     morphologically rich languages where BLEU works poorly)

---

## Section 7 — RQ6: Task-based evaluation culture differences

This section synthesises findings from Sections 1–6 with task as the primary lens.
Use `df_single` throughout; provide `df_tasks` version as appendix sensitivity check.

### Analyses

1. **Human eval rate by task, over time**
   - Small multiples: one panel per task, dual-axis (absolute + %)
   - Highlight tasks where human eval is declining vs. stable vs. increasing

2. **Metrics by task, over time**
   - For top 4 tasks (MT, summarisation, data-to-text, dialogue): line chart of
     top 3 metrics' usage rates over time
   - Do MT papers drop BLEU earlier or later than summarisation papers?

3. **Evaluation criteria by task**
   - Reference the heatmap from Section 3

4. **Methods by task**
   - Reference the heatmap from Section 4

5. **LLM judge by task**
   - Reference the breakdown from Section 2

6. **Evaluation profile comparison table**
   Produce a **summary table** with one row per major task and these columns:
   - % human eval
   - % auto eval
   - % LLM judge
   - % both human + auto
   - median num_annotators (where reported)
   - median metric diversity (# distinct auto metrics per paper)
   - most common criterion
   - most common method

   This gives a compact "evaluation fingerprint" per task that is more precise and
   reproducible than a radar chart. Optionally, produce a **parallel coordinates plot**
   with normalised axes as a visual companion — but the table is the primary output.

7. **Multi-task vs. single-task divergence**
   - For each task: compare eval_type distribution for single-task papers vs.
     multi-task papers where that task is one of the matched tasks
   - Report differences; if multi-task papers consistently show less human eval,
     this is a finding about benchmark-driven evaluation practice

---

## Section 8 — RQ7: Scale of human evaluation over time

**Operationalisation:** `num_annotators` and `num_items_rated` columns. Both are often
missing (not reported). Handle carefully — do not treat missing as zero.

### Analyses

1. **Reporting rate**
   - % of human-eval papers that report `num_annotators` by year
   - % that report `num_items_rated` by year
   - Dual-axis: absolute count of papers reporting + proportion
   - Is reporting practice improving or deteriorating?

2. **Annotator counts over time** (only papers that report this)
   - Box plots per year of `num_annotators`
   - Median trend line overlay
   - Note that shared-task overview papers have annotator pools orders of magnitude
     larger than typical system papers; these are legitimate data points but worth
     highlighting as outliers in the distribution

3. **Items rated over time** (only papers that report this)
   - Same box plot approach
   - Caveat: "items rated" can mean sentences, documents, or system outputs; the LLM
     extraction may conflate these — report cautiously

4. **Annotator pool type over time**
   - Stacked bar: crowdsourced vs. expert vs. author vs. mixed, per year
   - Hypothesis: crowdsourcing (MTurk) peaked pre-2020 and has declined
   - By task: is crowdsourcing more common in dialogue than MT?

5. **Agreement reporting rate**
   - % of human-eval papers reporting inter-annotator agreement, per year
   - Breakdown by agreement metric used (Cohen's kappa, Krippendorff's alpha,
     Fleiss' kappa, Pearson's r)
   - Are papers becoming more or less rigorous?

6. **Missingness matrix**
   - For each of the 7 human eval metadata fields (methods, criteria, num_annotators,
     num_items, annotator_pool, agreement_reported, agreement_metric): what % of
     human-eval papers are missing each field?
   - Plot over time: has missingness decreased (i.e. are evaluation sections becoming
     more detailed and better reported)?
   - This is itself a finding about reporting practices in the field

---

## Section 9 — Summary figure and discussion scaffolding

### Summary composite figure

Produce a single publication-ready composite figure:

- Panel A: Stacked area chart — eval modality over time (from Section 5 / RQ4)
- Panel B: Criteria category trends over time (from Section 3 / RQ3)
- Panel C: Human eval rate by task (from Section 7 / RQ6)
- Panel D: LLM-judge adoption since 2023 (from Section 2 / RQ2)

Use a consistent colour palette across all panels. Export at 300 DPI as PDF and PNG.

### Discussion outline

The main story of the paper is retrospective: looking at how evaluation practice in NLG
has evolved over ~7 years, what we have gained, and what we may have lost. The discussion
should cover:

1. **What has improved?** Possible themes:
   - Adoption of neural metrics (BERTScore, COMET) that correlate better with human judgment
   - Increased metric diversity (papers report more metrics than before)
   - More standardised annotation protocols in MT (DA → MQM → ESA)

2. **What have we lost?** Possible themes:
   - Declining rate of human evaluation (if confirmed by RQ1)
   - Criteria that used to be measured and are now ignored (e.g. if fluency has declined
     because neural systems are assumed to be fluent, is that assumption justified?)
   - Fewer annotators, fewer rated items (if confirmed by RQ7)
   - Reduction in crowdsourced evaluation (less diverse annotator pools?)

3. **New concerns:**
   - LLM-as-judge adoption without validation against human judgments
   - Language bias: if LLM judges are English-only, multilingual NLG is under-evaluated
   - Multi-task papers using only benchmark metrics without task-specific evaluation

4. **Recommendations for the community:**
   - Which abandoned practices should be brought back? (In the spirit of slower, more
     deliberate science)
   - Minimum reporting standards for human evaluation sections (annotator count, pool type,
     agreement metric — the missingness analysis from RQ7 gives ammunition here)
   - When is LLM-as-judge appropriate vs. when does it need human validation?

5. **Comparison with prior surveys:**
   - Howcroft et al. (2020, INLG) — human evaluation in NLG
   - van der Lee et al. (2021) — best practices for NLG evaluation
   - The papers referenced in the initial prompts (INLG 2020 criteria paper,
     INLG 2024 metrics paper)
   - How do our findings update, confirm, or contradict their recommendations?

---

## Section 10 — Robustness checks

Run the four primary summary metrics on the following subsets:

| Subset | Description |
|--------|-------------|
| Full dataset | All papers |
| No meta-eval | `is_meta_eval = False` |
| Heading extraction only | `extraction_method = heading_heuristic` |
| No parse failures | `parse_failed = False` |
| INLG only | Most focused NLG venue |
| Single-task only | `is_multi_task = False` |

For each subset, recompute:
- % human eval
- % auto eval
- % BLEU usage among auto-eval papers
- % LLM judge

Report as a table in the appendix. If any subset shows substantially different trends
from the full dataset, discuss in the paper body.

### Pipeline quality estimate

Select ~15–20 papers manually (diverse across years, venues, tasks). For each, a human
(you) reads the paper and fills in the same schema fields. Compare against the pipeline
output to estimate per-field precision and recall. Report in a small table:

| Field | Precision | Recall | Notes |
|-------|-----------|--------|-------|
| has_human_eval | | | |
| has_auto_eval | | | |
| has_llm_judge | | | |
| criteria (top 3) | | | |
| metrics (top 3) | | | |
| methods | | | |
| num_annotators | | | |

Additionally, note whether the regex filter excluded any papers from the manual sample
that should have been included (recall of the regex stage). This was partially addressed
by the special review of regex-filtered papers during pipeline development.

---

## Outputs to produce

| File | Content |
|------|---------|
| `analysis/figures/rq1_human_eval_trend.pdf` | RQ1 main dual-axis trend |
| `analysis/figures/rq1_task_heatmap.pdf` | RQ1 task × year heatmap |
| `analysis/figures/rq2_llm_judge_timeline.pdf` | RQ2 emergence chart |
| `analysis/figures/rq3_criteria_trends.pdf` | RQ3 criteria over time |
| `analysis/figures/rq3_criteria_by_task.pdf` | RQ3 task × criteria heatmap |
| `analysis/figures/rq3b_methods_trends.pdf` | RQ3b methods over time |
| `analysis/figures/rq3b_mt_methods.pdf` | RQ3b DA vs MQM vs ESA |
| `analysis/figures/rq4_eval_modality.pdf` | RQ4 stacked area (main paper figure) |
| `analysis/figures/rq5_metric_bump.pdf` | RQ5 bump chart |
| `analysis/figures/rq5_neural_vs_lexical.pdf` | RQ5 stacked area |
| `analysis/figures/rq6_eval_profile_table.pdf` | RQ6 summary table (rendered) |
| `analysis/figures/rq7_annotator_counts.pdf` | RQ7 box plots |
| `analysis/figures/rq7_missingness.pdf` | RQ7 missingness matrix over time |
| `analysis/figures/summary_composite.pdf` | Paper main composite figure |
| `analysis/tables/rq5_top_metrics_by_year.csv` | Machine-readable metric table |
| `analysis/tables/rq6_eval_profiles.csv` | Task evaluation fingerprints |
| `analysis/tables/robustness.csv` | Robustness check results |
| `analysis/tables/pipeline_quality.csv` | Manual quality estimate |

---

## Notes on the sample data

**Meta-evaluation papers** (e.g. `J19-3004`, "Taking MT Evaluation Metrics to Extremes"):
This paper evaluates automatic metrics against human judgments rather than running an NLG
system. It will likely be confirmed as a false positive by the LLM in Stage 3, but if it
passes through, its mentions of evaluation are not evidence of an NLG system being evaluated.
Flag with `is_meta_eval = True` using the regex in section 0B.

**WMT findings papers** (`2021.wmt-1.1`, `W16-2301`) are legitimate data points — they
compare many MT systems and make real evaluation methodology choices. Their annotator counts
will appear as high-value outliers in RQ7 box plots, which is expected and should be
noted rather than removed.