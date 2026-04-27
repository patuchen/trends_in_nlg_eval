"""
Stage 0-1: Data loading and single-pass filtering with regex matching.
"""
import re
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ============================================================================
# STAGE 0 — Data Loader
# ============================================================================

def load_papers(csv_path: str) -> pd.DataFrame:
    """Load papers CSV and add txt_path column.
    
    Args:
        csv_path: Path to papers.csv with columns: id, URL, path, name, volume, year, abstract
    
    Returns:
        DataFrame with added txt_path column
    """
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded {len(df)} papers from {csv_path}")
    
    # Create txt_path using paper ID directly (format: YYYY.venue-number.pdf)
    df['txt_path'] = df['id'].apply(lambda pid: f"txt_papers/{pid}.pdf" if pd.notna(pid) else None)
    
    logger.info(f"Added txt_path column. Papers with txt_path: {df['txt_path'].notna().sum()}")
    return df


# ============================================================================
# STAGE 1 — Single-Pass Filter and Regex Matching
# ============================================================================

# 1A: Pre-processing

def remove_references_and_appendices(text: str) -> str:
    """Remove references and appendices from text.
    Keep only the main body of the paper (before References section).
    
    Args:
        text: Full paper text
    
    Returns:
        Text with references and appendices removed
    """
    # Try standard reference section markers (case-insensitive)
    for marker in ["\nReferences\n", "\nAPPENDIX\n", "\nAppendix\n", "\nAPPENDICES\n", "\nAppendices\n"]:
        if marker.lower() in text.lower():
            # Find case-insensitive position
            lower_text = text.lower()
            pos = lower_text.find(marker.lower())
            if pos != -1:
                return text[:pos]
    
    # Fallback: split at any occurrence of References/Appendix/etc at start of line
    text = re.split(r'\n(?:References|REFERENCES|Appendix|APPENDIX|Appendices|APPENDICES)\n', text, maxsplit=1)[0]
    return text


def preprocess_text(text: str) -> str:
    """Preprocess text by normalizing whitespace and line breaks.
    
    Args:
        text: Raw text to preprocess
    
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Rejoin hyphenated line breaks
    text = re.sub(r'-[ \t]*\n[ \t]*', '', text)
    
    # Normalize multiple spaces/tabs to single space
    text = re.sub(r'[ \t]{2,}', ' ', text)
    
    return text


# 1B: Meta-paper exclusion

def is_meta_paper(paper_id: str, title: str, abstract: str = "") -> bool:
    """Check if paper is a meta-paper (proceedings, survey, etc).
    
    Args:
        paper_id: Paper identifier
        title: Paper title
        abstract: Paper abstract (optional, used for survey detection)
    
    Returns:
        True if paper is a meta-paper
    """
    # Check paper ID ends in .0
    if re.search(r'\.0$', paper_id.strip()):
        return True
    
    # Check title matches proceedings pattern
    if re.compile(r'^proceedings\s+of\s+the\b', re.IGNORECASE).search(title):
        return True
    
    # Check title/abstract for survey/position paper/etc
    survey_regex = re.compile(
        r'\b(survey|overview|position\s+paper|position\s+statement|'
        r'shared\s+task|meta[\s-]analysis|systematic\s+review|'
        r'literature\s+review|tutorial)\b',
        re.IGNORECASE
    )
    
    # Run survey regex on title + first 3 sentences of abstract only
    text_to_check = title
    if abstract and pd.notna(abstract):
        abstract = str(abstract)
        sentences = abstract.split('.')[:3]
        text_to_check += ' ' + '.'.join(sentences)
    
    if survey_regex.search(text_to_check):
        return True
    
    return False


# 1C: Generative task matching

GENERATIVE_TASKS: Dict[str, List[str]] = {
    "machine_translation": [
        r'machine\s+translation', r'\bMT\b', r'neural\s+translation',
        r'\btranslat(ion|ing|e)\b',
    ],
    "summarization": [
        r'(text\s+)?summarization', r'abstractive\s+summar',
        r'document\s+summar',
    ],
    "data_to_text": [
        r'data[\s-]to[\s-]text', r'table[\s-]to[\s-]text',
        r'AMR[\s-]to[\s-]text', r'graph[\s-]to[\s-]text',
        r'knowledge[\s-]graph.*generat',
    ],
    "dialogue": [
        r'dialogue\s+(generation|system|response)',
        r'response\s+generation', r'open[\s-]domain\s+dialogue',
        r'conversational\s+(AI|model|system)',
    ],
    "question_generation": [
        r'question\s+generation', r'answer\s+generation',
    ],
    "story_generation": [
        r'story\s+generation', r'narrative\s+generation',
        r'creative\s+(text\s+)?generation',
    ],
    "simplification": [
        r'text\s+simplification', r'sentence\s+simplification',
    ],
    "paraphrase": [
        r'paraphrase\s+generation', r'paraphras(ing|e)',
    ],
    "captioning": [
        r'(image\s+)?caption(ing|\s+generation)',
    ],
    "general_nlg": [
        r'natural\s+language\s+generation', r'\bNLG\b',
        r'\btext\s+generation\b', r'\blanguage\s+generation\b',
    ],
    "code_generation": [
        r'code\s+generation',
    ],
    "style_transfer": [
        r'style\s+transfer',
    ],
}


def match_tasks(text: str) -> List[str]:
    """Match generative tasks in title + abstract.
    
    Args:
        text: Title + abstract text to search
    
    Returns:
        List of matched task names
    """
    matched = []
    
    for task_name, patterns in GENERATIVE_TASKS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matched.append(task_name)
                break  # Only add task once
    
    return matched


# 1D: Evaluation signal regexes

humeval_regex = re.compile(
    r'('
    r'(human|manual)\s+evaluation|qualitative\s+analysis|'
    r'human\s+(rater|annotator|judge|evaluator|judgment|judgement)|'
    r'(rater|annotator)\s+agreement|'
    r'inter[\s-]?(annotator|rater)\s+agreement|'
    r'crowd[\s-]?sourc\w+\s+(annotation|evaluation)|'
    r'\bMTurk\b|Mechanical\s+Turk|Amazon\s+Turk|'
    r'preference\s+(study|evaluation|test|judgment|judgement)|'
    r'side[\s-]by[\s-]side\s+(evaluation|comparison)|'
    r'pairwise\s+(human\s+)?(evaluation|comparison|ranking)|'
    r'annotation\s+(study|campaign|task)|'
    r'(manual|human)\s+(error\s+)?analysis|'
    r'blind\s+annotation|error\s+annotation|'
    r'Direct\s+Assessment|\bDA\s+(evaluation|score|rating|protocol)\b|'
    r'Multidimensional\s+Quality\s+Metric|Multi[\s-]?dimensional\s+Quality|\bMQM\b|'
    r'Error\s+Span\s+Annotation|\bESA\b\s+(evaluation|protocol|annotation)|'
    r'Scalar\s+Quality\s+Metric|\bSQM\b|'
    r'Human[\s-]?Targeted\s+(Translation\s+)?Error\s+Rate|\bHTER\b|'
    r'post[\s-]edit(ing|ed)(\s+evaluation)?|\bMTPE\b|'
    r'adequacy[\s/]fluency\s+(evaluation|rating)|'
    r'\d[\s-]?point\s+(likert|scale|rating)|'
    r'likert[\s-]?(scale|type)|'
    r'binary\s+(evaluation|judgment|judgement|annotation)|'
    r'(best[\s-]worst|BWS)\s+scal'
    r')',
    re.IGNORECASE
)

autoeval_regex = re.compile(
    r'('
    r'automatic\s+(evaluation|metric|scoring|assessment)|'
    r'reference[\s-]based\s+metric|'
    r'\b(BLEU[\s-]?\d*|SacreBLEU|ROUGE[\s-]?\d*L?|METEOR|chrF[\+\d]*|TER|WER|CIDEr|SPICE|NIST|RIBES)\b|'
    r'\b(BERTScore|BLEURT|MoverScore|BARTScore|COMET|COMET[\s-]?DA|'
    r'UniTE|PRISM|YiSi|NUBIA|BLANC|SummaQA|QuestEval|UniEval|'
    r'GPTScore|FED|USR|GRADE)\b|'
    r'\b(perplexity|bits[\s-]per[\s-](character|word))\b|'
    r'\b(SER|slot\s+error\s+rate)\b|'
    r'\b(accuracy|F1[\s-]score|exact\s+match|PARENT[\s-]?T?)\b'
    r')',
    re.IGNORECASE
)

llm_judge_regex = re.compile(
    r'\bllm[-\s]*(as[-\s]*a[-\s]*judge|as[-\s]*an?[-\s]*evaluator|'
    r'based[-\s]*metric|metric|judge|evaluator)\b|'
    r'(GPT-?4|ChatGPT|claude|gemini)[\s-]*(as[\s-]*(a[\s-]*)?judge|'
    r'evaluation|evaluator|based\s+metric)|'
    r'model[\s-]based\s+(judge|evaluator)',
    re.IGNORECASE
)


# 1E: Context window extraction

def get_context_windows(text: str, regex: re.Pattern, window_chars: int = 400) -> List[str]:
    """Extract context windows around regex matches.
    
    Args:
        text: Text to search
        regex: Compiled regex pattern
        window_chars: Characters before/after match to include
    
    Returns:
        List of context windows (deduplicated)
    """
    contexts = []
    
    for match in regex.finditer(text):
        start_pos = max(0, match.start() - window_chars)
        end_pos = min(len(text), match.end() + window_chars)
        context = text[start_pos:end_pos]
        contexts.append(context)
    
    # Simple deduplication for overlapping windows
    if not contexts:
        return []
    
    # Keep first 5 windows to avoid huge storage
    contexts = contexts[:5]
    
    # Deduplicate near-identical windows
    unique = []
    for ctx in contexts:
        if not any(ctx in u or u in ctx for u in unique):
            unique.append(ctx)
    
    return unique


# 1F: Main filter loop

def run_filter(df: pd.DataFrame, txt_dir: str) -> pd.DataFrame:
    """Run single-pass filter on all papers.
    
    Args:
        df: DataFrame from load_papers()
        txt_dir: Directory containing text files (e.g., 'txt_papers')
    
    Returns:
        DataFrame with filter results and added columns
    """
    txt_path_obj = Path(txt_dir)
    
    results = {
        'passed_filter': [],
        'inferred_tasks': [],
        'humeval_count': [],
        'autoeval_count': [],
        'llm_judge_count': [],
        'humeval_contexts': [],
        'autoeval_contexts': [],
        'llm_judge_contexts': [],
        'filter_drop_reason': [],
    }
    
    # Statistics
    stats = {
        'total': 0,
        'meta': 0,
        'no_task': 0,
        'file_missing': 0,
        'no_eval_signal': 0,
        'passed': 0,
    }
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering papers"):
        stats['total'] += 1
        paper_id = row['id']
        
        # Ensure abstract is a string
        abstract = str(row.get('abstract', '')) if pd.notna(row.get('abstract')) else ''
        title = str(row['name']) if pd.notna(row['name']) else ''
        
        # Check meta-paper
        if is_meta_paper(paper_id, title, abstract):
            results['passed_filter'].append(False)
            results['filter_drop_reason'].append('meta_paper')
            results['inferred_tasks'].append(json.dumps([]))
            results['humeval_count'].append(0)
            results['autoeval_count'].append(0)
            results['llm_judge_count'].append(0)
            results['humeval_contexts'].append(json.dumps([]))
            results['autoeval_contexts'].append(json.dumps([]))
            results['llm_judge_contexts'].append(json.dumps([]))
            stats['meta'] += 1
            continue
        
        # Match tasks in title + abstract
        tasks = match_tasks(f"{title} {abstract}")
        if not tasks:
            results['passed_filter'].append(False)
            results['filter_drop_reason'].append('no_task')
            results['inferred_tasks'].append(json.dumps([]))
            results['humeval_count'].append(0)
            results['autoeval_count'].append(0)
            results['llm_judge_count'].append(0)
            results['humeval_contexts'].append(json.dumps([]))
            results['autoeval_contexts'].append(json.dumps([]))
            results['llm_judge_contexts'].append(json.dumps([]))
            stats['no_task'] += 1
            continue
        
        # Load text file - use paper ID directly
        txt_file = txt_path_obj / f"{paper_id}.pdf"
        
        if not txt_file.exists():
            logger.warning(f"Text file not found for {paper_id}: {txt_file}")
            results['passed_filter'].append(False)
            results['filter_drop_reason'].append('file_missing')
            results['inferred_tasks'].append(json.dumps(tasks))
            results['humeval_count'].append(0)
            results['autoeval_count'].append(0)
            results['llm_judge_count'].append(0)
            results['humeval_contexts'].append(json.dumps([]))
            results['autoeval_contexts'].append(json.dumps([]))
            results['llm_judge_contexts'].append(json.dumps([]))
            stats['file_missing'] += 1
            continue
        
        # Load and preprocess text
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            
            # Remove references and appendices to focus on main body
            text = remove_references_and_appendices(text)
            text = preprocess_text(text)
        except Exception as e:
            logger.warning(f"Error reading {txt_file}: {e}")
            results['passed_filter'].append(False)
            results['filter_drop_reason'].append('file_read_error')
            results['inferred_tasks'].append(json.dumps(tasks))
            results['humeval_count'].append(0)
            results['autoeval_count'].append(0)
            results['llm_judge_count'].append(0)
            results['humeval_contexts'].append(json.dumps([]))
            results['autoeval_contexts'].append(json.dumps([]))
            results['llm_judge_contexts'].append(json.dumps([]))
            continue
        
        # Run evaluation signal regexes
        humeval_count = len(list(humeval_regex.finditer(text)))
        autoeval_count = len(list(autoeval_regex.finditer(text)))
        llm_judge_count = len(list(llm_judge_regex.finditer(text)))
        
        # Check if has at least one eval signal
        if humeval_count == 0 and autoeval_count == 0 and llm_judge_count == 0:
            results['passed_filter'].append(False)
            results['filter_drop_reason'].append('no_eval_signal')
            results['inferred_tasks'].append(json.dumps(tasks))
            results['humeval_count'].append(0)
            results['autoeval_count'].append(0)
            results['llm_judge_count'].append(0)
            results['humeval_contexts'].append(json.dumps([]))
            results['autoeval_contexts'].append(json.dumps([]))
            results['llm_judge_contexts'].append(json.dumps([]))
            stats['no_eval_signal'] += 1
            continue
        
        # Passed filter — extract context windows
        humeval_contexts = get_context_windows(text, humeval_regex)
        autoeval_contexts = get_context_windows(text, autoeval_regex)
        llm_judge_contexts = get_context_windows(text, llm_judge_regex)
        
        results['passed_filter'].append(True)
        results['filter_drop_reason'].append(None)
        results['inferred_tasks'].append(json.dumps(tasks))
        results['humeval_count'].append(humeval_count)
        results['autoeval_count'].append(autoeval_count)
        results['llm_judge_count'].append(llm_judge_count)
        results['humeval_contexts'].append(json.dumps(humeval_contexts))
        results['autoeval_contexts'].append(json.dumps(autoeval_contexts))
        results['llm_judge_contexts'].append(json.dumps(llm_judge_contexts))
        stats['passed'] += 1
    
    # Add result columns to dataframe
    for col, values in results.items():
        df[col] = values
    
    # Log statistics
    logger.info("=" * 70)
    logger.info("FILTER STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Total papers processed:       {stats['total']}")
    logger.info(f"Dropped (meta-paper):         {stats['meta']}")
    logger.info(f"Dropped (no generative task): {stats['no_task']}")
    logger.info(f"Dropped (file missing):       {stats['file_missing']}")
    logger.info(f"Dropped (no eval signal):     {stats['no_eval_signal']}")
    logger.info(f"PASSED FILTER:                {stats['passed']}")
    logger.info("=" * 70)
    
    # Save filtered results
    output_path = Path('output') / 'filtered_papers.csv'
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved filtered papers to {output_path}")
    
    return df
