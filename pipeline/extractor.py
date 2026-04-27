"""
Stage 2: Evaluation section extraction from full text.
"""
import re
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# Heading pattern and section names
SECTION_NUMBER_RE = re.compile(
    r'\n(\d{1,2}(?:\.\d{1,2}){0,2})\n([^\n]{3,80})\n'
)

EVAL_SECTION_NAMES = re.compile(
    r'\b(evaluation|experiment|result|analysis|assessment|'
    r'human\s+eval|automatic\s+eval|discussion)\b',
    re.IGNORECASE
)


def extract_evaluation_section(text: str) -> Tuple[Optional[str], str]:
    """Extract evaluation section from paper text using heading heuristic.
    
    Logic:
    1. Find all headings using SECTION_NUMBER_RE
    2. Compute depth for each as num.count('.') + 1
    3. Find first heading whose name matches EVAL_SECTION_NAMES
    4. Extract text from that heading's end to next heading at equal/shallower depth
    5. Strip and cap at 8000 characters
    6. If no heading found, return (None, "context_fallback")
    
    Args:
        text: Full paper text with numbered sections
    
    Returns:
        Tuple of (section_text or None, method)
        method is "heading_heuristic" or "context_fallback"
    """
    if not text or not isinstance(text, str):
        return None, "context_fallback"
    
    # Find all headings
    headings = []
    for match in SECTION_NUMBER_RE.finditer(text):
        section_num = match.group(1)
        section_name = match.group(2)
        depth = section_num.count('.') + 1
        
        headings.append({
            'num': section_num,
            'name': section_name,
            'depth': depth,
            'start': match.start(2),  # Start of heading name
            'end': match.end(),         # End of full heading match (includes newline)
        })
    
    if not headings:
        return None, "context_fallback"
    
    # Find first heading matching evaluation pattern
    eval_heading_idx = None
    for i, heading in enumerate(headings):
        if EVAL_SECTION_NAMES.search(heading['name']):
            eval_heading_idx = i
            break
    
    if eval_heading_idx is None:
        return None, "context_fallback"
    
    eval_heading = headings[eval_heading_idx]
    target_depth = eval_heading['depth']
    
    # Find start of evaluation section (end of current heading match)
    section_start = eval_heading['end']
    
    # Find end: next heading at equal or shallower depth
    section_end = len(text)  # Default to end of text
    for i in range(eval_heading_idx + 1, len(headings)):
        if headings[i]['depth'] <= target_depth:
            section_end = headings[i]['start']
            break
    
    # Extract section
    section_text = text[section_start:section_end].strip()
    
    # Cap at 8000 characters (~2000 tokens)
    section_text = section_text[:8000]
    
    return section_text, "heading_heuristic"


def section_from_contexts(context_windows: List[str]) -> str:
    """Create section text from context windows (fallback method).
    
    Args:
        context_windows: List of context window strings
    
    Returns:
        Joined context text, capped at 8000 characters
    """
    if not context_windows:
        return ""
    
    # Join with separator
    section_text = "\n---\n".join(context_windows)
    
    # Cap at 8000 characters
    section_text = section_text[:8000]
    
    return section_text
