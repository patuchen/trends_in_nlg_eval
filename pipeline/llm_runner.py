"""
Stage 3: LLM-based extraction of evaluation details using vLLM.
Calls local Qwen2.5-14B-Instruct via OpenAI-compatible API.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import requests
from tqdm.auto import tqdm

from pipeline.filters import preprocess_text, remove_references_and_appendices
from pipeline.extractor import extract_evaluation_section, section_from_contexts

logger = logging.getLogger(__name__)

# Default prompts (loaded from files)
DEFAULT_CONFIRMATION_PROMPT = None
DEFAULT_EXTRACTION_PROMPT = None


def load_prompts(confirmation_file: str = "prompts/confirmation.txt",
                 extraction_file: str = "prompts/extraction.txt"):
    """Load prompt templates from files.
    
    Args:
        confirmation_file: Path to confirmation prompt
        extraction_file: Path to extraction prompt
    """
    global DEFAULT_CONFIRMATION_PROMPT, DEFAULT_EXTRACTION_PROMPT
    
    try:
        with open(confirmation_file, 'r') as f:
            DEFAULT_CONFIRMATION_PROMPT = f.read().strip()
    except FileNotFoundError:
        logger.error(f"Could not find {confirmation_file}")
        DEFAULT_CONFIRMATION_PROMPT = "PROMPT NOT FOUND"
    
    try:
        with open(extraction_file, 'r') as f:
            DEFAULT_EXTRACTION_PROMPT = f.read().strip()
    except FileNotFoundError:
        logger.error(f"Could not find {extraction_file}")
        DEFAULT_EXTRACTION_PROMPT = "PROMPT NOT FOUND"


class LLMRunner:
    """Runner for LLM extraction using vLLM OpenAI API."""
    
    def __init__(self, 
                 api_url: str = "http://localhost:8000/v1",
                 model: str = "Qwen/Qwen2.5-14B-Instruct",
                 max_tokens: int = 800,
                 temperature: float = 0.1,
                 timeout: int = 30):
        """Initialize LLM runner.
        
        Args:
            api_url: Base URL for OpenAI-compatible API
            model: Model name
            max_tokens: Max tokens in response
            temperature: Temperature for generation
            timeout: Request timeout in seconds
        """
        self.api_url = api_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # Check API connectivity
        self._check_api_availability()
    
    def _check_api_availability(self):
        """Check if vLLM API is available."""
        try:
            response = requests.get(
                f"{self.api_url}/models",
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"✓ vLLM API available at {self.api_url}")
                logger.info(f"  Model: {self.model}")
            else:
                logger.warning(f"vLLM API returned status {response.status_code}")
        except Exception as e:
            logger.error(f"✗ Cannot reach vLLM API at {self.api_url}: {e}")
            logger.error("  Make sure vLLM is running: python -m vllm.entrypoints.openai.api_server ...")
            raise
    
    def call_llm(self, system_prompt: str, user_message: str) -> str:
        """Call LLM with system + user message.
        
        Args:
            system_prompt: System prompt
            user_message: User message
        
        Returns:
            LLM response text
        """
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"LLM API error {response.status_code}: {response.text}")
                return ""
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        
        except requests.Timeout:
            logger.error("LLM request timed out")
            return ""
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return ""
    
    
    # Removed async method - using synchronous requests only


def parse_json_response(response_text: str, retry: bool = True) -> Optional[Dict[str, Any]]:
    """Parse JSON response from LLM.
    
    More forgiving parser that:
    1. Tries direct JSON parsing first
    2. Attempts to extract JSON from text with extra content
    3. Normalizes schema (modalities as arrays)
    
    Args:
        response_text: Response text from LLM
        retry: Whether this is a retry attempt
    
    Returns:
        Parsed JSON or None
    """
    if not response_text:
        return None
    
    # Try direct parsing first
    try:
        result = json.loads(response_text)
        # Normalize modality fields to always be lists
        _normalize_modalities(result)
        return result
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from text that has extra content
    import re
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
    if json_match:
        try:
            extracted = json_match.group(0)
            result = json.loads(extracted)
            _normalize_modalities(result)
            return result
        except json.JSONDecodeError:
            pass
    
    logger.debug(f"JSON parse error on response: {response_text[:200]}...")
    return None


def _normalize_modalities(data: Dict[str, Any]) -> None:
    """Normalize modality fields to always be lists.
    
    Handles cases where LLM returns string instead of array for modalities.
    Modifies data in-place.
    
    Args:
        data: Parsed JSON data
    """
    if not isinstance(data, dict):
        return
    
    for field in ['modality_in', 'modality_out']:
        if field in data:
            if isinstance(data[field], str):
                # Convert string to list
                data[field] = [data[field]]
            elif not isinstance(data[field], list):
                # If it's neither string nor list, default to text
                data[field] = ['text']
    
    # Same for nested modalities object
    if 'modalities' in data and isinstance(data['modalities'], dict):
        for field in ['modality_in', 'modality_out']:
            if field in data['modalities']:
                if isinstance(data['modalities'][field], str):
                    data['modalities'][field] = [data['modalities'][field]]
                elif not isinstance(data['modalities'][field], list):
                    data['modalities'][field] = ['text']


def run_extraction(filtered_csv: str,
                   txt_dir: str,
                   output_jsonl: str,
                   api_url: str = "http://localhost:8000/v1",
                   model: str = "Qwen/Qwen2.5-14B-Instruct",
                   batch_size: int = 32,
                   max_papers: Optional[int] = None,
                   append_mode: bool = False) -> int:
    """Run full extraction pipeline.
    
    Args:
        filtered_csv: Path to filtered_papers.csv
        txt_dir: Path to text directory
        output_jsonl: Output JSONL file
        api_url: vLLM API URL
        model: Model name
        batch_size: Batch size for async requests
        max_papers: Maximum papers to process (None = all)
        append_mode: If True, append to existing file; if False, overwrite
    
    Returns:
        Number of papers processed
    """
    # Load prompts
    load_prompts()
    
    # Initialize LLM runner
    runner = LLMRunner(api_url=api_url, model=model)
    
    # Load filtered papers
    logger.info(f"Loading filtered papers from {filtered_csv}...")
    papers_df = pd.read_csv(filtered_csv, low_memory=False)
    
    if max_papers:
        papers_df = papers_df.head(max_papers)
    
    logger.info(f"Processing {len(papers_df)} papers")
    
    # Output file
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(exist_ok=True)
    
    txt_path = Path(txt_dir)
    
    # Statistics
    stats = {
        'processed': 0,
        'parse_failed': 0,
        'eval_all_false': 0,
        'eval_counts': {'human_eval': 0, 'auto_eval': 0, 'llm_judge': 0},
    }
    
    # Process papers
    file_mode = 'a' if append_mode else 'w'
    with open(output_path, file_mode) as out_f:
        for idx, (_, row) in enumerate(tqdm(papers_df.iterrows(), total=len(papers_df),
                                             desc="Extracting evaluation details")):
            paper_id = row['id']
            
            # Load text
            paper_path = row['path'] if pd.notna(row['path']) else paper_id
            txt_file = txt_path / f"{paper_path}.pdf"
            
            if not txt_file.exists():
                txt_file = txt_path / f"{paper_id}.pdf"
            
            if not txt_file.exists():
                logger.warning(f"Text file not found for {paper_id}")
                continue
            
            try:
                with open(txt_file, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
            except Exception as e:
                logger.warning(f"Error reading {txt_file}: {e}")
                continue
            
            # Preprocess and remove references + appendices, then extract section
            text = remove_references_and_appendices(text)
            text = preprocess_text(text)
            section_text, extraction_method = extract_evaluation_section(text)
            
            # Fallback to context windows if heading extraction failed
            if section_text is None:
                # Parse context windows from filtered CSV
                try:
                    contexts = []
                    for col in ['humeval_contexts', 'autoeval_contexts', 'llm_judge_contexts']:
                        if pd.notna(row[col]):
                            try:
                                ctx_list = json.loads(row[col])
                                contexts.extend(ctx_list)
                            except:
                                pass
                    
                    if contexts:
                        section_text = section_from_contexts(contexts)
                    else:
                        section_text = text[:8000]
                except:
                    section_text = text[:8000]
            
            # Skip papers with completely empty extraction (unable to extract any content)
            # This check happens AFTER fallback, so we've tried our best to get text
            if not section_text or len(section_text.strip()) < 20:
                logger.warning(f"Empty extraction for {paper_id} - no text available after fallback")
                record = {
                    'paper_id': paper_id,
                    'title': row['name'],
                    'year': int(row['year']) if pd.notna(row['year']) else None,
                    'venue': row.get('venue', ''),
                    'inferred_tasks': json.loads(row['inferred_tasks']) if pd.notna(row['inferred_tasks']) else [],
                    'pipeline': {
                        'passed_generative_filter': True,
                        'passed_survey_filter': True,
                        'extraction_skipped': True,
                        'extraction_skip_reason': 'empty_text',
                    },
                    'human_evaluation': {'conducted': False},
                    'automatic_evaluation': {'conducted': False},
                    'llm_as_judge': {'conducted': False},
                }
                out_f.write(json.dumps(record) + '\n')
                out_f.flush()
                stats['processed'] += 1
                continue
            
            # OPTIMIZATION: Check regex signals FIRST before calling confirmation prompt
            # This allows us to skip the confirmation call for papers with strong regex evidence
            regex_signals = {
                'human_eval': int(row['humeval_count']) if pd.notna(row['humeval_count']) else 0,
                'auto_eval': int(row['autoeval_count']) if pd.notna(row['autoeval_count']) else 0,
                'llm_judge': int(row['llm_judge_count']) if pd.notna(row['llm_judge_count']) else 0,
            }
            
            REGEX_SIGNAL_THRESHOLD = 10
            
            # Prepare truncated text once (used by both confirmation and extraction if needed)
            section_text_truncated = section_text[:1500] if section_text else ""
            
            # Pre-populate confirmation_data with strong regex signals (skip expensive LLM call)
            confirmation_data = {
                'human_eval': regex_signals['human_eval'] >= REGEX_SIGNAL_THRESHOLD,
                'auto_eval': regex_signals['auto_eval'] >= REGEX_SIGNAL_THRESHOLD,
                'llm_judge': regex_signals['llm_judge'] >= REGEX_SIGNAL_THRESHOLD,
            }
            
            # Only call confirmation prompt for eval types with weak signals (<10)
            needs_confirmation = not any([
                regex_signals['human_eval'] >= REGEX_SIGNAL_THRESHOLD,
                regex_signals['auto_eval'] >= REGEX_SIGNAL_THRESHOLD,
                regex_signals['llm_judge'] >= REGEX_SIGNAL_THRESHOLD,
            ])
            
            hybrid_override_used = any(confirmation_data.values())  # True if any regex-based signal was strong
            
            if needs_confirmation:
                # Call Prompt A (confirmation) only for papers with uncertain signals
                confirmation_response = runner.call_llm(
                    DEFAULT_CONFIRMATION_PROMPT,
                    f"Paper excerpt:\n\n{section_text_truncated}"
                )
                
                llm_confirmation = parse_json_response(confirmation_response)
                
                if not llm_confirmation:
                    logger.warning(f"Parse failed for confirmation prompt on {paper_id}")
                    stats['parse_failed'] += 1
                    llm_confirmation = {
                        'human_eval': False,
                        'auto_eval': False,
                        'llm_judge': False,
                        'reasoning': 'Parse error'
                    }
                
                # Use LLM confirmation for weak-signal cases
                confirmation_data['human_eval'] = confirmation_data['human_eval'] or llm_confirmation.get('human_eval', False)
                confirmation_data['auto_eval'] = confirmation_data['auto_eval'] or llm_confirmation.get('auto_eval', False)
                confirmation_data['llm_judge'] = confirmation_data['llm_judge'] or llm_confirmation.get('llm_judge', False)
            else:
                # Strong regex signals found - skip confirmation prompt
                logger.info(f"Skipping confirmation prompt (strong regex signals): {paper_id}")
            
            # Check if all evaluation types are false
            if not (confirmation_data.get('human_eval') or 
                    confirmation_data.get('auto_eval') or 
                    confirmation_data.get('llm_judge')):
                # Extract modalities even if no evaluation detected
                modality_in = confirmation_data.get('modality_in', ['text'])
                modality_out = confirmation_data.get('modality_out', ['text'])
                
                record = {
                    'paper_id': paper_id,
                    'title': row['name'],
                    'year': int(row['year']) if pd.notna(row['year']) else None,
                    'venue': row.get('venue', ''),
                    'inferred_tasks': json.loads(row['inferred_tasks']) if pd.notna(row['inferred_tasks']) else [],
                    'pipeline': {
                        'passed_generative_filter': True,
                        'passed_survey_filter': True,
                        'regex_signals': {
                            'human_eval_match_count': int(row['humeval_count']) if pd.notna(row['humeval_count']) else 0,
                            'auto_eval_match_count': int(row['autoeval_count']) if pd.notna(row['autoeval_count']) else 0,
                            'llm_judge_match_count': int(row['llm_judge_count']) if pd.notna(row['llm_judge_count']) else 0,
                        },
                        'evaluation_section_extracted': section_text is not None,
                        'extraction_method': extraction_method,
                        'llm_processed': True,
                        'llm_model': model,
                        'parse_failed': False,
                        'haiku_spot_checked': False,
                    },
                    'human_evaluation': {'conducted': False},
                    'automatic_evaluation': {'conducted': False},
                    'llm_as_judge': {'conducted': False},
                    'modalities': {
                        'modality_in': modality_in if isinstance(modality_in, list) else ['text'],
                        'modality_out': modality_out if isinstance(modality_out, list) else ['text'],
                    },
                }
                out_f.write(json.dumps(record) + '\n')
                out_f.flush()
                stats['eval_all_false'] += 1
                stats['processed'] += 1
                continue
            
            # Call Prompt B (structured extraction) for confirmed categories
            # Use same truncated text for consistency
            extraction_response = runner.call_llm(
                DEFAULT_EXTRACTION_PROMPT,
                f"Paper excerpt:\n\n{section_text_truncated}"
            )
            
            extraction_data = parse_json_response(extraction_response)
            
            if not extraction_data:
                logger.warning(f"Parse failed for extraction prompt on {paper_id}")
                # Retry with explicit instruction
                extraction_response = runner.call_llm(
                    DEFAULT_EXTRACTION_PROMPT + "\n\nRespond with ONLY valid JSON.",
                    f"Paper excerpt:\n\n{section_text_truncated}"
                )
                extraction_data = parse_json_response(extraction_response)
                
                if not extraction_data:
                    stats['parse_failed'] += 1
                    extraction_data = {}
            
            # Count eval types
            for eval_type in ['human_eval', 'auto_eval', 'llm_judge']:
                if confirmation_data.get(eval_type):
                    stats['eval_counts'][eval_type] += 1
            
            # Assemble final record
            # Extract modalities from confirmation and extraction data
            modality_in = confirmation_data.get('modality_in', ['text'])
            modality_out = confirmation_data.get('modality_out', ['text'])
            if extraction_data.get('modalities'):
                if extraction_data['modalities'].get('modality_in'):
                    modality_in = extraction_data['modalities']['modality_in']
                if extraction_data['modalities'].get('modality_out'):
                    modality_out = extraction_data['modalities']['modality_out']
            
            record = {
                'paper_id': paper_id,
                'title': row['name'],
                'year': int(row['year']) if pd.notna(row['year']) else None,
                'venue': row.get('venue', ''),
                'inferred_tasks': json.loads(row['inferred_tasks']) if pd.notna(row['inferred_tasks']) else [],
                'pipeline': {
                    'passed_generative_filter': True,
                    'passed_survey_filter': True,
                    'regex_signals': {
                        'human_eval_match_count': int(row['humeval_count']) if pd.notna(row['humeval_count']) else 0,
                        'auto_eval_match_count': int(row['autoeval_count']) if pd.notna(row['autoeval_count']) else 0,
                        'llm_judge_match_count': int(row['llm_judge_count']) if pd.notna(row['llm_judge_count']) else 0,
                    },
                    'evaluation_section_extracted': section_text is not None,
                    'extraction_method': extraction_method,
                    'llm_processed': True,
                    'llm_model': model,
                    'parse_failed': extraction_data == {},
                    'haiku_spot_checked': False,
                    'hybrid_override': hybrid_override_used,
                },
                'human_evaluation': extraction_data.get('human_evaluation', {'conducted': confirmation_data.get('human_eval', False)}),
                'automatic_evaluation': extraction_data.get('automatic_evaluation', {'conducted': confirmation_data.get('auto_eval', False)}),
                'llm_as_judge': extraction_data.get('llm_as_judge', {'conducted': confirmation_data.get('llm_judge', False)}),
                'languages': extraction_data.get('languages', {}),
                'modalities': {
                    'modality_in': modality_in if isinstance(modality_in, list) else ['text'],
                    'modality_out': modality_out if isinstance(modality_out, list) else ['text'],
                },
                'extraction_text': section_text[:1000] if section_text else "",  # Store first 1000 chars
            }
            
            out_f.write(json.dumps(record) + '\n')
            out_f.flush()
            stats['processed'] += 1
            
            # Log progress every 100 papers
            if (stats['processed'] % 100) == 0:
                logger.info(f"Processed {stats['processed']} papers. "
                           f"Parse failures: {stats['parse_failed']}")
    
    # Log final statistics
    logger.info("=" * 70)
    logger.info("EXTRACTION STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Total papers processed:       {stats['processed']}")
    logger.info(f"Parse failures:               {stats['parse_failed']}")
    logger.info(f"All evals false (skipped):    {stats['eval_all_false']}")
    logger.info(f"Human eval present:           {stats['eval_counts']['human_eval']}")
    logger.info(f"Auto eval present:            {stats['eval_counts']['auto_eval']}")
    logger.info(f"LLM judge present:            {stats['eval_counts']['llm_judge']}")
    logger.info("=" * 70)
    
    logger.info(f"Saved {stats['processed']} records to {output_path}")
    
    return stats['processed']
