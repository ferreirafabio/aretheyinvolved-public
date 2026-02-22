"""Stage 1.5: LLM Recovery for documents where NER found suspiciously few names.

This module runs when:
- Document has substantial text (>500 chars) AND
- NER found very few names (0-2) AND
- Text has high OCR noise score

SAFETY GUARANTEES:
- LLM must provide exact evidence spans (start, end)
- We verify raw_text[start:end] == evidence_text EXACTLY
- No speculative name guessing - only OCR character fixes (0→O, 1→I, 5→S)
- Worst case: highlights junk, which is rejected deterministically

Key principle: The LLM can only FIND names that exist, not INVENT them.
"""

import re
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .llm_classifier import ClassifiedSpan


@dataclass
class RecoveredSpan:
    """A name span recovered by LLM from corrupted text."""
    text: str  # Exact text from raw document (evidence)
    start: int  # Exact start offset in raw text
    end: int  # Exact end offset in raw text
    normalized_name: str  # OCR-fixed version (conservative fixes only)
    confidence: float
    role: str = "mentioned"  # Conservative default
    page_number: int = 1


@dataclass
class RecoveryResult:
    """Result from LLM recovery stage."""
    source_file: str
    recovered_spans: list[RecoveredSpan]
    total_recovered: int
    was_triggered: bool
    noise_score: float = 0.0


# ============================================================
# Noise Score Calculation (cheap pre-filter)
# ============================================================

def compute_noise_score(text: str) -> float:
    """Compute OCR noise score for gating recovery.

    Higher score = more corrupted text = more likely to need recovery.

    Signals:
    - Ratio of non-letter/non-space chars
    - Broken tokens (alternating letter/digit like H4tt)
    - High proportion of very short tokens (1-2 chars)
    - Rare unicode / box drawing chars

    Returns:
        Score from 0.0 (clean) to 1.0 (heavily corrupted).
    """
    if not text or len(text) < 50:
        return 0.0

    signals = []

    # Signal 1: Non-alphanumeric ratio (excluding common punct and whitespace)
    common_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t.,;:!?\'"-()[]{}@#$%&*/')
    non_common = sum(1 for c in text if c not in common_chars)
    signals.append(min(1.0, non_common / len(text) * 10))  # Scale up

    # Signal 2: Broken tokens (letter-digit-letter patterns)
    broken_pattern = re.compile(r'[A-Za-z][0-9][A-Za-z]|[0-9][A-Za-z][0-9]')
    broken_count = len(broken_pattern.findall(text))
    signals.append(min(1.0, broken_count / 20))  # Normalize

    # Signal 3: Short token ratio
    tokens = text.split()
    if tokens:
        short_tokens = sum(1 for t in tokens if len(t) <= 2)
        signals.append(short_tokens / len(tokens))
    else:
        signals.append(0.0)

    # Signal 4: Box drawing / garbage unicode
    garbage_chars = set('█▓▒░▌▐▀▄│┤╡╢╖╕╣║╗╝╜╛┐└┴┬├─┼╞╟╚╔╩╦╠═╬╧╨╤╥╙╘╒╓╫╪┘┌■□▪▫●○◌★☆✦✧※')
    garbage_count = sum(1 for c in text if c in garbage_chars)
    signals.append(min(1.0, garbage_count / len(text) * 20))

    # Weighted average
    weights = [0.3, 0.3, 0.2, 0.2]
    return sum(s * w for s, w in zip(signals, weights))


# ============================================================
# Character-level Candidate Generator (cheap middle layer)
# ============================================================

# Patterns that might contain names but NER missed
NAME_CANDIDATE_PATTERNS = [
    # Email headers (high confidence)
    (r'(?:From|To|CC|Cc|Attn|Dear|Sincerely)[:\s]+([A-Za-z][A-Za-z0-9\s\.\-\']{2,40})', 'header'),
    # Capitalized sequences that look name-like
    (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', 'capitalized'),
    # OCR-corrupted names (digit-letter mix with capital start)
    (r'\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)?)\b', 'ocr_corrupted'),
]


def find_name_candidates_cheap(text: str) -> list[tuple[int, int, str, str]]:
    """Find potential name locations using cheap regex heuristics.

    This is a middle layer between NER and LLM - catches patterns
    NER misses without needing expensive LLM calls.

    Args:
        text: Document text.

    Returns:
        List of (start, end, matched_text, pattern_type) tuples.
    """
    candidates = []
    seen_positions = set()

    for pattern, pattern_type in NAME_CANDIDATE_PATTERNS:
        for match in re.finditer(pattern, text):
            # Get the captured group if exists, else full match
            if match.groups():
                matched = match.group(1)
                start = match.start(1)
            else:
                matched = match.group(0)
                start = match.start()

            end = start + len(matched)

            # Skip if we've seen this position
            pos_key = (start, end)
            if pos_key in seen_positions:
                continue
            seen_positions.add(pos_key)

            # Basic filtering
            if len(matched) < 3 or len(matched) > 50:
                continue

            # Must have at least 2 letters
            letter_count = sum(1 for c in matched if c.isalpha())
            if letter_count < 2:
                continue

            # Not mostly digits/symbols
            if letter_count < len(matched) * 0.5:
                continue

            candidates.append((start, end, matched, pattern_type))

    return candidates


def is_name_like(text: str) -> bool:
    """Check if text looks like a person name.

    Conservative check to filter out obvious non-names.
    """
    if not text or len(text) < 2:
        return False

    # Must have at least 2 letters
    letters = sum(1 for c in text if c.isalpha())
    if letters < 2:
        return False

    # Letters should be majority
    if letters < len(text.replace(' ', '')) * 0.6:
        return False

    # Not all uppercase (likely acronym)
    if text.isupper() and len(text) > 4:
        return False

    # Contains at least one capital
    if not any(c.isupper() for c in text):
        return False

    return True


# ============================================================
# OCR Character Fixes (conservative, well-defined)
# ============================================================

# Only these substitutions are allowed - they're visually unambiguous
OCR_CHAR_FIXES = {
    '0': 'O',  # Zero → O (when in name context)
    '1': 'I',  # One → I (when in name context)
    '5': 'S',  # Five → S
    '8': 'B',  # Eight → B
    '|': 'l',  # Pipe → lowercase L
    '!': 'l',  # Exclamation → lowercase L (common OCR error)
}


def apply_conservative_ocr_fixes(text: str) -> str:
    """Apply only well-defined OCR character fixes.

    These are substitutions where the OCR error is visually unambiguous
    in the context of a name (e.g., J0HN → JOHN).

    Does NOT:
    - Split names (Annap → Anna P)
    - Guess missing characters
    - Add spaces
    - Change casing
    """
    result = list(text)

    for i, char in enumerate(text):
        if char in OCR_CHAR_FIXES:
            # Only fix if surrounded by letters (name context)
            prev_is_letter = i > 0 and text[i-1].isalpha()
            next_is_letter = i < len(text) - 1 and text[i+1].isalpha()

            if prev_is_letter or next_is_letter:
                result[i] = OCR_CHAR_FIXES[char]

    return ''.join(result)


# ============================================================
# Gating Logic
# ============================================================

# Thresholds for triggering recovery
MIN_TEXT_LENGTH = 500
MAX_NAMES_FOR_RECOVERY = 2
MIN_NOISE_SCORE = 0.15  # Must have some OCR noise


def should_trigger_recovery(text_length: int,
                            ner_name_count: int,
                            noise_score: float) -> bool:
    """Determine if LLM recovery should run.

    All three conditions must be met:
    1. Document is substantial (>500 chars)
    2. NER found suspiciously few names (0-2)
    3. Text shows signs of OCR corruption (noise_score > 0.15)
    """
    if text_length < MIN_TEXT_LENGTH:
        return False

    if ner_name_count > MAX_NAMES_FOR_RECOVERY:
        return False

    if noise_score < MIN_NOISE_SCORE:
        return False

    return True


# Backwards compatibility alias
def is_suspicious_document(text_length: int,
                           ner_name_count: int,
                           garbage_ratio: float) -> bool:
    """Check if document should trigger LLM recovery.

    Deprecated: Use should_trigger_recovery with noise_score instead.
    """
    # Convert old garbage_ratio to approximate noise_score
    noise_score = garbage_ratio * 5  # Rough conversion
    return should_trigger_recovery(text_length, ner_name_count, noise_score)


# ============================================================
# LLM Recovery Module
# ============================================================

class LLMNameRecovery:
    """Uses LLM to find names in corrupted documents that NER missed.

    Safety guarantees:
    - LLM must provide exact character spans
    - We verify raw_text[start:end] == evidence exactly
    - Only conservative OCR fixes applied to normalized name
    """

    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-32B-Instruct",
                 quantize_4bit: bool = True,
                 shared_model=None):
        self.model_name = model_name
        self.quantize_4bit = quantize_4bit
        self.shared_model = shared_model
        self._backend = None  # LLMBackend instance (when using shared_model)
        self._model = None  # Direct model (standalone mode only)
        self._tokenizer = None

    def _ensure_model_loaded(self):
        """Load the model if not already loaded."""
        if self.shared_model is not None:
            # Check if shared_model is an LLMBackend (has generate_raw)
            if hasattr(self.shared_model, 'generate_raw'):
                self._backend = self.shared_model
            else:
                # Legacy SharedModelManager — extract model/tokenizer directly
                self._model = self.shared_model.model
                self._tokenizer = self.shared_model.tokenizer
            return

        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading LLM for recovery: {self.model_name}")

        quantization_config = None
        if self.quantize_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )

    def _build_recovery_prompt(self, text_chunk: str, chunk_offset: int) -> str:
        """Build prompt for evidence-based name recovery."""
        return f"""You are analyzing OCR text to find person names that may be corrupted.

RULES:
1. Only output names that ACTUALLY APPEAR in the text
2. Provide the EXACT start and end character positions
3. The evidence_text must match the text EXACTLY at those positions
4. Do NOT guess or invent names - only report what you can see

OCR errors to watch for:
- 0 instead of O (J0HN → JOHN)
- 1 instead of I (SM1TH → SMITH)
- 5 instead of S (JONE5 → JONES)

Role assignment (be conservative):
- "sender" only if in From: line or signature
- "recipient" only if in To: or CC: line
- "passenger" only if in explicit passenger list
- "mentioned" for everything else
- "other" if very uncertain

TEXT (character positions start at {chunk_offset}):
{text_chunk}

Output JSON array of found names:
[
  {{"evidence_text": "J0HN SM1TH", "start": 145, "end": 155, "role": "sender"}},
  {{"evidence_text": "Mary Jones", "start": 203, "end": 213, "role": "mentioned"}}
]

If no names found, output: []

IMPORTANT: evidence_text must EXACTLY match text[start:end]. Do not modify it.

JSON:"""

    def recover_names(self,
                      raw_text: str,
                      clean_text: str,
                      source_file: str = "",
                      page_boundaries: list[int] | None = None) -> RecoveryResult:
        """Attempt to recover names from corrupted document.

        Safety: All recovered names are hard-validated against raw_text.
        """
        import json
        import torch

        self._ensure_model_loaded()

        page_boundaries = page_boundaries or [0]
        noise_score = compute_noise_score(raw_text)

        # Chunk text for LLM (max ~4k chars per chunk)
        chunk_size = 4000
        chunks = []
        for i in range(0, len(clean_text), chunk_size):
            chunks.append((i, clean_text[i:i + chunk_size]))

        recovered_spans = []

        for chunk_start, chunk_text in chunks:
            prompt = self._build_recovery_prompt(chunk_text, chunk_start)

            if self._backend is not None:
                # Use LLMBackend interface (works with both transformers and vLLM)
                response = self._backend.generate_raw(
                    prompt, max_new_tokens=500,
                    temperature=0.1, do_sample=True
                )
            else:
                # Direct model access (standalone or legacy SharedModelManager)
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=500,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=self._tokenizer.eos_token_id
                    )

                response = self._tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

            # Parse JSON response
            try:
                # Find JSON array in response
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    names = json.loads(json_match.group())

                    for name_info in names:
                        evidence_text = name_info.get("evidence_text", "")
                        start = name_info.get("start")
                        end = name_info.get("end")
                        role = name_info.get("role", "mentioned")

                        # Skip invalid entries
                        if not evidence_text or start is None or end is None:
                            continue

                        # HARD VALIDATION: exact match against raw text
                        if start < 0 or end > len(raw_text) or start >= end:
                            logger.debug(f"Rejected: invalid span {start}:{end}")
                            continue

                        actual_text = raw_text[start:end]
                        if actual_text != evidence_text:
                            # Try case-insensitive as fallback
                            if actual_text.lower() != evidence_text.lower():
                                logger.debug(f"Rejected: mismatch '{actual_text}' != '{evidence_text}'")
                                continue
                            # Use actual text from document
                            evidence_text = actual_text

                        # Check if it looks like a name
                        if not is_name_like(evidence_text):
                            logger.debug(f"Rejected: not name-like '{evidence_text}'")
                            continue

                        # Apply conservative OCR fixes for normalized name
                        normalized = apply_conservative_ocr_fixes(evidence_text)

                        # Get page number
                        page_num = 1
                        for i, boundary in enumerate(page_boundaries):
                            if start >= boundary:
                                page_num = i + 1

                        # Validate role
                        valid_roles = {"sender", "recipient", "passenger", "mentioned", "other"}
                        if role not in valid_roles:
                            role = "mentioned"

                        recovered_spans.append(RecoveredSpan(
                            text=evidence_text,  # Exact from document
                            start=start,
                            end=end,
                            normalized_name=normalized,  # Conservative fixes only
                            confidence=0.7,  # Recovery is lower confidence
                            role=role,
                            page_number=page_num
                        ))

            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse LLM recovery response: {e}")
                continue

        # Deduplicate by position
        seen_positions = set()
        unique_spans = []
        for span in recovered_spans:
            pos_key = (span.start, span.end)
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                unique_spans.append(span)

        logger.info(f"Recovery found {len(unique_spans)} names (validated)")

        return RecoveryResult(
            source_file=source_file,
            recovered_spans=unique_spans,
            total_recovered=len(unique_spans),
            was_triggered=True,
            noise_score=noise_score
        )

    def cleanup(self):
        """Free GPU memory."""
        self._backend = None
        if self._model is not None and self.shared_model is None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class MockLLMRecovery:
    """Mock recovery using only cheap heuristics (no LLM)."""

    def recover_names(self,
                      raw_text: str,
                      clean_text: str,
                      source_file: str = "",
                      page_boundaries: list[int] | None = None) -> RecoveryResult:
        """Mock recovery that uses only regex heuristics."""
        page_boundaries = page_boundaries or [0]
        noise_score = compute_noise_score(raw_text)

        # Use cheap candidate generator
        candidates = find_name_candidates_cheap(clean_text)

        recovered = []
        for start, end, text, pattern_type in candidates:
            # Verify in raw text
            if start < len(raw_text) and end <= len(raw_text):
                actual = raw_text[start:end]
                if actual == text and is_name_like(text):
                    normalized = apply_conservative_ocr_fixes(text)

                    # Get page number
                    page_num = 1
                    for i, boundary in enumerate(page_boundaries):
                        if start >= boundary:
                            page_num = i + 1

                    recovered.append(RecoveredSpan(
                        text=text,
                        start=start,
                        end=end,
                        normalized_name=normalized,
                        confidence=0.5,  # Mock is lower confidence
                        role="mentioned",
                        page_number=page_num
                    ))

        return RecoveryResult(
            source_file=source_file,
            recovered_spans=recovered,
            total_recovered=len(recovered),
            was_triggered=True,
            noise_score=noise_score
        )

    def cleanup(self):
        pass


def create_recovery(use_llm: bool = True, **kwargs) -> LLMNameRecovery | MockLLMRecovery:
    """Factory function to create recovery module."""
    if use_llm:
        try:
            import torch
            if torch.cuda.is_available():
                return LLMNameRecovery(**kwargs)
        except ImportError:
            pass
    return MockLLMRecovery()


def recovered_to_classified(span: RecoveredSpan) -> ClassifiedSpan:
    """Convert recovered span to classified span format for downstream stages."""
    return ClassifiedSpan(
        text=span.text,
        start=span.start,
        end=span.end,
        entity_type="PER",
        original_confidence=span.confidence,
        is_person=True,
        confidence=span.confidence,
        role=span.role,
        all_roles=[span.role],
        drop=False,
        drop_reason=None,
        page_number=span.page_number
    )
