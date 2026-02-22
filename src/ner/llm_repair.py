"""Stage 4: LLM repair for corrupted name spans.

This module uses an LLM to repair OCR-corrupted name spans.
It produces a normalized_name while preserving the original_text.

IMPORTANT: This is ONLY called for spans that need repair
(detected by hard_validator.needs_repair()). This keeps LLM costs low.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger

from .hard_validator import ValidatedSpan


def basic_normalize(name: str) -> str:
    """Basic name normalization without LLM.

    Handles:
    - Extra whitespace collapse
    - "LASTNAME, FIRSTNAME" -> "Firstname Lastname"
    - ALL-CAPS -> Title Case
    """
    name = ' '.join(name.split())
    if ',' in name and name.count(',') == 1:
        parts = [p.strip() for p in name.split(',')]
        if len(parts) == 2 and all(p for p in parts):
            name = f"{parts[1]} {parts[0]}"
    if len(name) > 3 and name.isupper():
        name = name.title()
    return name


@dataclass
class RepairedName:
    """A name with both original and repaired versions."""
    original_text: str      # Exact text from source document
    normalized_name: str    # Human-readable repaired name
    start: int              # Character offset in source
    end: int                # Character offset in source
    role: str               # Role classification
    all_roles: list[str]    # All applicable roles
    confidence: float       # Combined confidence
    page_number: int | None
    was_repaired: bool      # Whether LLM repair was applied


@dataclass
class RepairResult:
    """Result of repair stage."""
    source_file: str
    repaired_names: list[RepairedName]
    total_names: int
    repaired_count: int     # How many needed repair


@dataclass
class BatchRepairItem:
    """Input item for inter-document batched repair."""
    validated_spans: list[ValidatedSpan]
    source_file: str


class LLMNameRepairer:
    """LLM-based name repair for OCR-corrupted spans.

    This module repairs common OCR errors:
    - 0 → O (zero to letter O)
    - 1 → I or l (one to letter I/l)
    - 5 → S (five to letter S)
    - Weird punctuation normalization
    - Case normalization

    Only called for spans that need_repair() returned True.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct"

    SYSTEM_PROMPT = """You are an OCR correction expert specializing in fixing person names.

You will receive person names that have OCR errors. Common errors include:
- 0 (zero) confused with O (letter)
- 1 (one) confused with I or l (letters)
- 5 (five) confused with S (letter)
- Weird punctuation like /, \\, |, ~
- Mixed case issues

Your task is to output the corrected, human-readable version of each name.

Rules:
1. Fix obvious OCR character confusions
2. Normalize to proper title case (e.g., "JOHN SMITH" → "John Smith")
3. Remove OCR artifacts but preserve intentional punctuation (hyphens in names, etc.)
4. If uncertain, prefer the most common spelling
5. Do NOT change names that already look correct

Output ONLY valid JSON array:
[
    {"index": 0, "corrected": "John Smith"},
    {"index": 1, "corrected": "Sarah O'Brien"}
]
"""

    REPAIR_PROMPT = """Fix the OCR errors in these person names:

{names_json}

Return ONLY the JSON array with corrected names."""

    def __init__(self,
                 model_name: str | None = None,
                 device: str | None = None,
                 quantize_4bit: bool = True,
                 shared_model=None):
        """Initialize LLM repairer.

        Args:
            model_name: HuggingFace model name.
            device: Device to use.
            quantize_4bit: Use 4-bit quantization.
            shared_model: LLM backend (LLMBackend or SharedModelManager).
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.quantize_4bit = quantize_4bit
        self._shared_model = shared_model

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load the model."""
        # If using shared model, skip loading our own
        if self._shared_model is not None:
            return

        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading repair model: {self.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if self.quantize_4bit and self.device == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self._model.eval()
        logger.info("Repair model loaded")

    # Token budget constants
    TOKENS_PER_NAME_OUTPUT = 15  # Estimated output tokens per repaired name
    MAX_OUTPUT_TOKENS = 1024  # Conservative ceiling for repair
    MAX_INPUT_TOKENS = 24000  # Leave headroom in 32k context

    def _estimate_batch_tokens(self, spans: list[ValidatedSpan]) -> int:
        """Estimate input token count for a repair batch.

        Uses real tokenizer when available, heuristic otherwise.
        """
        total_chars = sum(len(s.verified_text) + 30 for s in spans)  # text + JSON overhead
        total_chars += len(self.SYSTEM_PROMPT) + len(self.REPAIR_PROMPT)

        # Prefer real tokenizer if available
        tokenizer = None
        if self._shared_model is not None and self._shared_model.is_loaded:
            tokenizer = self._shared_model.tokenizer
        elif self._tokenizer is not None:
            tokenizer = self._tokenizer

        if tokenizer is not None:
            try:
                sample = " ".join(s.verified_text for s in spans)
                tokens = tokenizer.encode(sample, add_special_tokens=False)
                return len(tokens) + 300  # overhead
            except Exception:
                pass

        return int(total_chars / 4 * 1.5)

    def repair_names(self,
                     validated_spans: list[ValidatedSpan],
                     source_file: str = "",
                     batch_size: int = 80) -> RepairResult:
        """Repair corrupted names in validated spans.

        Args:
            validated_spans: Spans that passed hard validation.
            source_file: Source file name.
            batch_size: Max names to repair at once (may be reduced by token budget).

        Returns:
            RepairResult with repaired names.
        """
        # Separate spans that need repair from those that don't
        needs_repair_spans = [s for s in validated_spans if s.needs_repair]
        clean_spans = [s for s in validated_spans if not s.needs_repair]

        # Process clean spans (just convert format)
        repaired_names = []
        for span in clean_spans:
            repaired_names.append(RepairedName(
                original_text=span.verified_text,
                normalized_name=self._basic_normalize(span.verified_text),
                start=span.classified.span.start,
                end=span.classified.span.end,
                role=span.classified.role,
                all_roles=span.classified.all_roles,
                confidence=span.classified.span.confidence,
                page_number=span.classified.span.page_number,
                was_repaired=False
            ))

        # If nothing needs repair, return early
        if not needs_repair_spans:
            return RepairResult(
                source_file=source_file,
                repaired_names=repaired_names,
                total_names=len(repaired_names),
                repaired_count=0
            )

        # Load model for repair
        self._load_model()

        # Process repair batches with token budgeting
        i = 0
        while i < len(needs_repair_spans):
            end = min(i + batch_size, len(needs_repair_spans))
            batch = needs_repair_spans[i:end]

            # Check output budget
            output_tokens = self.TOKENS_PER_NAME_OUTPUT * len(batch) + 50
            if output_tokens > self.MAX_OUTPUT_TOKENS:
                max_names = (self.MAX_OUTPUT_TOKENS - 50) // self.TOKENS_PER_NAME_OUTPUT
                batch = needs_repair_spans[i:i + max_names]
                end = i + len(batch)

            # Check input token budget
            est_tokens = self._estimate_batch_tokens(batch)
            if est_tokens > self.MAX_INPUT_TOKENS and len(batch) > 1:
                half = len(batch) // 2
                batch = needs_repair_spans[i:i + half]
                end = i + len(batch)

            batch_repaired = self._repair_batch(batch)
            repaired_names.extend(batch_repaired)
            i = end

        # Sort by start offset
        repaired_names.sort(key=lambda n: n.start)

        return RepairResult(
            source_file=source_file,
            repaired_names=repaired_names,
            total_names=len(repaired_names),
            repaired_count=len(needs_repair_spans)
        )

    def _build_repair_prompt(self, spans: list[ValidatedSpan]) -> tuple[list[dict], int]:
        """Build repair prompt messages and compute max_new_tokens.

        Args:
            spans: Validated spans that need repair.

        Returns:
            Tuple of (messages, max_new_tokens).
        """
        names_data = []
        for i, span in enumerate(spans):
            names_data.append({
                "index": i,
                "text": span.verified_text
            })

        names_json = json.dumps(names_data)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.REPAIR_PROMPT.format(names_json=names_json)}
        ]

        max_new_tokens = min(
            self.MAX_OUTPUT_TOKENS,
            self.TOKENS_PER_NAME_OUTPUT * len(spans) + 50
        )

        return messages, max_new_tokens

    def _repair_batch(self, spans: list[ValidatedSpan]) -> list[RepairedName]:
        """Repair a batch of corrupted spans."""
        messages, max_new_tokens = self._build_repair_prompt(spans)

        # Use shared model if available, otherwise use own model
        if self._shared_model is not None:
            response = self._shared_model.generate(messages, max_new_tokens=max_new_tokens)
        else:
            # Tokenize
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt"
            )

            if isinstance(input_ids, torch.Tensor):
                pass
            elif hasattr(input_ids, 'input_ids'):
                input_ids = input_ids.input_ids
            elif isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids] if isinstance(input_ids[0], int) else input_ids)

            device = next(self._model.parameters()).device
            input_ids = input_ids.to(device)
            input_length = input_ids.shape[1]

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            response = self._tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            ).strip()

        return self._parse_repairs(response, spans)

    def _parse_repairs(self,
                       response: str,
                       spans: list[ValidatedSpan]) -> list[RepairedName]:
        """Parse repair response."""
        repaired = []

        try:
            # Find JSON array
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                data = json.loads(json_match.group())
                repair_map = {item.get('index', i): item.get('corrected', '')
                              for i, item in enumerate(data)}
            else:
                repair_map = {}

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse repair response: {e}")
            repair_map = {}

        for i, span in enumerate(spans):
            # Get repaired text or fall back to basic normalization
            if i in repair_map and repair_map[i]:
                normalized = repair_map[i]
            else:
                normalized = self._basic_normalize(span.verified_text)

            repaired.append(RepairedName(
                original_text=span.verified_text,
                normalized_name=normalized,
                start=span.classified.span.start,
                end=span.classified.span.end,
                role=span.classified.role,
                all_roles=span.classified.all_roles,
                confidence=span.classified.span.confidence,
                page_number=span.classified.span.page_number,
                was_repaired=True
            ))

        return repaired

    def repair_names_batch(
        self,
        items: list[BatchRepairItem],
        inter_batch_size: int = 8,
        repair_batch_size: int = 80,
    ) -> dict[str, RepairResult]:
        """Repair names from multiple documents in batched LLM calls.

        Each document gets its own independent prompt. Multiple prompts are
        batched into a single generate() call for GPU efficiency.

        Args:
            items: List of documents with their validated spans.
            inter_batch_size: Max prompts per batched generate() call.
            repair_batch_size: Max names per repair prompt (intra-doc).

        Returns:
            Dict mapping source_file to RepairResult.
        """
        if not items:
            return {}

        # Separate clean/repair spans per document
        # For docs with no repair spans, just convert clean spans
        doc_clean: dict[int, list[RepairedName]] = {}
        doc_repair_spans: dict[int, list[ValidatedSpan]] = {}

        for doc_idx, item in enumerate(items):
            clean_names = []
            repair_spans = []
            for span in item.validated_spans:
                if span.needs_repair:
                    repair_spans.append(span)
                else:
                    clean_names.append(RepairedName(
                        original_text=span.verified_text,
                        normalized_name=self._basic_normalize(span.verified_text),
                        start=span.classified.span.start,
                        end=span.classified.span.end,
                        role=span.classified.role,
                        all_roles=span.classified.all_roles,
                        confidence=span.classified.span.confidence,
                        page_number=span.classified.span.page_number,
                        was_repaired=False,
                    ))
            doc_clean[doc_idx] = clean_names
            doc_repair_spans[doc_idx] = repair_spans

        # Collect repair prompts across docs
        # Each entry: (doc_idx, spans_slice, messages, max_new_tokens)
        prompt_entries: list[tuple[int, list[ValidatedSpan], list[dict], int]] = []

        for doc_idx, repair_spans in doc_repair_spans.items():
            if not repair_spans:
                continue

            # Intra-doc splitting by token budget
            batch_size = repair_batch_size
            i = 0
            while i < len(repair_spans):
                end = min(i + batch_size, len(repair_spans))
                batch = repair_spans[i:end]

                output_tokens = self.TOKENS_PER_NAME_OUTPUT * len(batch) + 50
                if output_tokens > self.MAX_OUTPUT_TOKENS:
                    max_names = (self.MAX_OUTPUT_TOKENS - 50) // self.TOKENS_PER_NAME_OUTPUT
                    batch = repair_spans[i:i + max_names]
                    end = i + len(batch)

                est_tokens = self._estimate_batch_tokens(batch)
                if est_tokens > self.MAX_INPUT_TOKENS and len(batch) > 1:
                    half = len(batch) // 2
                    batch = repair_spans[i:i + half]
                    end = i + len(batch)

                messages, max_new_tokens = self._build_repair_prompt(batch)
                prompt_entries.append((doc_idx, batch, messages, max_new_tokens))
                i = end

        # If no repair prompts, assemble results from clean spans only
        if not prompt_entries:
            results = {}
            for doc_idx, item in enumerate(items):
                names = doc_clean.get(doc_idx, [])
                names.sort(key=lambda n: n.start)
                results[item.source_file] = RepairResult(
                    source_file=item.source_file,
                    repaired_names=names,
                    total_names=len(names),
                    repaired_count=0,
                )
            return results

        self._load_model()

        if self._shared_model is None:
            # No shared model — fall back to sequential repair
            results = {}
            for item in items:
                results[item.source_file] = self.repair_names(
                    validated_spans=item.validated_spans,
                    source_file=item.source_file,
                )
            return results

        # Sort by max_new_tokens (output budget) for generation-budget bucketing.
        # Same principle as classifier: group similar output budgets to avoid
        # a small repair batch forcing high max_new_tokens on the whole batch.
        prompt_entries.sort(key=lambda e: e[3])  # sort by max_new_tokens

        doc_repaired: dict[int, list[RepairedName]] = {i: [] for i in range(len(items))}

        for chunk_start in range(0, len(prompt_entries), inter_batch_size):
            chunk = prompt_entries[chunk_start:chunk_start + inter_batch_size]
            messages_list = [e[2] for e in chunk]
            chunk_max_tokens = max(e[3] for e in chunk)

            logger.debug(
                f"Repair batch: {len(chunk)} prompts, "
                f"max_new_tokens={chunk_max_tokens}"
            )

            responses = self._shared_model.generate_batch(
                messages_list, max_new_tokens=chunk_max_tokens
            )

            for entry, response in zip(chunk, responses):
                doc_idx, spans_slice, _, _ = entry
                try:
                    repaired = self._parse_repairs(response, spans_slice)
                except Exception as e:
                    logger.warning(
                        f"Parse failed for repair batch of {items[doc_idx].source_file}, "
                        f"retrying single: {e}"
                    )
                    try:
                        messages, max_new_tokens = self._build_repair_prompt(spans_slice)
                        retry_response = self._shared_model.generate(
                            messages, max_new_tokens=max_new_tokens
                        )
                        repaired = self._parse_repairs(retry_response, spans_slice)
                    except Exception:
                        repaired = [
                            RepairedName(
                                original_text=s.verified_text,
                                normalized_name=self._basic_normalize(s.verified_text),
                                start=s.classified.span.start,
                                end=s.classified.span.end,
                                role=s.classified.role,
                                all_roles=s.classified.all_roles,
                                confidence=s.classified.span.confidence,
                                page_number=s.classified.span.page_number,
                                was_repaired=True,
                            )
                            for s in spans_slice
                        ]
                doc_repaired[doc_idx].extend(repaired)

        # Assemble final results
        results = {}
        for doc_idx, item in enumerate(items):
            all_names = doc_clean.get(doc_idx, []) + doc_repaired.get(doc_idx, [])
            all_names.sort(key=lambda n: n.start)
            repaired_count = len(doc_repair_spans.get(doc_idx, []))
            results[item.source_file] = RepairResult(
                source_file=item.source_file,
                repaired_names=all_names,
                total_names=len(all_names),
                repaired_count=repaired_count,
            )

        return results

    def _basic_normalize(self, name: str) -> str:
        """Basic normalization without LLM."""
        return basic_normalize(name)

    def cleanup(self):
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MockLLMRepairer:
    """Mock repairer for testing without GPU."""

    def repair_names_batch(
        self,
        items: list[BatchRepairItem],
        inter_batch_size: int = 8,
    ) -> dict[str, RepairResult]:
        """Batch repair using mock character mapping."""
        results = {}
        for item in items:
            results[item.source_file] = self.repair_names(
                validated_spans=item.validated_spans,
                source_file=item.source_file,
            )
        return results

    # Common OCR repair mappings
    OCR_FIXES = {
        '0': 'O',
        '1': 'I',
        '5': 'S',
        '/': '',
        '\\': '',
        '|': 'l',
        '~': '',
    }

    def repair_names(self,
                     validated_spans: list[ValidatedSpan],
                     source_file: str = "",
                     **kwargs) -> RepairResult:
        """Repair names using simple character mapping."""
        repaired_names = []

        for span in validated_spans:
            original = span.verified_text

            if span.needs_repair:
                # Apply OCR fixes
                fixed = original
                for bad, good in self.OCR_FIXES.items():
                    fixed = fixed.replace(bad, good)

                # Title case if all caps
                if fixed.isupper():
                    fixed = fixed.title()

                # Clean up whitespace
                fixed = ' '.join(fixed.split())

                normalized = fixed
                was_repaired = True
            else:
                normalized = self._basic_normalize(original)
                was_repaired = False

            repaired_names.append(RepairedName(
                original_text=original,
                normalized_name=normalized,
                start=span.classified.span.start,
                end=span.classified.span.end,
                role=span.classified.role,
                all_roles=span.classified.all_roles,
                confidence=span.classified.span.confidence,
                page_number=span.classified.span.page_number,
                was_repaired=was_repaired
            ))

        repaired_count = sum(1 for n in repaired_names if n.was_repaired)

        return RepairResult(
            source_file=source_file,
            repaired_names=repaired_names,
            total_names=len(repaired_names),
            repaired_count=repaired_count
        )

    def _basic_normalize(self, name: str) -> str:
        """Basic normalization."""
        return basic_normalize(name)

    def cleanup(self):
        pass


def create_repairer(use_llm: bool = True, **kwargs):
    """Factory to create appropriate repairer.

    Args:
        use_llm: Whether to use actual LLM.
        **kwargs: Arguments passed to repairer.

    Returns:
        Repairer instance.
    """
    if use_llm and torch.cuda.is_available():
        return LLMNameRepairer(**kwargs)
    else:
        logger.warning("Using mock repairer (no GPU or use_llm=False)")
        return MockLLMRepairer()


if __name__ == "__main__":
    # Test mock repairer
    from .xlmr_extractor import NERSpan
    from .llm_classifier import ClassifiedSpan
    from .hard_validator import ValidatedSpan

    # Create test spans
    test_spans = [
        ValidatedSpan(
            classified=ClassifiedSpan(
                span=NERSpan(text="J0HN SM1TH", start=0, end=10, entity_type="PER", confidence=0.9),
                is_person=True, role="mentioned", all_roles=["mentioned"],
                drop=False, drop_reason=None, classification_confidence=0.8
            ),
            verified_text="J0HN SM1TH",
            needs_repair=True
        ),
        ValidatedSpan(
            classified=ClassifiedSpan(
                span=NERSpan(text="Mary Brown", start=20, end=30, entity_type="PER", confidence=0.95),
                is_person=True, role="sender", all_roles=["sender"],
                drop=False, drop_reason=None, classification_confidence=0.9
            ),
            verified_text="Mary Brown",
            needs_repair=False
        ),
        ValidatedSpan(
            classified=ClassifiedSpan(
                span=NERSpan(text="5ARAH J0NE5", start=40, end=51, entity_type="PER", confidence=0.85),
                is_person=True, role="recipient", all_roles=["recipient"],
                drop=False, drop_reason=None, classification_confidence=0.8
            ),
            verified_text="5ARAH J0NE5",
            needs_repair=True
        ),
    ]

    repairer = create_repairer(use_llm=False)
    result = repairer.repair_names(test_spans, source_file="test.pdf")

    print(f"Repaired {result.repaired_count} of {result.total_names} names:")
    for name in result.repaired_names:
        repair_marker = "[REPAIRED]" if name.was_repaired else "[CLEAN]"
        print(f"  {repair_marker} '{name.original_text}' -> '{name.normalized_name}'")
