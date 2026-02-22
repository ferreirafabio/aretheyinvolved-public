"""Stage 2: LLM classifier for NER span classification.

This module classifies NER spans using an LLM to determine:
1. Whether the span is actually a person name (is_person)
2. The person's role in the document (sender, recipient, etc.)

IMPORTANT: This is a closed-set classifier. It CANNOT modify span text.
It only classifies spans that were extracted by NER.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger

from .xlmr_extractor import NERSpan


@dataclass
class ClassifiedSpan:
    """A NER span with classification results."""
    span: NERSpan           # Original NER span
    is_person: bool         # Whether this is actually a person name
    role: str               # Role classification (sender, recipient, etc.)
    all_roles: list[str]    # All applicable roles
    drop: bool              # Whether to drop this span
    drop_reason: str | None # Why it should be dropped
    classification_confidence: float  # LLM's confidence in classification


@dataclass
class ClassificationResult:
    """Result of classifying NER spans."""
    source_file: str
    classified_spans: list[ClassifiedSpan]
    total_spans: int
    person_spans: int       # Spans classified as persons
    dropped_spans: int      # Spans marked for dropping
    document_type: str | None


@dataclass
class BatchClassifyItem:
    """Input item for inter-document batched classification."""
    document_text: str
    spans: list[NERSpan]
    source_file: str


class LLMSpanClassifier:
    """LLM-based classifier for NER spans.

    Uses Qwen to classify NER spans as:
    - Is this actually a person name?
    - What is their role? (sender, recipient, mentioned, passenger, other)

    This is a CLOSED-SET classifier - it cannot modify or create new spans,
    only classify the spans that NER extracted.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct"

    SYSTEM_PROMPT = """You are a document analysis expert. Your task is to classify named entity spans.

You will receive:
1. The full document text for context
2. A list of potential person name spans extracted by NER

For each span, determine:
1. is_person: Is this actually a person's name? (true/false)
2. role: If a person, what is their role? (sender, recipient, mentioned, passenger, other)
3. all_roles: All roles that apply to this person (list)
4. drop: Should this span be dropped? (true/false)
5. reason: If dropping, why? (not_a_person, organization, location, incomplete, ocr_garbage)

IMPORTANT RULES:
- You CANNOT modify the span text - only classify it
- You CANNOT add new spans - only classify what's given
- If a span is an organization (IBM, AT&T), mark drop=true, reason="organization"
- If a span is a location (New York, Paris), mark drop=true, reason="location"
- If a span is incomplete (just "John" with no last name visible), keep it if confident
- If a span is OCR garbage (random chars), mark drop=true, reason="ocr_garbage"

Role definitions (in priority order):
1. sender: Author or sender of document (From field, signature, author byline)
2. recipient: Recipient of document (To/CC fields, addressee)
3. passenger: Person listed as passenger in flight log
4. mentioned: Person named in document text
5. other: Role unclear or doesn't fit above

Output ONLY valid JSON array:
[
    {
        "span_index": 0,
        "is_person": true,
        "role": "sender",
        "all_roles": ["sender", "mentioned"],
        "drop": false,
        "reason": null
    },
    {
        "span_index": 1,
        "is_person": false,
        "role": null,
        "all_roles": [],
        "drop": true,
        "reason": "organization"
    }
]
"""

    CLASSIFICATION_PROMPT = """Classify each of the following NER spans from the document.

Document text:
---
{document_text}
---

NER spans to classify:
{spans_json}

Return ONLY the JSON array with classifications for each span."""

    def __init__(self,
                 model_name: str | None = None,
                 device: str | None = None,
                 quantize_4bit: bool = True,
                 shared_model=None,
                 strict: bool = False):
        """Initialize LLM classifier.

        Args:
            model_name: HuggingFace model name.
            device: Device to use ('cuda', 'cpu', or None for auto).
            quantize_4bit: Use 4-bit quantization for memory efficiency.
            shared_model: LLM backend (LLMBackend or SharedModelManager).
            strict: If True, raise on batch-size mismatches (use in CI/tests).
                    If False, warn and fall back to defaults (use in prod).
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.quantize_4bit = quantize_4bit
        self.strict = strict
        self._shared_model = shared_model

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None
        self._tokenizer = None

        # Diagnostic counters (cumulative across all calls)
        self.parse_fail_count = 0
        self.json_repair_used_count = 0
        self.defaults_count = 0

    def _load_model(self):
        """Lazy load the model and tokenizer."""
        # If using shared model, skip loading our own
        if self._shared_model is not None:
            return

        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading classifier model: {self.model_name}")

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
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        except Exception as e:
            if "flash_attn" in str(e):
                model_kwargs.pop("attn_implementation", None)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            else:
                raise

        self._model.eval()
        logger.info(f"Classifier model loaded")

    # Token budget constants
    MAX_INPUT_TOKENS = 24000  # Leave headroom in 32k context
    TOKENS_PER_SPAN_OUTPUT = 50  # Estimated output tokens per classified span (compact JSON ~30-35, margin for safety)
    MAX_OUTPUT_TOKENS = 4096  # Qwen 32B max_new_tokens ceiling (was 2048, caused truncation for >25 spans)

    def _estimate_batch_tokens(self, document_text: str, spans: list[NERSpan]) -> int:
        """Estimate input token count for a batch.

        Prefers real tokenization when shared model is loaded,
        falls back to heuristic (len/4 with 1.5x safety margin for OCR text).
        """
        # Build approximate prompt text
        spans_text_len = sum(len(s.text) + 200 for s in spans)  # text + context + JSON overhead
        prompt_len = len(document_text[:8000]) + spans_text_len + len(self.SYSTEM_PROMPT)

        # Prefer real tokenizer if available
        tokenizer = None
        if self._shared_model is not None and self._shared_model.is_loaded:
            tokenizer = self._shared_model.tokenizer
        elif self._tokenizer is not None:
            tokenizer = self._tokenizer

        if tokenizer is not None:
            try:
                # Quick estimate: tokenize just the document + span texts
                sample = document_text[:8000] + " ".join(s.text for s in spans)
                tokens = tokenizer.encode(sample, add_special_tokens=False)
                # Add overhead for system prompt + JSON formatting
                return len(tokens) + 500
            except Exception:
                pass

        # Heuristic fallback: chars/4 with 1.5x safety margin
        return int(prompt_len / 4 * 1.5)

    def classify_spans(self,
                       document_text: str,
                       spans: list[NERSpan],
                       source_file: str = "",
                       batch_size: int = 50) -> ClassificationResult:
        """Classify NER spans with token-budgeted batching.

        Args:
            document_text: Full document text for context.
            spans: NER spans to classify.
            source_file: Source file name.
            batch_size: Max number of spans to classify at once (may be reduced by token budget).

        Returns:
            ClassificationResult with classified spans.
        """
        if not spans:
            return ClassificationResult(
                source_file=source_file,
                classified_spans=[],
                total_spans=0,
                person_spans=0,
                dropped_spans=0,
                document_type=None
            )

        self._load_model()

        all_classified = []
        batch_count = 0
        llm_classifications_total = 0
        num_defaulted = 0

        # Process in token-budgeted batches
        i = 0
        while i < len(spans):
            # Start with requested batch_size, reduce if token budget exceeded
            end = min(i + batch_size, len(spans))
            batch = spans[i:end]

            # Check if output would exceed max_new_tokens ceiling
            output_tokens = self.TOKENS_PER_SPAN_OUTPUT * len(batch) + 100
            if output_tokens > self.MAX_OUTPUT_TOKENS:
                # Reduce batch to fit output budget
                max_spans = (self.MAX_OUTPUT_TOKENS - 100) // self.TOKENS_PER_SPAN_OUTPUT
                batch = spans[i:i + max_spans]
                end = i + len(batch)

            # Check input token budget
            est_tokens = self._estimate_batch_tokens(document_text, batch)
            if est_tokens > self.MAX_INPUT_TOKENS and len(batch) > 1:
                # Split in half and retry
                half = len(batch) // 2
                batch = spans[i:i + half]
                end = i + len(batch)

            classified = self._classify_batch(document_text, batch, start_index=i)

            # Half-batch retry on >50% defaults
            batch_defaulted = sum(1 for c in classified if c.classification_confidence == 0.5)
            if batch_defaulted > len(batch) * 0.5 and len(batch) > 1:
                logger.warning(
                    f"High default rate ({batch_defaulted}/{len(batch)}), retrying with half batch"
                )
                half = len(batch) // 2
                classified = []
                c1 = self._classify_batch(document_text, spans[i:i + half], start_index=i)
                c2 = self._classify_batch(document_text, spans[i + half:end], start_index=i + half)
                classified = c1 + c2

            all_classified.extend(classified)
            batch_count += 1
            i = end

        # Count statistics
        person_count = sum(1 for c in all_classified if c.is_person and not c.drop)
        dropped_count = sum(1 for c in all_classified if c.drop)
        num_defaulted = sum(1 for c in all_classified if c.classification_confidence == 0.5)
        llm_classifications_total = len(all_classified) - num_defaulted

        # Runtime metrics: makes batch-index bugs and LLM omissions visible
        logger.info(
            f"classify_spans metrics | file={source_file} "
            f"ner_spans_total={len(spans)} "
            f"llm_classifications_total={llm_classifications_total} "
            f"num_defaulted={num_defaulted} "
            f"batch_count={batch_count}"
        )

        return ClassificationResult(
            source_file=source_file,
            classified_spans=all_classified,
            total_spans=len(all_classified),
            person_spans=person_count,
            dropped_spans=dropped_count,
            document_type=self._detect_document_type(document_text)
        )

    def _build_classify_prompt(self, document_text: str, spans: list[NERSpan]) -> tuple[list[dict], int]:
        """Build classification prompt messages and compute max_new_tokens.

        Args:
            document_text: Full document text for context.
            spans: NER spans to classify.

        Returns:
            Tuple of (messages, max_new_tokens).
        """
        # Build spans JSON — always 0-based indices
        spans_data = []
        for i, span in enumerate(spans):
            spans_data.append({
                "index": i,
                "text": span.text,
                "start": span.start,
                "end": span.end,
                "context": self._get_context(document_text, span.start, span.end, window=150)
            })

        spans_json = json.dumps(spans_data)

        # Truncate document if too long
        max_doc_length = 8000
        if len(document_text) > max_doc_length:
            doc_preview = document_text[:max_doc_length] + "\n...[truncated]..."
        else:
            doc_preview = document_text

        prompt = self.CLASSIFICATION_PROMPT.format(
            document_text=doc_preview,
            spans_json=spans_json
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        max_new_tokens = min(
            self.MAX_OUTPUT_TOKENS,
            self.TOKENS_PER_SPAN_OUTPUT * len(spans) + 100
        )

        return messages, max_new_tokens

    def _classify_batch(self,
                        document_text: str,
                        spans: list[NERSpan],
                        start_index: int = 0) -> list[ClassifiedSpan]:
        """Classify a batch of spans."""
        messages, max_new_tokens = self._build_classify_prompt(document_text, spans)

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

        # Parse response
        return self._parse_classifications(response, spans, start_index=start_index, document_text=document_text)

    def classify_spans_batch(
        self,
        items: list[BatchClassifyItem],
        inter_batch_size: int = 8,
        classifier_batch_size: int = 50,
    ) -> dict[str, ClassificationResult]:
        """Classify spans from multiple documents in batched LLM calls.

        Each document gets its own independent prompt. Multiple prompts are
        batched into a single generate() call for GPU efficiency.

        Args:
            items: List of documents with their spans to classify.
            inter_batch_size: Max prompts per batched generate() call.
            classifier_batch_size: Max names per classifier prompt (intra-doc).

        Returns:
            Dict mapping source_file to ClassificationResult.
        """
        if not items:
            return {}

        self._load_model()

        if self._shared_model is None:
            # No shared model — fall back to sequential classification
            results = {}
            for item in items:
                results[item.source_file] = self.classify_spans(
                    document_text=item.document_text,
                    spans=item.spans,
                    source_file=item.source_file,
                    batch_size=classifier_batch_size,
                )
            return results

        # Build one prompt per document (splitting large docs into intra-doc batches)
        # Each entry: (doc_idx, intra_batch_idx, spans_slice, messages, max_new_tokens)
        prompt_entries: list[tuple[int, int, list[NERSpan], list[dict], int]] = []

        for doc_idx, item in enumerate(items):
            if not item.spans:
                continue

            # Intra-doc splitting: if too many spans, split into sub-batches
            batch_size = classifier_batch_size
            i = 0
            intra_idx = 0
            while i < len(item.spans):
                end = min(i + batch_size, len(item.spans))
                span_slice = item.spans[i:end]

                # Token budget check — reduce if needed
                output_tokens = self.TOKENS_PER_SPAN_OUTPUT * len(span_slice) + 100
                if output_tokens > self.MAX_OUTPUT_TOKENS:
                    max_spans = (self.MAX_OUTPUT_TOKENS - 100) // self.TOKENS_PER_SPAN_OUTPUT
                    span_slice = item.spans[i:i + max_spans]
                    end = i + len(span_slice)

                est_tokens = self._estimate_batch_tokens(item.document_text, span_slice)
                if est_tokens > self.MAX_INPUT_TOKENS and len(span_slice) > 1:
                    half = len(span_slice) // 2
                    span_slice = item.spans[i:i + half]
                    end = i + len(span_slice)

                messages, max_new_tokens = self._build_classify_prompt(
                    item.document_text, span_slice
                )
                prompt_entries.append((doc_idx, intra_idx, span_slice, messages, max_new_tokens))
                intra_idx += 1
                i = end

        if not prompt_entries:
            # All items had empty spans
            return {
                item.source_file: ClassificationResult(
                    source_file=item.source_file, classified_spans=[],
                    total_spans=0, person_spans=0, dropped_spans=0,
                    document_type=None,
                )
                for item in items
            }

        # Sort by max_new_tokens (output budget) for generation-budget bucketing.
        # HF generate() runs ALL sequences until the longest hits EOS or max_new_tokens.
        # By grouping prompts with similar output budgets, each batch's max_new_tokens
        # is tight — a 10-span doc (400 tokens) won't share a batch with 80-span (2500).
        # True outliers naturally end up in their own small chunk.
        prompt_entries.sort(key=lambda e: e[4])  # sort by max_new_tokens

        all_responses: list[tuple[int, int, list[NERSpan], str]] = []

        for chunk_start in range(0, len(prompt_entries), inter_batch_size):
            chunk = prompt_entries[chunk_start:chunk_start + inter_batch_size]
            messages_list = [e[3] for e in chunk]
            # Tight budget: max of this chunk only (not global max)
            chunk_max_tokens = max(e[4] for e in chunk)
            chunk_min_tokens = min(e[4] for e in chunk)

            logger.debug(
                f"Classify batch: {len(chunk)} prompts, "
                f"max_new_tokens={chunk_max_tokens} "
                f"(range {chunk_min_tokens}-{chunk_max_tokens})"
            )

            responses = self._shared_model.generate_batch(
                messages_list, max_new_tokens=chunk_max_tokens
            )

            for entry, response in zip(chunk, responses):
                doc_idx, intra_idx, span_slice, _, _ = entry
                all_responses.append((doc_idx, intra_idx, span_slice, response))

        # Parse responses and assemble per-document results
        # Group by doc_idx
        doc_classified: dict[int, list[ClassifiedSpan]] = {i: [] for i in range(len(items))}

        for doc_idx, intra_idx, span_slice, response in all_responses:
            item = items[doc_idx]
            try:
                classified = self._parse_classifications(
                    response, span_slice, start_index=0,
                    document_text=item.document_text,
                )
            except Exception as e:
                logger.warning(
                    f"Parse failed for {item.source_file} batch {intra_idx}, "
                    f"retrying single: {e}"
                )
                # Per-doc error isolation: retry this single prompt
                try:
                    messages, max_new_tokens = self._build_classify_prompt(
                        item.document_text, span_slice
                    )
                    retry_response = self._shared_model.generate(
                        messages, max_new_tokens=max_new_tokens
                    )
                    classified = self._parse_classifications(
                        retry_response, span_slice, start_index=0,
                        document_text=item.document_text,
                    )
                except Exception:
                    classified = [self._default_classification(s) for s in span_slice]

            doc_classified[doc_idx].extend(classified)

        # Half-batch retry: if >50% defaults for a doc, re-run that doc individually
        for doc_idx, classified_list in doc_classified.items():
            if not classified_list:
                continue
            num_defaulted = sum(1 for c in classified_list if c.classification_confidence == 0.5)
            if num_defaulted > len(classified_list) * 0.5 and len(classified_list) > 1:
                item = items[doc_idx]
                logger.warning(
                    f"High default rate for {item.source_file} "
                    f"({num_defaulted}/{len(classified_list)}), re-running individually"
                )
                result = self.classify_spans(
                    document_text=item.document_text,
                    spans=item.spans,
                    source_file=item.source_file,
                    batch_size=50,
                )
                doc_classified[doc_idx] = result.classified_spans

        # Build final results
        results: dict[str, ClassificationResult] = {}
        for doc_idx, item in enumerate(items):
            classified_list = doc_classified.get(doc_idx, [])
            person_count = sum(1 for c in classified_list if c.is_person and not c.drop)
            dropped_count = sum(1 for c in classified_list if c.drop)

            results[item.source_file] = ClassificationResult(
                source_file=item.source_file,
                classified_spans=classified_list,
                total_spans=len(classified_list),
                person_spans=person_count,
                dropped_spans=dropped_count,
                document_type=self._detect_document_type(item.document_text),
            )

        return results

    def _parse_classifications(self,
                               response: str,
                               spans: list[NERSpan],
                               start_index: int = 0,
                               document_text: str = "") -> list[ClassifiedSpan]:
        """Parse LLM classification response."""
        classified = []

        try:
            # Find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                self.parse_fail_count += 1
                self.defaults_count += len(spans)
                logger.warning(f"No JSON array found in response")
                # Return default classifications
                return [self._default_classification(span) for span in spans]

            # Validate batch size: LLM output should match input span count
            if len(data) != len(spans):
                msg = (
                    f"LLM returned {len(data)} classifications for {len(spans)} spans "
                    f"(start_index={start_index})"
                )
                if self.strict:
                    raise ValueError(msg)
                logger.warning(f"{msg}. Using positional fallback.")

            # Map classifications by 0-based index (matches prompt "index": 0,1,2...)
            class_map = {item.get('span_index', i): item for i, item in enumerate(data)}

            for i, span in enumerate(spans):
                if i in class_map:
                    item = class_map[i]
                    drop = item.get('drop', False)
                    drop_reason = item.get('reason')

                    cspan = ClassifiedSpan(
                        span=span,
                        is_person=item.get('is_person', True),
                        role=item.get('role', 'mentioned') or 'mentioned',
                        all_roles=item.get('all_roles', ['mentioned']),
                        drop=drop,
                        drop_reason=drop_reason,
                        classification_confidence=0.8  # Default confidence
                    )

                    # NER confidence safeguard: high-confidence NER spans should not be dropped
                    if cspan.span.confidence > 0.85 and cspan.drop:
                        logger.warning(
                            f"Overriding drop for high-confidence NER span: "
                            f"'{cspan.span.text}' (NER conf={cspan.span.confidence:.2f}, "
                            f"drop_reason={cspan.drop_reason})"
                        )
                        cspan.drop = False
                        cspan.drop_reason = None

                    # Per-span drop logging
                    if cspan.drop:
                        ctx_snippet = ""
                        if document_text:
                            ctx_start = max(0, span.start - 30)
                            ctx_end = min(len(document_text), span.end + 30)
                            ctx_snippet = document_text[ctx_start:ctx_end]
                        logger.debug(
                            f"Dropping span: '{span.text}' "
                            f"(NER conf={span.confidence:.2f}, "
                            f"reason={cspan.drop_reason}, "
                            f"context='...{ctx_snippet}...')"
                        )

                    classified.append(cspan)
                else:
                    self.defaults_count += 1
                    classified.append(self._default_classification(span))

        except json.JSONDecodeError as e:
            self.parse_fail_count += 1
            logger.warning(f"JSON parse failed: {e}. Attempting truncation repair...")
            repaired = self._try_repair_json(response, spans)
            if repaired is not None:
                self.json_repair_used_count += 1
                return repaired
            logger.error(f"JSON repair failed, defaulting all {len(spans)} spans")
            self.defaults_count += len(spans)
            return [self._default_classification(span) for span in spans]

        return classified

    def _try_repair_json(self, response: str, spans: list["NERSpan"]) -> list[ClassifiedSpan] | None:
        """Attempt to repair truncated JSON output from LLM.

        When the LLM runs out of output tokens, the JSON array gets cut mid-entry.
        This method tries to salvage partial results by:
        1. Finding the last complete JSON object (ending with })
        2. Closing the array
        3. Parsing what we recovered

        Returns list of ClassifiedSpan if repair succeeded, None otherwise.
        """
        # Find the JSON array start
        arr_start = response.find('[')
        if arr_start < 0:
            return None

        text = response[arr_start:]

        # Find last complete object: look for last '}' followed by optional comma/whitespace
        last_brace = text.rfind('}')
        if last_brace < 0:
            return None

        # Truncate after last complete object and close array
        candidate = text[:last_brace + 1].rstrip().rstrip(',') + ']'

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, list) or len(data) == 0:
            return None

        logger.info(f"Repaired truncated JSON: recovered {len(data)}/{len(spans)} entries")

        # Parse the recovered entries using same logic as _parse_classifications
        class_map = {item.get('span_index', i): item for i, item in enumerate(data)}

        classified = []
        for i, span in enumerate(spans):
            if i in class_map:
                item = class_map[i]
                cspan = ClassifiedSpan(
                    span=span,
                    is_person=item.get('is_person', True),
                    role=item.get('role', 'mentioned') or 'mentioned',
                    all_roles=item.get('all_roles', ['mentioned']),
                    drop=item.get('drop', False),
                    drop_reason=item.get('reason'),
                    classification_confidence=0.8,
                )
                classified.append(cspan)
            else:
                classified.append(self._default_classification(span))

        return classified

    def _default_classification(self, span: NERSpan) -> ClassifiedSpan:
        """Create default classification for a span."""
        return ClassifiedSpan(
            span=span,
            is_person=True,
            role='mentioned',
            all_roles=['mentioned'],
            drop=False,
            drop_reason=None,
            classification_confidence=0.5
        )

    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get surrounding context for a span."""
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        return text[ctx_start:ctx_end]

    def _detect_document_type(self, text: str) -> str | None:
        """Detect document type from text patterns."""
        text_lower = text.lower()

        if 'from:' in text_lower and 'to:' in text_lower:
            return 'email'
        elif 'flight' in text_lower and ('passenger' in text_lower or 'manifest' in text_lower):
            return 'flight_log'
        elif 'dear ' in text_lower or 'sincerely' in text_lower:
            return 'letter'
        elif 'plaintiff' in text_lower or 'defendant' in text_lower:
            return 'legal'
        else:
            return 'other'

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


class MockLLMClassifier:
    """Mock classifier for testing without GPU."""

    def classify_spans_batch(
        self,
        items: list[BatchClassifyItem],
        inter_batch_size: int = 8,
    ) -> dict[str, ClassificationResult]:
        """Batch classification using mock heuristics."""
        results = {}
        for item in items:
            results[item.source_file] = self.classify_spans(
                document_text=item.document_text,
                spans=item.spans,
                source_file=item.source_file,
            )
        return results

    def classify_spans(self,
                       document_text: str,
                       spans: list[NERSpan],
                       source_file: str = "",
                       **kwargs) -> ClassificationResult:
        """Classify spans using simple heuristics."""
        classified = []

        # Simple patterns for filtering
        org_patterns = ['inc', 'corp', 'llc', 'ltd', 'company']
        location_patterns = ['new york', 'los angeles', 'florida', 'california']

        for span in spans:
            text_lower = span.text.lower()

            # Check if organization
            is_org = any(p in text_lower for p in org_patterns)
            # Check if location
            is_loc = any(p in text_lower for p in location_patterns)

            if is_org:
                classified.append(ClassifiedSpan(
                    span=span,
                    is_person=False,
                    role='',
                    all_roles=[],
                    drop=True,
                    drop_reason='organization',
                    classification_confidence=0.8
                ))
            elif is_loc:
                classified.append(ClassifiedSpan(
                    span=span,
                    is_person=False,
                    role='',
                    all_roles=[],
                    drop=True,
                    drop_reason='location',
                    classification_confidence=0.8
                ))
            else:
                # Default: assume person, role=mentioned
                classified.append(ClassifiedSpan(
                    span=span,
                    is_person=True,
                    role='mentioned',
                    all_roles=['mentioned'],
                    drop=False,
                    drop_reason=None,
                    classification_confidence=0.7
                ))

        person_count = sum(1 for c in classified if c.is_person and not c.drop)
        dropped_count = sum(1 for c in classified if c.drop)

        return ClassificationResult(
            source_file=source_file,
            classified_spans=classified,
            total_spans=len(classified),
            person_spans=person_count,
            dropped_spans=dropped_count,
            document_type='other'
        )

    def cleanup(self):
        pass


def create_classifier(use_llm: bool = True, **kwargs):
    """Factory to create appropriate classifier.

    Args:
        use_llm: Whether to use actual LLM.
        **kwargs: Arguments passed to classifier.

    Returns:
        Classifier instance.
    """
    if use_llm and torch.cuda.is_available():
        return LLMSpanClassifier(**kwargs)
    else:
        logger.warning("Using mock classifier (no GPU or use_llm=False)")
        return MockLLMClassifier()


if __name__ == "__main__":
    from .xlmr_extractor import NERSpan

    sample_spans = [
        NERSpan(text="Jeffrey Epstein", start=10, end=25, entity_type="PER", confidence=0.95),
        NERSpan(text="IBM", start=50, end=53, entity_type="ORG", confidence=0.90),
        NERSpan(text="Ghislaine Maxwell", start=70, end=87, entity_type="PER", confidence=0.92),
        NERSpan(text="New York", start=100, end=108, entity_type="LOC", confidence=0.88),
    ]

    sample_doc = """
    From: Jeffrey Epstein
    To: Ghislaine Maxwell

    The meeting at IBM headquarters in New York went well.
    """

    classifier = create_classifier(use_llm=False)
    result = classifier.classify_spans(sample_doc, sample_spans)

    print(f"Classified {result.total_spans} spans:")
    print(f"  Persons: {result.person_spans}")
    print(f"  Dropped: {result.dropped_spans}")

    for c in result.classified_spans:
        status = "DROP" if c.drop else "KEEP"
        print(f"  [{status}] '{c.span.text}' - is_person={c.is_person}, role={c.role}, reason={c.drop_reason}")
