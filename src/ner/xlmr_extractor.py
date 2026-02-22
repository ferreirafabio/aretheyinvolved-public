"""Stage 1: XLM-R NER extractor for high-recall name span extraction.

Uses xlm-roberta-large-finetuned-conll03-english for identifying PERSON entities.
Configured for high recall (low threshold) to catch as many potential names as possible.
Downstream stages will filter and classify.
"""

from dataclasses import dataclass, field
from typing import Generator

import re

import torch
from loguru import logger

NAME_PARTICLES = frozenset({
    'de', 'del', 'della', 'di', 'da', 'do', 'dos', 'das',
    'la', 'le', 'les', 'el',
    'van', 'von', 'ver', 'vander',
    'den', 'der', 'het', 'ten', 'ter',
    'bin', 'ibn', 'al', 'ul',
    'mac', 'mc',
    'san', 'santa', 'saint',
    'y', 'e', 'i',
})

NAME_SUFFIXES = frozenset({
    'jr', 'jr.', 'sr', 'sr.',
    'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii',
    'ph.d', 'ph.d.', 'md', 'm.d', 'm.d.',
    'esq', 'esq.',
})


@dataclass
class NERSpan:
    """A named entity span from XLM-R NER."""
    text: str               # The extracted text
    start: int              # Character start offset in source text
    end: int                # Character end offset in source text
    entity_type: str        # Entity type (PER, ORG, LOC, MISC)
    confidence: float       # Model confidence score
    page_number: int | None = None  # Page number if available

    def __hash__(self):
        return hash((self.start, self.end, self.text))

    def __eq__(self, other):
        if not isinstance(other, NERSpan):
            return False
        return self.start == other.start and self.end == other.end and self.text == other.text


@dataclass
class NERResult:
    """Result of NER extraction from a document."""
    source_file: str
    spans: list[NERSpan]
    total_spans: int
    person_spans: int  # Count of PER entities
    model_name: str = "xlm-roberta-large-finetuned-conll03-english"


class XLMRNERExtractor:
    """XLM-R NER extractor for high-recall person name extraction.

    Uses xlm-roberta-large-finetuned-conll03-english which is trained on
    CoNLL-2003 for English NER with entity types: PER, ORG, LOC, MISC.

    Configured for high recall - we want to catch all potential names,
    downstream LLM classifier will filter false positives.
    """

    DEFAULT_MODEL = "xlm-roberta-large-finetuned-conll03-english"

    def __init__(self,
                 model_name: str | None = None,
                 device: str | None = None,
                 confidence_threshold: float = 0.3,
                 max_length: int = 512,
                 batch_size: int = 32):
        """Initialize XLM-R NER extractor.

        Args:
            model_name: HuggingFace model name (default: xlm-roberta-large-finetuned-conll03-english).
            device: Device to use ('cuda', 'cpu', or None for auto).
            confidence_threshold: Minimum confidence to include span (low = high recall).
            max_length: Maximum sequence length for model input.
            batch_size: Batch size for chunk processing (default: 8).
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._pipeline = None

    def _load_model(self):
        """Lazy load the NER pipeline."""
        if self._pipeline is not None:
            return

        from transformers import pipeline

        logger.info(f"Loading NER model: {self.model_name}")

        self._pipeline = pipeline(
            "ner",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1,
            aggregation_strategy="simple",  # Merge B-PER, I-PER into single spans
            batch_size=self.batch_size,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        logger.info(f"NER model loaded on {self.device}")
        if torch.cuda.is_available():
            logger.info(f"NER GPU: {torch.cuda.get_device_name(0)} (compute {torch.cuda.get_device_capability(0)})")

    def extract_spans(self,
                      text: str,
                      source_file: str = "",
                      page_boundaries: list[int] | None = None,
                      filter_types: list[str] | None = None) -> NERResult:
        """Extract named entity spans from text.

        Args:
            text: The text to process (should be cleaned for better results).
            source_file: Source file name for tracking.
            page_boundaries: Character offsets where pages start.
            filter_types: Entity types to include (default: ['PER']).

        Returns:
            NERResult with extracted spans.
        """
        if not text or not text.strip():
            return NERResult(
                source_file=source_file,
                spans=[],
                total_spans=0,
                person_spans=0
            )

        self._load_model()

        # Default to person entities only
        if filter_types is None:
            filter_types = ['PER']

        # Process text in chunks to handle long documents
        all_spans = []
        chunk_size = self.max_length * 4  # ~4 chars per token estimate
        overlap = 100  # Overlap to catch entities at boundaries

        # Phase 1: Collect all chunks
        chunks = []  # list of (chunk_start, chunk_text)
        for chunk_start in range(0, len(text), chunk_size - overlap):
            chunk_end = min(chunk_start + chunk_size, len(text))
            chunk_text = text[chunk_start:chunk_end]

            if chunk_text.strip():
                chunks.append((chunk_start, chunk_text))

        if not chunks:
            return NERResult(
                source_file=source_file,
                spans=[],
                total_spans=0,
                person_spans=0
            )

        # Phase 2: Batch NER inference (with OOM fallback)
        try:
            chunk_texts = [ct for _, ct in chunks]
            batch_results = self._pipeline(chunk_texts)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.batch_size > 1:
                # OOM: clear cache and retry at half batch size
                torch.cuda.empty_cache()
                old_bs = self._pipeline.batch_size
                new_bs = max(1, old_bs // 2)
                logger.warning(f"CUDA OOM with batch_size={old_bs}, retrying at {new_bs}")
                self._pipeline.batch_size = new_bs
                try:
                    batch_results = self._pipeline(chunk_texts)
                except Exception as e2:
                    logger.warning(f"Retry also failed: {e2}, falling back to sequential")
                    batch_results = []
                    for _, ct in chunks:
                        try:
                            batch_results.append(self._pipeline(ct))
                        except Exception as ex:
                            logger.warning(f"Sequential NER also failed for chunk: {ex}")
                            batch_results.append([])
                finally:
                    self._pipeline.batch_size = old_bs  # restore original
            else:
                raise
        except Exception as e:
            logger.warning(f"Batch NER failed: {e}, falling back to sequential")
            batch_results = []
            for _, ct in chunks:
                try:
                    batch_results.append(self._pipeline(ct))
                except Exception as ex:
                    logger.warning(f"Sequential NER also failed for chunk: {ex}")
                    batch_results.append([])

        # Phase 3: Convert to NERSpan objects
        for (chunk_start, _), entities in zip(chunks, batch_results):
            for entity in entities:
                # Filter by entity type
                entity_type = entity.get('entity_group', entity.get('entity', ''))
                if entity_type not in filter_types:
                    continue

                # Filter by confidence
                score = entity.get('score', 0)
                if score < self.confidence_threshold:
                    continue

                # Calculate global offsets
                local_start = entity.get('start', 0)
                local_end = entity.get('end', 0)
                global_start = chunk_start + local_start
                global_end = chunk_start + local_end

                # Get span text and verify
                span_text = entity.get('word', text[global_start:global_end])

                # Clean up tokenizer artifacts (leading/trailing spaces, ## tokens)
                span_text = span_text.strip()
                if span_text.startswith('##'):
                    span_text = span_text[2:]

                # Skip empty or very short spans
                if len(span_text) < 2:
                    continue

                # Calculate page number if boundaries provided
                page_num = None
                if page_boundaries:
                    for i, boundary in enumerate(page_boundaries):
                        if i + 1 < len(page_boundaries):
                            if global_start >= boundary and global_start < page_boundaries[i + 1]:
                                page_num = i + 1
                                break
                        else:
                            page_num = i + 1

                span = NERSpan(
                    text=span_text,
                    start=global_start,
                    end=global_end,
                    entity_type=entity_type,
                    confidence=score,
                    page_number=page_num
                )
                all_spans.append(span)

        # Deduplicate overlapping spans (keep highest confidence)
        all_spans = self._deduplicate_spans(all_spans)

        # Split spans that contain multiple concatenated names
        all_spans = self._split_multiname_spans(all_spans, text)

        # Count person entities
        person_count = sum(1 for s in all_spans if s.entity_type == 'PER')

        return NERResult(
            source_file=source_file,
            spans=all_spans,
            total_spans=len(all_spans),
            person_spans=person_count,
            model_name=self.model_name
        )

    def _deduplicate_spans(self, spans: list[NERSpan]) -> list[NERSpan]:
        """Remove duplicate and overlapping spans, keeping highest confidence."""
        if not spans:
            return []

        # Sort by start position, then by confidence (descending)
        sorted_spans = sorted(spans, key=lambda s: (s.start, -s.confidence))

        # Remove exact duplicates first
        seen = set()
        unique_spans = []
        for span in sorted_spans:
            key = (span.start, span.end, span.text)
            if key not in seen:
                seen.add(key)
                unique_spans.append(span)

        # Remove overlapping spans (keep the one with higher confidence)
        result = []
        for span in unique_spans:
            # Check if this span overlaps with any span we're keeping
            overlaps = False
            for kept in result:
                # Check for overlap
                if span.start < kept.end and span.end > kept.start:
                    # Overlap exists - keep the one with higher confidence
                    if span.confidence > kept.confidence:
                        result.remove(kept)
                        result.append(span)
                    overlaps = True
                    break

            if not overlaps:
                result.append(span)

        return sorted(result, key=lambda s: s.start)

    # Patterns for structured list detection in mega-spans
    _NUMBERED_ITEM_RE = re.compile(r'(?:^|\n)\s*(?:\d{1,3}[.)]\s*|[-•]\s*)')
    _NAME_LIKE_RE = re.compile(r'^[A-Z][a-zA-Z\'.\\-]+(?:\s+[A-Za-z\'.\\-]+){0,4}$')

    def _split_multiname_spans(self, spans: list[NERSpan], source_text: str) -> list[NERSpan]:
        """Split spans that contain multiple concatenated names.

        NER models sometimes merge adjacent names into a single span,
        e.g. "Jeffrey Epstein Ghislaine Maxwell" as one PER entity.
        This is especially common in list-formatted documents (personnel lists,
        manifests, contact lists).

        Split strategies (in order):
        1. Structured separators: newlines, semicolons, numbered lists, tabs
        2. Capitalized-word boundary heuristic (original logic)

        Args:
            spans: List of NERSpan objects (already deduplicated).
            source_text: The full source text for offset verification.

        Returns:
            List of NERSpan objects with multi-name spans split.
        """
        result = []

        for span in spans:
            if span.entity_type != 'PER':
                result.append(span)
                continue

            # Try structured splitting first (newlines, semicolons, numbered items)
            structured = self._try_structured_split(span, source_text)
            if structured is not None:
                result.extend(structured)
                continue

            # Fall back to capitalized-word boundary heuristic
            cap_split = self._try_capitalized_split(span, source_text)
            if cap_split is not None:
                result.extend(cap_split)
            else:
                result.append(span)

        return sorted(result, key=lambda s: s.start)

    def _try_structured_split(self, span: NERSpan, source_text: str) -> list[NERSpan] | None:
        """Try to split a span using structured delimiters (newlines, semicolons, numbered items).

        Returns list of sub-spans if split succeeded, None if not applicable.
        """
        text = span.text

        # Detect which delimiter to use
        candidates = None

        # Strategy 1: Newline-separated (most common in lists)
        if '\n' in text:
            candidates = [line.strip() for line in text.split('\n')]
        # Strategy 2: Semicolon-separated
        elif ';' in text and text.count(';') >= 1:
            candidates = [part.strip() for part in text.split(';')]
        # Strategy 3: Tab-separated
        elif '\t' in text:
            candidates = [part.strip() for part in text.split('\t')]
        # Strategy 4: Comma-separated (only if 3+ commas — avoid "Last, First" splits)
        elif text.count(',') >= 3:
            candidates = [part.strip() for part in text.split(',')]

        if candidates is None:
            return None

        # Strip numbered prefixes (1., 2., - , •)
        cleaned = []
        for c in candidates:
            c = re.sub(r'^\s*(?:\d{1,3}[.)]\s*|[-•]\s*)', '', c).strip()
            if c:
                cleaned.append(c)

        # Need at least 2 name-like candidates to justify splitting
        name_like = [c for c in cleaned if self._is_name_like(c)]
        if len(name_like) < 2:
            return None

        # Build sub-spans with verified offsets
        return self._build_sub_spans(span, name_like, source_text)

    def _try_capitalized_split(self, span: NERSpan, source_text: str) -> list[NERSpan] | None:
        """Split using capitalized-word boundary heuristic (original algorithm).

        Returns list of sub-spans if split succeeded, None otherwise.
        """
        cap_word_re = re.compile(r'[A-Z][a-zA-Z\'.\-]*')

        # Find all capitalized-word tokens in the span text
        tokens = list(cap_word_re.finditer(span.text))
        if len(tokens) < 4:
            return None

        # Build groups: accumulate tokens, split when a new capitalized
        # word starts after we already have 2+ tokens in current group
        groups = []       # list of (start_in_span, end_in_span) tuples
        group_start = tokens[0].start()
        group_count = 1

        for idx in range(1, len(tokens)):
            tok_text = tokens[idx].group()
            tok_lower = tok_text.lower().rstrip('.')

            # Name particles and suffixes do NOT trigger a split
            if tok_lower in NAME_PARTICLES or tok_lower in NAME_SUFFIXES:
                group_count += 1
                continue

            if group_count >= 2:
                group_end = tokens[idx - 1].end()
                groups.append((group_start, group_end))
                group_start = tokens[idx].start()
                group_count = 1
            else:
                group_count += 1

        # Final group
        groups.append((group_start, tokens[-1].end()))

        if len(groups) <= 1:
            return None

        # Create sub-spans and verify offsets
        sub_spans = []
        for (gs, ge) in groups:
            sub_text = span.text[gs:ge].strip()
            if len(sub_text) < 4:
                continue

            sub_start = span.start + gs
            sub_end = span.start + ge

            if sub_start < 0 or sub_end > len(source_text):
                return None
            if source_text[sub_start:sub_end] != span.text[gs:ge]:
                return None

            sub_spans.append(NERSpan(
                text=sub_text,
                start=sub_start,
                end=sub_end,
                entity_type=span.entity_type,
                confidence=span.confidence,
                page_number=span.page_number,
            ))

        if len(sub_spans) <= 1:
            return None
        return sub_spans

    @staticmethod
    def _is_name_like(text: str) -> bool:
        """Check if text looks like a person name (2-6 words, starts with capital)."""
        text = text.strip()
        if not text or len(text) < 2:
            return False
        words = text.split()
        if len(words) < 1 or len(words) > 6:
            return False
        # First word must start with uppercase letter
        if not words[0][0].isupper():
            return False
        # At least one word must be 2+ chars
        if not any(len(w) >= 2 for w in words):
            return False
        # Reject if mostly digits
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars < len(text) * 0.5:
            return False
        return True

    def _build_sub_spans(self, parent: NERSpan, name_texts: list[str],
                         source_text: str) -> list[NERSpan] | None:
        """Build verified sub-spans from extracted name texts.

        Finds each name text within the parent span's range in source_text,
        verifying exact offset matches.

        Returns list of sub-spans, or None if verification fails.
        """
        sub_spans = []
        search_start = parent.start

        for name_text in name_texts:
            # Find this name in the source text within the parent span's range
            idx = source_text.find(name_text, search_start, parent.end + len(name_text))
            if idx == -1:
                # Try case-insensitive fallback — find the original casing in source
                # by scanning for the text
                continue

            sub_spans.append(NERSpan(
                text=name_text,
                start=idx,
                end=idx + len(name_text),
                entity_type=parent.entity_type,
                confidence=parent.confidence,
                page_number=parent.page_number,
            ))
            search_start = idx + len(name_text)

        if len(sub_spans) < 2:
            # Splitting didn't produce enough results — keep parent
            return None

        return sub_spans

    def extract_streaming(self,
                          texts: list[str],
                          source_files: list[str] | None = None,
                          page_boundaries_list: list[list[int]] | None = None
                          ) -> Generator[NERResult, None, None]:
        """Stream NER extraction from multiple documents.

        Args:
            texts: List of document texts.
            source_files: Optional list of source file names.
            page_boundaries_list: Optional list of page boundaries per document.

        Yields:
            NERResult for each document.
        """
        if source_files is None:
            source_files = [f"doc_{i}" for i in range(len(texts))]

        if page_boundaries_list is None:
            page_boundaries_list = [None] * len(texts)

        for text, source_file, boundaries in zip(texts, source_files, page_boundaries_list):
            yield self.extract_spans(
                text,
                source_file=source_file,
                page_boundaries=boundaries
            )

    def cleanup(self):
        """Free GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("NER model unloaded")


class MockXLMRExtractor:
    """Mock NER extractor for testing without GPU."""

    def extract_spans(self,
                      text: str,
                      source_file: str = "",
                      page_boundaries: list[int] | None = None,
                      filter_types: list[str] | None = None) -> NERResult:
        """Extract names using simple regex (for testing)."""
        import re

        # Simple capitalized name pattern
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        spans = []

        for match in re.finditer(pattern, text):
            # Skip very short matches
            if len(match.group()) < 3:
                continue

            span = NERSpan(
                text=match.group(),
                start=match.start(),
                end=match.end(),
                entity_type='PER',
                confidence=0.7
            )
            spans.append(span)

        return NERResult(
            source_file=source_file,
            spans=spans,
            total_spans=len(spans),
            person_spans=len(spans),
            model_name="mock-ner"
        )

    def cleanup(self):
        pass


def create_ner_extractor(use_gpu: bool = True, **kwargs):
    """Factory to create appropriate NER extractor.

    Args:
        use_gpu: Whether to use GPU-based model.
        **kwargs: Arguments passed to extractor.

    Returns:
        NER extractor instance.
    """
    if use_gpu and torch.cuda.is_available():
        return XLMRNERExtractor(**kwargs)
    else:
        logger.warning("Using mock NER extractor (no GPU or use_gpu=False)")
        return MockXLMRExtractor()


if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    From: Jeffrey Epstein <jepstein@mail.com>
    To: Ghislaine Maxwell <gmax@mail.com>
    Cc: Sarah Kellen

    Hi Ghislaine,

    Please confirm with Bill Clinton and Donald Trump about the
    meeting next week. Also, Alan Dershowitz called about the
    legal matter.

    Regards,
    Jeffrey
    """

    extractor = create_ner_extractor(use_gpu=False)

    print("=== NER Test ===")
    result = extractor.extract_spans(sample_text, source_file="email_test.txt")
    print(f"Found {result.total_spans} spans ({result.person_spans} persons):")
    for span in result.spans:
        print(f"  - '{span.text}' ({span.entity_type}) @ {span.start}-{span.end}, conf={span.confidence:.2f}")
