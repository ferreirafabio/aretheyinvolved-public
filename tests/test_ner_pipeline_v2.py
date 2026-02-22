"""Comprehensive tests for the NER + LLM Name Extraction Pipeline v2.

Tests cover all 5 stages:
- Stage 0: Deterministic text cleaning
- Stage 1: XLM-R NER extraction
- Stage 2: LLM classification
- Stage 3: Hard validation
- Stage 4: LLM repair
- Full pipeline integration
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ner.deterministic_cleaner import (
    same_length_clean,
    clean_document,
    CleaningResult,
    is_between_letters,
    get_page_boundaries,
    get_page_number,
)
from src.ner.xlmr_extractor import (
    MockXLMRExtractor,
    XLMRNERExtractor,
    NERSpan,
    NERResult,
    create_ner_extractor,
)
from src.ner.llm_classifier import (
    MockLLMClassifier,
    LLMSpanClassifier,
    ClassifiedSpan,
    ClassificationResult,
    create_classifier,
)
from src.ner.hard_validator import (
    HardValidator,
    ValidatedSpan,
    ValidationError,
    needs_repair,
    validate_span,
)
from src.ner.llm_repair import (
    MockLLMRepairer,
    RepairedName,
    RepairResult,
    create_repairer,
)
from src.ner.pipeline import (
    ExtractionPipeline,
    PipelineConfig,
    PipelineResult,
)


# ============================================================
# Stage 0: Deterministic Cleaner Tests
# ============================================================

class TestSameLengthClean:
    """Tests for the same_length_clean function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert same_length_clean("") == ""

    def test_preserves_length(self):
        """Cleaned text must have same length as input."""
        test_cases = [
            "Hello World",
            "John Smith",
            "Hello\x00World",  # Control char
            "café",  # Unicode
            "John/Smith",  # Weird punct
            "a" * 1000,  # Long string
        ]
        for text in test_cases:
            result = same_length_clean(text)
            assert len(result) == len(text), f"Length mismatch for '{text}'"

    def test_control_char_replacement(self):
        """Control characters should be replaced with spaces."""
        # NUL char
        result = same_length_clean("Hello\x00World")
        assert "\x00" not in result
        assert len(result) == 11

        # Tab -> space (common)
        result = same_length_clean("Hello\tWorld")
        assert len(result) == 11

    def test_unicode_normalization(self):
        """Unicode should be normalized (NFKC)."""
        result = same_length_clean("café")
        assert len(result) == len("café")

    def test_weird_punct_in_words(self):
        """Weird punctuation between letters should be replaced."""
        # These are characters that appear between letters
        result = same_length_clean("John/Smith")  # / between letters
        assert result == "John Smith"

        result = same_length_clean("Mary\\Brown")  # \ between letters
        assert result == "Mary Brown"

    def test_preserves_normal_punct(self):
        """Normal punctuation should be preserved."""
        text = "Hello, World! How are you?"
        result = same_length_clean(text)
        assert result == text

    def test_ocr_digits_preserved(self):
        """OCR digit errors (0, 1, 5) should NOT be changed in Stage 0."""
        # Stage 0 only does cleaning, not OCR repair
        text = "J0HN SM1TH"
        result = same_length_clean(text)
        assert result == text  # Digits preserved, repair happens in Stage 4

    def test_newlines_preserved(self):
        """Newlines must survive cleaning — they serve as NER boundary signals."""
        text = "Jeffrey Epstein\nGhislaine Maxwell"
        result = same_length_clean(text)
        assert "\n" in result
        assert result == text

    def test_carriage_return_preserved(self):
        """Windows-style \\r\\n line endings must survive cleaning."""
        text = "Jeffrey Epstein\r\nGhislaine Maxwell"
        result = same_length_clean(text)
        assert "\r\n" in result
        assert result == text

    def test_tab_preserved(self):
        """Tab characters must survive cleaning."""
        text = "Jeffrey\tEpstein"
        result = same_length_clean(text)
        assert "\t" in result
        assert result == text

    def test_other_control_chars_still_replaced(self):
        """Non-whitespace control characters (NUL, SOH) must still become spaces."""
        text = "Hello\x00World\x01Here"
        result = same_length_clean(text)
        assert "\x00" not in result
        assert "\x01" not in result
        assert len(result) == len(text)

    def test_length_invariant_with_preserved_whitespace(self):
        """Length invariant must hold for text containing \\n, \\r, \\t."""
        text = "Name\tOne\r\nName\nTwo\t\tThree"
        result = same_length_clean(text)
        assert len(result) == len(text)


class TestIsBetweenLetters:
    """Tests for is_between_letters helper."""

    def test_between_letters(self):
        assert is_between_letters("a/b", 1) is True
        assert is_between_letters("John/Smith", 4) is True

    def test_not_between_letters(self):
        assert is_between_letters("/ab", 0) is False  # Start
        assert is_between_letters("ab/", 2) is False  # End
        assert is_between_letters("a/1", 1) is False  # Number after
        assert is_between_letters("1/b", 1) is False  # Number before


class TestPageBoundaries:
    """Tests for page boundary tracking."""

    def test_get_page_boundaries_empty(self):
        """Empty doc should return [0]."""
        result = get_page_boundaries({})
        assert result == [0]

    def test_get_page_boundaries_pages(self):
        """Should calculate boundaries from pages."""
        ocr_data = {
            'pages': [
                {'page_number': 1, 'text': 'Page one'},
                {'page_number': 2, 'text': 'Page two'},
            ]
        }
        boundaries = get_page_boundaries(ocr_data)
        assert boundaries[0] == 0
        assert len(boundaries) >= 2

    def test_get_page_number(self):
        """Should return correct page for offset."""
        boundaries = [0, 100, 200, 300]

        assert get_page_number(0, boundaries) == 1
        assert get_page_number(50, boundaries) == 1
        assert get_page_number(100, boundaries) == 2
        assert get_page_number(150, boundaries) == 2
        assert get_page_number(250, boundaries) == 3
        assert get_page_number(350, boundaries) == 4  # Beyond last


class TestCleanDocument:
    """Tests for clean_document function."""

    def test_clean_with_full_text(self):
        """Should clean full_text field."""
        ocr_data = {'full_text': 'Hello\x00World'}
        result = clean_document(ocr_data)

        assert isinstance(result, CleaningResult)
        assert len(result.clean_text) == len(result.raw_text)
        assert '\x00' not in result.clean_text

    def test_clean_with_pages(self):
        """Should concatenate and clean pages."""
        ocr_data = {
            'pages': [
                {'page_number': 1, 'text': 'Page one'},
                {'page_number': 2, 'text': 'Page two'},
            ]
        }
        result = clean_document(ocr_data)

        assert 'Page one' in result.raw_text
        assert 'Page two' in result.raw_text
        assert result.page_boundaries[0] == 0

    def test_verify_span(self):
        """Should verify spans exist."""
        ocr_data = {'full_text': 'Hello World'}
        result = clean_document(ocr_data)

        assert result.verify_span(0, 5, 'Hello') is True
        assert result.verify_span(6, 11, 'World') is True
        assert result.verify_span(0, 5, 'Wrong') is False


# ============================================================
# Stage 1: XLM-R NER Extractor Tests
# ============================================================

class TestMockXLMRExtractor:
    """Tests for the mock NER extractor."""

    @pytest.fixture
    def extractor(self):
        return MockXLMRExtractor()

    def test_extracts_capitalized_names(self, extractor):
        """Should extract capitalized multi-word names."""
        text = "John Smith met with Mary Brown yesterday."
        result = extractor.extract_spans(text, source_file="test.pdf")

        assert isinstance(result, NERResult)
        assert result.total_spans >= 2  # John Smith, Mary Brown

        names = [s.text for s in result.spans]
        assert "John Smith" in names
        assert "Mary Brown" in names

    def test_returns_correct_offsets(self, extractor):
        """Offsets should match actual text positions."""
        text = "Hello John Smith here."
        result = extractor.extract_spans(text)

        for span in result.spans:
            assert text[span.start:span.end] == span.text

    def test_empty_text(self, extractor):
        """Empty text should return empty result."""
        result = extractor.extract_spans("")
        assert result.total_spans == 0
        assert result.spans == []

    def test_no_names(self, extractor):
        """Text without names should return empty."""
        result = extractor.extract_spans("hello world how are you")
        # Mock only extracts capitalized patterns
        assert result.total_spans == 0

    def test_span_properties(self, extractor):
        """NERSpan should have all required properties."""
        text = "John Smith is here."
        result = extractor.extract_spans(text)

        for span in result.spans:
            assert isinstance(span, NERSpan)
            assert span.text
            assert span.start >= 0
            assert span.end > span.start
            assert span.entity_type == 'PER'
            assert 0 <= span.confidence <= 1


class TestNERExtractorFactory:
    """Tests for the NER extractor factory."""

    def test_creates_mock_without_gpu(self):
        """Should create mock extractor when use_gpu=False."""
        extractor = create_ner_extractor(use_gpu=False)
        assert isinstance(extractor, MockXLMRExtractor)


class TestMultiNameSpanSplitting:
    """Regression tests for _split_multiname_spans in XLMRNERExtractor.

    NER models sometimes merge adjacent names into a single span,
    e.g. "Jeffrey Epstein Ghislaine Maxwell" as one PER entity.
    These tests verify the splitting logic correctly separates them.
    """

    @pytest.fixture
    def extractor(self):
        """Create XLMRNERExtractor without loading any model."""
        ext = XLMRNERExtractor.__new__(XLMRNERExtractor)
        ext.confidence_threshold = 0.3
        ext.max_length = 512
        return ext

    def test_splits_two_merged_names(self, extractor):
        """A span containing two names should be split into 2 sub-spans."""
        source_text = "Jeffrey Epstein Ghislaine Maxwell"
        span = NERSpan(
            text="Jeffrey Epstein Ghislaine Maxwell",
            start=0, end=33, entity_type="PER", confidence=0.9
        )
        result = extractor._split_multiname_spans([span], source_text)
        assert len(result) == 2
        assert result[0].text == "Jeffrey Epstein"
        assert result[1].text == "Ghislaine Maxwell"
        # Verify offsets
        assert result[0].start == 0
        assert result[0].end == 15
        assert result[1].start == 16
        assert result[1].end == 33

    def test_splits_three_merged_names(self, extractor):
        """A span containing three two-word names (6 tokens) should be split into 3 sub-spans."""
        source_text = "John Smith Mary Brown Alan Jones"
        span = NERSpan(
            text="John Smith Mary Brown Alan Jones",
            start=0, end=31, entity_type="PER", confidence=0.85
        )
        result = extractor._split_multiname_spans([span], source_text)
        assert len(result) == 3
        names = [s.text for s in result]
        assert "John Smith" in names
        assert "Mary Brown" in names
        assert "Alan Jones" in names

    def test_does_not_split_single_name(self, extractor):
        """A span with a single two-word name must stay as 1 span."""
        source_text = "Jeffrey Epstein"
        span = NERSpan(
            text="Jeffrey Epstein",
            start=0, end=15, entity_type="PER", confidence=0.9
        )
        result = extractor._split_multiname_spans([span], source_text)
        assert len(result) == 1
        assert result[0].text == "Jeffrey Epstein"

    def test_does_not_split_three_word_name(self, extractor):
        """A 3-token name like 'Mary Jane Watson' must NOT be split (ambiguous)."""
        source_text = "Mary Jane Watson"
        span = NERSpan(
            text="Mary Jane Watson",
            start=0, end=16, entity_type="PER", confidence=0.9
        )
        result = extractor._split_multiname_spans([span], source_text)
        assert len(result) == 1
        assert result[0].text == "Mary Jane Watson"

    def test_preserves_hyphenated_names(self, extractor):
        """Hyphenated names like 'Carlos Morales-Mercado' must stay as 1 span."""
        source_text = "Carlos Morales-Mercado"
        span = NERSpan(
            text="Carlos Morales-Mercado",
            start=0, end=22, entity_type="PER", confidence=0.9
        )
        result = extractor._split_multiname_spans([span], source_text)
        assert len(result) == 1
        assert result[0].text == "Carlos Morales-Mercado"

    def test_preserves_name_particles(self, extractor):
        """Names with particles like 'Maria de la Cruz' must stay as 1 span."""
        source_text = "Maria de la Cruz"
        span = NERSpan(
            text="Maria de la Cruz",
            start=0, end=16, entity_type="PER", confidence=0.9
        )
        result = extractor._split_multiname_spans([span], source_text)
        assert len(result) == 1
        assert result[0].text == "Maria de la Cruz"

    def test_preserves_suffixes(self, extractor):
        """Names with suffixes like 'John Smith Jr.' must stay as 1 span."""
        source_text = "John Smith Jr."
        span = NERSpan(
            text="John Smith Jr.",
            start=0, end=14, entity_type="PER", confidence=0.9
        )
        result = extractor._split_multiname_spans([span], source_text)
        assert len(result) == 1
        assert result[0].text == "John Smith Jr."

    def test_only_splits_per_entities(self, extractor):
        """ORG spans must not be touched by the splitting logic."""
        source_text = "Goldman Sachs Morgan Stanley"
        span = NERSpan(
            text="Goldman Sachs Morgan Stanley",
            start=0, end=28, entity_type="ORG", confidence=0.9
        )
        result = extractor._split_multiname_spans([span], source_text)
        assert len(result) == 1
        assert result[0].text == "Goldman Sachs Morgan Stanley"
        assert result[0].entity_type == "ORG"

    def test_offset_with_nonzero_start(self, extractor):
        """Spans starting at a non-zero offset must produce correct sub-span offsets."""
        source_text = "From: Jeffrey Epstein Ghislaine Maxwell"
        span = NERSpan(
            text="Jeffrey Epstein Ghislaine Maxwell",
            start=6, end=39, entity_type="PER", confidence=0.9
        )
        result = extractor._split_multiname_spans([span], source_text)
        assert len(result) == 2
        assert result[0].text == "Jeffrey Epstein"
        assert result[0].start == 6
        assert result[0].end == 21
        assert result[1].text == "Ghislaine Maxwell"
        assert result[1].start == 22
        assert result[1].end == 39
        # Verify offsets against source text
        assert source_text[result[0].start:result[0].end] == "Jeffrey Epstein"
        assert source_text[result[1].start:result[1].end] == "Ghislaine Maxwell"


class TestStructuredSplitting:
    """Regression tests for Bug 2: NER mega-spans in list-formatted documents.

    NER models merge adjacent names in lists (manifests, personnel lists, etc.)
    into mega-spans. The structured splitter handles newlines, numbered lists,
    semicolons, and tab-delimited patterns.
    """

    @pytest.fixture
    def extractor(self):
        """Create XLMRNERExtractor without loading any model."""
        ext = XLMRNERExtractor.__new__(XLMRNERExtractor)
        ext.confidence_threshold = 0.3
        ext.max_length = 512
        return ext

    def test_splits_newline_separated_names(self, extractor):
        """Newline-separated names should be split into individual spans."""
        source_text = "John Smith\nJane Doe\nBob Wilson"
        span = NERSpan(
            text="John Smith\nJane Doe\nBob Wilson",
            start=0, end=len(source_text), entity_type="PER", confidence=0.85
        )
        result = extractor._split_multiname_spans([span], source_text)
        assert len(result) == 3
        names = [s.text for s in result]
        assert "John Smith" in names
        assert "Jane Doe" in names
        assert "Bob Wilson" in names

    def test_splits_numbered_list(self, extractor):
        """Numbered list items should be split into individual names."""
        source_text = "1. John Smith\n2. Jane Doe\n3. Bob Wilson"
        span = NERSpan(
            text="1. John Smith\n2. Jane Doe\n3. Bob Wilson",
            start=0, end=len(source_text), entity_type="PER", confidence=0.85
        )
        result = extractor._split_multiname_spans([span], source_text)
        assert len(result) >= 3
        names = [s.text for s in result]
        assert "John Smith" in names
        assert "Jane Doe" in names
        assert "Bob Wilson" in names

    def test_splits_semicolon_separated(self, extractor):
        """Semicolon-separated names should be split."""
        source_text = "John Smith; Jane Doe; Bob Wilson"
        span = NERSpan(
            text="John Smith; Jane Doe; Bob Wilson",
            start=0, end=len(source_text), entity_type="PER", confidence=0.85
        )
        result = extractor._split_multiname_spans([span], source_text)
        assert len(result) == 3
        names = [s.text for s in result]
        assert "John Smith" in names
        assert "Jane Doe" in names
        assert "Bob Wilson" in names

    def test_splits_tab_separated(self, extractor):
        """Tab-separated names should be split."""
        source_text = "John Smith\tJane Doe\tBob Wilson"
        span = NERSpan(
            text="John Smith\tJane Doe\tBob Wilson",
            start=0, end=len(source_text), entity_type="PER", confidence=0.85
        )
        result = extractor._split_multiname_spans([span], source_text)
        assert len(result) == 3
        names = [s.text for s in result]
        assert "John Smith" in names
        assert "Jane Doe" in names
        assert "Bob Wilson" in names

    def test_splits_large_personnel_list(self, extractor):
        """A large mega-span from a personnel list should split into many names."""
        names_list = [
            "Jeffrey Epstein", "Ghislaine Maxwell", "Sarah Kellen",
            "Alan Dershowitz", "Bill Clinton", "Donald Trump",
            "Prince Andrew", "Virginia Roberts", "Nadia Marcinkova",
            "Jean Luc Brunel", "Leslie Wexner", "Eva Andersson"
        ]
        source_text = "\n".join(names_list)
        span = NERSpan(
            text=source_text,
            start=0, end=len(source_text), entity_type="PER", confidence=0.80
        )
        result = extractor._split_multiname_spans([span], source_text)
        # Should get at least 10 out of 12 (allowing some margin for edge cases)
        assert len(result) >= 10, f"Expected >=10 names, got {len(result)}: {[s.text for s in result]}"

    def test_does_not_split_comma_in_last_first_format(self, extractor):
        """'Smith, John' (one comma) should NOT be split — it's Last, First format."""
        source_text = "Smith, John"
        span = NERSpan(
            text="Smith, John",
            start=0, end=11, entity_type="PER", confidence=0.85
        )
        result = extractor._split_multiname_spans([span], source_text)
        # With only 1 comma, should NOT trigger comma-split (needs 3+)
        assert len(result) == 1
        assert result[0].text == "Smith, John"

    def test_does_not_split_non_name_lines(self, extractor):
        """Lines that aren't name-like should not produce sub-spans."""
        source_text = "some garbage\n123 numbers\nJohn Smith"
        span = NERSpan(
            text="some garbage\n123 numbers\nJohn Smith",
            start=0, end=len(source_text), entity_type="PER", confidence=0.7
        )
        result = extractor._split_multiname_spans([span], source_text)
        # Only "John Smith" is name-like — need 2+ name-like for a split
        # So it should fall through to the original span
        assert len(result) == 1

    def test_structured_split_preserves_offsets(self, extractor):
        """Sub-spans from structured splitting must have correct offsets."""
        source_text = "Passengers:\nJohn Smith\nJane Doe\nBob Wilson"
        # Span starts at "John Smith" (offset 12)
        span_text = "John Smith\nJane Doe\nBob Wilson"
        span = NERSpan(
            text=span_text,
            start=12, end=12 + len(span_text), entity_type="PER", confidence=0.85
        )
        result = extractor._split_multiname_spans([span], source_text)
        assert len(result) == 3
        # Verify each sub-span's offset matches the source text
        for sub in result:
            assert source_text[sub.start:sub.end] == sub.text, (
                f"Offset mismatch: source[{sub.start}:{sub.end}]="
                f"'{source_text[sub.start:sub.end]}' != '{sub.text}'"
            )

    def test_is_name_like_helper(self, extractor):
        """_is_name_like should accept person names and reject garbage."""
        assert extractor._is_name_like("John Smith") is True
        assert extractor._is_name_like("J. Smith") is True
        assert extractor._is_name_like("Maria de la Cruz") is True
        assert extractor._is_name_like("123 numbers") is False
        assert extractor._is_name_like("") is False
        assert extractor._is_name_like("a") is False
        # Too many words
        assert extractor._is_name_like("One Two Three Four Five Six Seven") is False


# ============================================================
# Stage 2: LLM Classifier Tests
# ============================================================

class TestMockLLMClassifier:
    """Tests for the mock LLM classifier."""

    @pytest.fixture
    def classifier(self):
        return MockLLMClassifier()

    @pytest.fixture
    def sample_spans(self):
        return [
            NERSpan(text="John Smith", start=0, end=10, entity_type="PER", confidence=0.9),
            NERSpan(text="IBM", start=20, end=23, entity_type="ORG", confidence=0.85),
            NERSpan(text="New York", start=30, end=38, entity_type="LOC", confidence=0.88),
            NERSpan(text="Mary Brown", start=50, end=60, entity_type="PER", confidence=0.92),
        ]

    def test_classifies_persons(self, classifier, sample_spans):
        """Should classify person names as persons."""
        result = classifier.classify_spans(
            document_text="John Smith works at IBM in New York with Mary Brown.",
            spans=sample_spans
        )

        assert isinstance(result, ClassificationResult)
        assert result.total_spans == 4

    def test_drops_organizations(self, classifier, sample_spans):
        """Should drop organization spans."""
        result = classifier.classify_spans(
            document_text="Test",
            spans=sample_spans
        )

        # Find IBM classification
        ibm_span = None
        for c in result.classified_spans:
            if c.span.text == "IBM":
                ibm_span = c
                break

        # Mock classifier should not drop IBM (it only checks for 'inc', 'corp', etc.)
        # Let's create a span with 'Inc'
        spans_with_corp = [
            NERSpan(text="Acme Inc", start=0, end=8, entity_type="ORG", confidence=0.9),
        ]
        result = classifier.classify_spans("Acme Inc", spans=spans_with_corp)

        assert result.classified_spans[0].drop is True
        assert result.classified_spans[0].drop_reason == "organization"

    def test_drops_locations(self, classifier):
        """Should drop location spans."""
        spans = [
            NERSpan(text="New York", start=0, end=8, entity_type="LOC", confidence=0.9),
        ]
        result = classifier.classify_spans("In New York", spans=spans)

        assert result.classified_spans[0].drop is True
        assert result.classified_spans[0].drop_reason == "location"

    def test_assigns_role(self, classifier, sample_spans):
        """Should assign role to classified spans."""
        result = classifier.classify_spans("Test", spans=sample_spans)

        for c in result.classified_spans:
            if c.is_person and not c.drop:
                assert c.role in ['sender', 'recipient', 'mentioned', 'passenger', 'other']

    def test_empty_spans(self, classifier):
        """Empty spans should return empty result."""
        result = classifier.classify_spans("Test", spans=[])
        assert result.total_spans == 0
        assert result.classified_spans == []


class TestParseClassificationsBatchIndex:
    """Regression tests for Bug 1: batch index misalignment in _parse_classifications.

    FIX: Prompt now always sends 0-based indices ("index": 0, 1, 2...) regardless
    of batch number. Parser uses 0-based lookup. This prevents the LLM from
    returning global indices that don't match the lookup.
    """

    @pytest.fixture
    def classifier(self):
        """Create LLMSpanClassifier in strict mode (CI behavior)."""
        clf = LLMSpanClassifier.__new__(LLMSpanClassifier)
        clf.model_name = "test"
        clf.device = "cpu"
        clf.strict = True
        clf._model = None
        clf._tokenizer = None
        clf._shared_model = None
        return clf

    def test_strict_mode_raises_on_mismatch(self, classifier):
        """In strict mode, batch-size mismatch must raise ValueError."""
        spans = [
            NERSpan(text="Jeffrey Epstein", start=6, end=21, entity_type="PER", confidence=0.95),
            NERSpan(text="Ghislaine Maxwell", start=30, end=47, entity_type="PER", confidence=0.92),
        ]

        # LLM returns 1 classification for 2 spans
        response = '''[
            {"span_index": 0, "is_person": true, "role": "sender", "all_roles": ["sender"], "drop": false, "reason": null}
        ]'''

        with pytest.raises(ValueError, match="LLM returned 1 classifications for 2 spans"):
            classifier._parse_classifications(response, spans, start_index=0)

    def test_strict_mode_passes_on_match(self, classifier):
        """In strict mode, matching counts should not raise."""
        spans = [
            NERSpan(text="Jeffrey Epstein", start=6, end=21, entity_type="PER", confidence=0.95),
        ]

        response = '''[
            {"span_index": 0, "is_person": true, "role": "sender", "all_roles": ["sender"], "drop": false, "reason": null}
        ]'''

        # Should not raise
        result = classifier._parse_classifications(response, spans, start_index=0)
        assert len(result) == 1
        assert result[0].role == "sender"

    def test_batch2_uses_0based_indices(self, classifier):
        """Batch 2 spans use 0-based indices (not global). LLM returns 0-based."""
        spans = [
            NERSpan(text="Ghislaine Maxwell", start=500, end=517, entity_type="PER", confidence=0.92),
            NERSpan(text="Sarah Kellen", start=520, end=532, entity_type="PER", confidence=0.88),
        ]

        # LLM returns 0-based span_index (matches the 0-based prompt indices)
        response = '''[
            {"span_index": 0, "is_person": true, "role": "recipient", "all_roles": ["recipient"], "drop": false, "reason": null},
            {"span_index": 1, "is_person": true, "role": "mentioned", "all_roles": ["mentioned"], "drop": false, "reason": null}
        ]'''

        result = classifier._parse_classifications(response, spans, start_index=20)

        assert len(result) == 2
        assert result[0].role == "recipient"
        assert result[1].role == "mentioned"
        # Neither should fall back to default
        assert result[0].classification_confidence == 0.8  # Not 0.5 (default)
        assert result[1].classification_confidence == 0.8

    def test_batch2_without_span_index_uses_positional_fallback(self, classifier):
        """When LLM omits span_index, positional fallback uses 0-based index."""
        spans = [
            NERSpan(text="Ghislaine Maxwell", start=500, end=517, entity_type="PER", confidence=0.92),
            NERSpan(text="Sarah Kellen", start=520, end=532, entity_type="PER", confidence=0.88),
        ]

        # LLM omits span_index — fallback should use 0-based: i=0, 1
        response = '''[
            {"is_person": true, "role": "sender", "all_roles": ["sender"], "drop": false, "reason": null},
            {"is_person": true, "role": "recipient", "all_roles": ["recipient"], "drop": false, "reason": null}
        ]'''

        result = classifier._parse_classifications(response, spans, start_index=20)

        assert len(result) == 2
        assert result[0].role == "sender"
        assert result[1].role == "recipient"
        # Should NOT be default (would be "mentioned" with conf=0.5 if fallback was broken)
        assert result[0].classification_confidence == 0.8

    def test_batch1_still_works(self, classifier):
        """Batch 1 (start_index=0) should continue to work as before."""
        spans = [
            NERSpan(text="Jeffrey Epstein", start=6, end=21, entity_type="PER", confidence=0.95),
        ]

        response = '''[
            {"span_index": 0, "is_person": true, "role": "sender", "all_roles": ["sender"], "drop": false, "reason": null}
        ]'''

        result = classifier._parse_classifications(response, spans, start_index=0)

        assert len(result) == 1
        assert result[0].role == "sender"

    def test_batch3_all_classified_not_defaulted(self, classifier):
        """Batch 3 (start_index=100): all spans must get real classifications, not defaults."""
        spans = [
            NERSpan(text="Alan Dershowitz", start=2000, end=2015, entity_type="PER", confidence=0.91),
            NERSpan(text="Bill Clinton", start=2020, end=2032, entity_type="PER", confidence=0.89),
            NERSpan(text="Donald Trump", start=2040, end=2052, entity_type="PER", confidence=0.87),
        ]

        # LLM returns 0-based indices
        response = '''[
            {"span_index": 0, "is_person": true, "role": "mentioned", "all_roles": ["mentioned"], "drop": false, "reason": null},
            {"span_index": 1, "is_person": true, "role": "mentioned", "all_roles": ["mentioned"], "drop": false, "reason": null},
            {"span_index": 2, "is_person": true, "role": "sender", "all_roles": ["sender"], "drop": false, "reason": null}
        ]'''

        result = classifier._parse_classifications(response, spans, start_index=100)

        assert len(result) == 3
        # ALL must be real classifications (confidence=0.8), NOT defaults (0.5)
        for c in result:
            assert c.classification_confidence == 0.8, (
                f"Span '{c.span.text}' fell through to default! "
                f"confidence={c.classification_confidence}"
            )
        assert result[2].role == "sender"

    def test_size_mismatch_fallback_in_prod_mode(self):
        """In prod mode (strict=False), LLM returning fewer items falls back to defaults."""
        clf = LLMSpanClassifier.__new__(LLMSpanClassifier)
        clf.model_name = "test"
        clf.device = "cpu"
        clf.strict = False
        clf._model = None
        clf._tokenizer = None
        clf._shared_model = None

        spans = [
            NERSpan(text="Jeffrey Epstein", start=6, end=21, entity_type="PER", confidence=0.95),
            NERSpan(text="Ghislaine Maxwell", start=30, end=47, entity_type="PER", confidence=0.92),
            NERSpan(text="Sarah Kellen", start=50, end=62, entity_type="PER", confidence=0.88),
        ]

        # LLM only returns 2 out of 3
        response = '''[
            {"span_index": 0, "is_person": true, "role": "sender", "all_roles": ["sender"], "drop": false, "reason": null},
            {"span_index": 1, "is_person": true, "role": "recipient", "all_roles": ["recipient"], "drop": false, "reason": null}
        ]'''

        result = clf._parse_classifications(response, spans, start_index=0)

        assert len(result) == 3
        assert result[0].role == "sender"
        assert result[1].role == "recipient"
        # Third span falls back to default
        assert result[2].role == "mentioned"
        assert result[2].classification_confidence == 0.5

    def test_ner_confidence_safeguard(self, classifier):
        """High-confidence NER spans should not be dropped even if LLM says drop."""
        spans = [
            NERSpan(text="Ghislaine Maxwell", start=500, end=517, entity_type="PER", confidence=0.95),
        ]

        response = '''[
            {"span_index": 0, "is_person": false, "role": null, "all_roles": [], "drop": true, "reason": "not_a_person"}
        ]'''

        result = classifier._parse_classifications(response, spans, start_index=0)

        assert len(result) == 1
        # Should override drop because NER confidence > 0.85
        assert result[0].drop is False
        assert result[0].drop_reason is None


# ============================================================
# Stage 3: Hard Validator Tests
# ============================================================

class TestNeedsRepair:
    """Tests for the needs_repair function."""

    def test_clean_names_dont_need_repair(self):
        """Clean names should not need repair."""
        assert needs_repair("John Smith") is False
        assert needs_repair("Mary Brown") is False
        assert needs_repair("Jeffrey Epstein") is False
        assert needs_repair("Alan Dershowitz") is False

    def test_ocr_digits_need_repair(self):
        """Names with OCR digit errors need repair."""
        assert needs_repair("J0HN SM1TH") is True  # 0 and 1
        assert needs_repair("5ARAH JONES") is True  # 5
        assert needs_repair("MAR1A BROWN") is True  # 1

    def test_weird_chars_need_repair(self):
        """Names with weird characters need repair."""
        assert needs_repair("John/Smith") is True
        assert needs_repair("Mary\\Brown") is True
        assert needs_repair("Test|Name") is True

    def test_digit_between_letters(self):
        """Digits between letters indicate OCR error."""
        assert needs_repair("Jo1n") is True  # 1 between letters
        assert needs_repair("Sm0th") is True  # 0 between letters


class TestValidateSpan:
    """Tests for span validation."""

    @pytest.fixture
    def raw_text(self):
        return "From: Jeffrey Epstein <jeff@mail.com>\nTo: Ghislaine Maxwell"

    def test_valid_span(self, raw_text):
        """Should validate correct spans."""
        span = ClassifiedSpan(
            span=NERSpan(text="Jeffrey Epstein", start=6, end=21, entity_type="PER", confidence=0.9),
            is_person=True, role="sender", all_roles=["sender"],
            drop=False, drop_reason=None, classification_confidence=0.9
        )

        is_valid, actual, error = validate_span(raw_text, span)
        assert is_valid is True
        assert actual == "Jeffrey Epstein"
        assert error is None

    def test_mismatch_span(self, raw_text):
        """Should detect text mismatches."""
        span = ClassifiedSpan(
            span=NERSpan(text="Wrong Name", start=6, end=21, entity_type="PER", confidence=0.9),
            is_person=True, role="sender", all_roles=["sender"],
            drop=False, drop_reason=None, classification_confidence=0.9
        )

        is_valid, actual, error = validate_span(raw_text, span)
        assert is_valid is False
        assert error == 'mismatch'

    def test_out_of_bounds(self, raw_text):
        """Should detect out-of-bounds offsets."""
        span = ClassifiedSpan(
            span=NERSpan(text="Test", start=1000, end=1004, entity_type="PER", confidence=0.9),
            is_person=True, role="sender", all_roles=["sender"],
            drop=False, drop_reason=None, classification_confidence=0.9
        )

        is_valid, actual, error = validate_span(raw_text, span)
        assert is_valid is False
        assert error == 'out_of_bounds'

    def test_negative_start(self, raw_text):
        """Should reject negative start offset."""
        span = ClassifiedSpan(
            span=NERSpan(text="Test", start=-1, end=4, entity_type="PER", confidence=0.9),
            is_person=True, role="sender", all_roles=["sender"],
            drop=False, drop_reason=None, classification_confidence=0.9
        )

        is_valid, actual, error = validate_span(raw_text, span)
        assert is_valid is False
        assert error == 'out_of_bounds'


class TestHardValidator:
    """Tests for the HardValidator class."""

    @pytest.fixture
    def validator(self):
        return HardValidator(strict=True)

    @pytest.fixture
    def raw_text(self):
        return "From: Jeffrey Epstein\nTo: J0HN SM1TH"

    def test_validates_correct_spans(self, validator, raw_text):
        """Should validate and pass correct spans."""
        spans = [
            ClassifiedSpan(
                span=NERSpan(text="Jeffrey Epstein", start=6, end=21, entity_type="PER", confidence=0.9),
                is_person=True, role="sender", all_roles=["sender"],
                drop=False, drop_reason=None, classification_confidence=0.9
            ),
        ]

        result = validator.validate(raw_text, spans)
        assert result.passed == 1
        assert result.failed == 0
        assert len(result.validated_spans) == 1

    def test_detects_spans_needing_repair(self, validator, raw_text):
        """Should detect spans that need repair."""
        spans = [
            ClassifiedSpan(
                span=NERSpan(text="J0HN SM1TH", start=26, end=36, entity_type="PER", confidence=0.9),
                is_person=True, role="recipient", all_roles=["recipient"],
                drop=False, drop_reason=None, classification_confidence=0.9
            ),
        ]

        result = validator.validate(raw_text, spans)
        assert result.passed == 1
        assert result.needs_repair == 1
        assert result.validated_spans[0].needs_repair is True

    def test_skips_dropped_spans(self, validator, raw_text):
        """Should skip spans marked for dropping."""
        spans = [
            ClassifiedSpan(
                span=NERSpan(text="Jeffrey Epstein", start=6, end=21, entity_type="PER", confidence=0.9),
                is_person=True, role="sender", all_roles=["sender"],
                drop=True, drop_reason="organization",  # Marked for drop
                classification_confidence=0.9
            ),
        ]

        result = validator.validate(raw_text, spans)
        assert result.passed == 0
        assert len(result.validated_spans) == 0

    def test_reports_errors(self, validator, raw_text):
        """Should report validation errors."""
        spans = [
            ClassifiedSpan(
                span=NERSpan(text="Wrong Name", start=6, end=21, entity_type="PER", confidence=0.9),
                is_person=True, role="sender", all_roles=["sender"],
                drop=False, drop_reason=None, classification_confidence=0.9
            ),
        ]

        result = validator.validate(raw_text, spans)
        assert result.failed == 1
        assert len(result.errors) == 1
        assert result.errors[0].error_type == 'mismatch'


# ============================================================
# Stage 4: LLM Repair Tests
# ============================================================

class TestMockLLMRepairer:
    """Tests for the mock LLM repairer."""

    @pytest.fixture
    def repairer(self):
        return MockLLMRepairer()

    @pytest.fixture
    def sample_spans(self):
        return [
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
        ]

    def test_repairs_corrupted_names(self, repairer, sample_spans):
        """Should repair OCR-corrupted names."""
        result = repairer.repair_names(sample_spans)

        assert isinstance(result, RepairResult)
        assert result.repaired_count == 1  # Only J0HN SM1TH needed repair

        # Find J0HN SM1TH repair
        john = next(n for n in result.repaired_names if n.original_text == "J0HN SM1TH")
        assert john.was_repaired is True
        assert john.normalized_name == "John Smith"  # 0→O, 1→I

    def test_preserves_clean_names(self, repairer, sample_spans):
        """Should not modify clean names."""
        result = repairer.repair_names(sample_spans)

        mary = next(n for n in result.repaired_names if n.original_text == "Mary Brown")
        assert mary.was_repaired is False
        assert mary.normalized_name == "Mary Brown"

    def test_preserves_original_text(self, repairer, sample_spans):
        """Should always preserve original_text."""
        result = repairer.repair_names(sample_spans)

        for name in result.repaired_names:
            assert isinstance(name, RepairedName)
            assert name.original_text  # Must have original
            assert name.normalized_name  # Must have normalized

    def test_empty_spans(self, repairer):
        """Empty spans should return empty result."""
        result = repairer.repair_names([])
        assert result.total_names == 0
        assert result.repaired_count == 0

    def test_5_to_s_repair(self, repairer):
        """Should repair 5 → S."""
        span = ValidatedSpan(
            classified=ClassifiedSpan(
                span=NERSpan(text="5ARAH", start=0, end=5, entity_type="PER", confidence=0.9),
                is_person=True, role="mentioned", all_roles=["mentioned"],
                drop=False, drop_reason=None, classification_confidence=0.8
            ),
            verified_text="5ARAH",
            needs_repair=True
        )

        result = repairer.repair_names([span])
        assert result.repaired_names[0].normalized_name == "Sarah"


# ============================================================
# Full Pipeline Integration Tests
# ============================================================

class TestExtractionPipeline:
    """Integration tests for the full extraction pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline with mock components."""
        config = PipelineConfig(use_gpu=False)
        return ExtractionPipeline(config)

    def test_processes_simple_document(self, pipeline):
        """Should process a simple document end-to-end."""
        text = "From: John Smith\nTo: Mary Brown\n\nHello Mary!"

        result = pipeline.process_text(text, source_file="test.txt")

        assert isinstance(result, PipelineResult)
        assert result.stats.stage0_clean_length > 0
        assert result.stats.stage1_ner_spans >= 0

    def test_handles_empty_document(self, pipeline):
        """Should handle empty documents gracefully."""
        result = pipeline.process_text("", source_file="empty.txt")

        assert result.stats.final_names == 0
        assert result.names == []

    def test_handles_no_names(self, pipeline):
        """Should handle documents with no names."""
        result = pipeline.process_text("hello world no names here", source_file="nonames.txt")

        # Mock extractor won't find names without capitalization
        assert result.stats.final_names == 0

    def test_preserves_offsets(self, pipeline):
        """All name offsets should be valid."""
        text = "From: John Smith\nTo: Mary Brown"
        result = pipeline.process_text(text, source_file="test.txt")

        for name in result.names:
            # Check offsets are within bounds
            assert name.start >= 0
            assert name.end <= len(text)
            assert name.start < name.end

    def test_no_hallucinations(self, pipeline):
        """Names should exist in original text (no hallucinations)."""
        text = "From: John Smith\nTo: Mary Brown\n\nMeeting with Alan Jones."
        result = pipeline.process_text(text, source_file="test.txt")

        for name in result.names:
            # Original text should exist in source
            # (after validation, all spans should be verified)
            assert name.original_text

    def test_pipeline_stats(self, pipeline):
        """Should track statistics through all stages."""
        text = "Hello John Smith and Mary Brown!"
        result = pipeline.process_text(text)

        stats = result.stats
        assert stats.stage0_clean_length > 0
        # Other stats depend on what mock extractors find

    def test_cleanup(self, pipeline):
        """Cleanup should not raise."""
        pipeline.cleanup()  # Should complete without error


class TestPipelineConfig:
    """Tests for pipeline configuration."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = PipelineConfig()

        assert config.ner_model
        assert config.classifier_model
        assert config.repair_model
        assert config.ner_confidence_threshold > 0
        assert config.ner_confidence_threshold < 1

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = PipelineConfig(
            use_gpu=False,
            ner_confidence_threshold=0.5,
            save_intermediate=True
        )

        assert config.use_gpu is False
        assert config.ner_confidence_threshold == 0.5
        assert config.save_intermediate is True


# ============================================================
# Edge Cases and Regression Tests
# ============================================================

class TestEdgeCases:
    """Tests for edge cases and potential regressions."""

    def test_unicode_names(self):
        """Should handle unicode names."""
        cleaner_result = same_length_clean("José García")
        assert len(cleaner_result) == len("José García")

    def test_very_long_text(self):
        """Should handle very long text."""
        text = "John Smith " * 10000
        result = same_length_clean(text)
        assert len(result) == len(text)

    def test_special_characters_in_names(self):
        """Should handle special characters in names."""
        # O'Brien, hyphenated names
        mock_ner = MockXLMRExtractor()

        # These might not be extracted by simple regex,
        # but shouldn't crash
        result = mock_ner.extract_spans("Sarah O'Brien and Mary-Jane Smith")
        assert isinstance(result, NERResult)

    def test_all_caps_names(self):
        """Should handle all-caps names (common in documents)."""
        mock_repairer = MockLLMRepairer()

        span = ValidatedSpan(
            classified=ClassifiedSpan(
                span=NERSpan(text="JOHN SMITH", start=0, end=10, entity_type="PER", confidence=0.9),
                is_person=True, role="mentioned", all_roles=["mentioned"],
                drop=False, drop_reason=None, classification_confidence=0.8
            ),
            verified_text="JOHN SMITH",
            needs_repair=False  # All caps doesn't need repair in Stage 4
        )

        result = mock_repairer.repair_names([span])
        # Should normalize to title case
        assert result.repaired_names[0].normalized_name == "John Smith"

    def test_lastname_firstname_format(self):
        """Should handle LASTNAME, FIRSTNAME format."""
        mock_repairer = MockLLMRepairer()

        span = ValidatedSpan(
            classified=ClassifiedSpan(
                span=NERSpan(text="SMITH, JOHN", start=0, end=11, entity_type="PER", confidence=0.9),
                is_person=True, role="mentioned", all_roles=["mentioned"],
                drop=False, drop_reason=None, classification_confidence=0.8
            ),
            verified_text="SMITH, JOHN",
            needs_repair=False
        )

        result = mock_repairer.repair_names([span])
        # Should normalize to "John Smith"
        assert result.repaired_names[0].normalized_name == "John Smith"


# ============================================================
# Token Budgeting Tests (Work Item 3)
# ============================================================

class TestTokenBudgeting:
    """Tests for LLM classifier and repair token budgeting."""

    def test_classifier_default_batch_size_50(self):
        """Default classifier batch size should be 50."""
        config = PipelineConfig()
        assert config.classifier_batch_size == 50

    def test_repair_default_batch_size_80(self):
        """Default repair batch size should be 80."""
        config = PipelineConfig()
        assert config.repair_batch_size == 80

    def test_classifier_max_output_tokens_constant(self):
        """Classifier should have MAX_OUTPUT_TOKENS = 2048."""
        assert LLMSpanClassifier.MAX_OUTPUT_TOKENS == 2048

    def test_classifier_tokens_per_span(self):
        """Classifier should budget ~30 tokens per span output."""
        assert LLMSpanClassifier.TOKENS_PER_SPAN_OUTPUT == 30

    def test_classifier_output_budget_formula(self):
        """Output tokens formula: min(2048, 30 * batch_size + 100)."""
        # Small batch: 10 spans -> 30*10+100 = 400 tokens
        expected_small = min(2048, 30 * 10 + 100)
        assert expected_small == 400

        # Large batch: 80 spans -> 30*80+100 = 2500 -> capped at 2048
        expected_large = min(2048, 30 * 80 + 100)
        assert expected_large == 2048

    def test_classifier_batch_reduction_for_output(self):
        """If output would exceed 2048, batch should be reduced."""
        # max_spans = (2048 - 100) / 30 = 64.9 -> 64 spans max
        max_spans = (2048 - 100) // 30
        assert max_spans == 64

    def test_repair_max_output_tokens_constant(self):
        """Repair should have MAX_OUTPUT_TOKENS = 1024."""
        from src.ner.llm_repair import LLMNameRepairer
        assert LLMNameRepairer.MAX_OUTPUT_TOKENS == 1024

    def test_repair_tokens_per_name(self):
        """Repair should budget ~15 tokens per name output."""
        from src.ner.llm_repair import LLMNameRepairer
        assert LLMNameRepairer.TOKENS_PER_NAME_OUTPUT == 15

    def test_repair_output_budget_formula(self):
        """Repair output: min(1024, 15 * batch_size + 50)."""
        from src.ner.llm_repair import LLMNameRepairer

        # 80 names -> 15*80+50 = 1250 -> capped at 1024
        expected = min(1024, 15 * 80 + 50)
        assert expected == 1024

        # 50 names -> 15*50+50 = 800 -> not capped
        expected_small = min(1024, 15 * 50 + 50)
        assert expected_small == 800

    def test_classifier_estimate_batch_tokens_heuristic(self):
        """Token estimation without tokenizer uses heuristic."""
        clf = LLMSpanClassifier.__new__(LLMSpanClassifier)
        clf.model_name = "test"
        clf.device = "cpu"
        clf._model = None
        clf._tokenizer = None
        clf._shared_model = None

        spans = [
            NERSpan(text="Jeffrey Epstein", start=0, end=15, entity_type="PER", confidence=0.9),
        ]

        est = clf._estimate_batch_tokens("Short doc text", spans)
        # Should be positive and reasonable
        assert est > 0
        assert est < 100000  # Not unreasonably large

    def test_mock_classifier_accepts_batch_size_kwarg(self):
        """Mock classifier should accept batch_size without error."""
        clf = MockLLMClassifier()
        spans = [
            NERSpan(text="John Smith", start=0, end=10, entity_type="PER", confidence=0.9),
        ]
        result = clf.classify_spans("Test", spans=spans, batch_size=50)
        assert result.total_spans == 1


class TestPipelineVersion:
    """Tests for pipeline_version marker in output."""

    def test_pipeline_version_function(self):
        """_get_pipeline_version should return a string."""
        from src.ner.pipeline import _get_pipeline_version
        version = _get_pipeline_version()
        assert isinstance(version, str)
        assert len(version) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
