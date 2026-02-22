"""Tests for the classifier output truncation fix in llm_classifier.py.

The LLM classifier had a bug where TOKENS_PER_SPAN_OUTPUT=30 and MAX_OUTPUT_TOKENS=2048
caused JSON output truncation for documents with >25 NER spans. The fix:
1. Increased TOKENS_PER_SPAN_OUTPUT to 50, MAX_OUTPUT_TOKENS to 4096
2. Removed indent=2 from JSON in prompts (compact JSON saves tokens)
3. Added _try_repair_json() for salvaging truncated output
4. Added diagnostic counters: parse_fail_count, json_repair_used_count, defaults_count
5. Added GPU validation in shared_model.py

Test categories:
    1. Token budget constants (regression guard)
    2. Batch splitting behavior when output exceeds budget
    3. _try_repair_json() for truncated/malformed JSON
    4. _parse_classifications() with truncated, missing, and valid input
    5. Diagnostic counter accumulation
    6. GPU validation in SharedModelManager
    7. Compact JSON in prompts
    8. Default classification properties
    9. Edge cases (empty spans, single span)
   10. Stress scenarios (200+, 500+ spans)

All tests are CPU-only. torch is mocked if not installed.
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure torch is importable (mock it if not installed)
try:
    import torch
except ImportError:
    # Create a minimal mock torch module so src.ner modules can import
    torch = types.ModuleType("torch")
    torch.cuda = MagicMock()
    torch.cuda.is_available = MagicMock(return_value=False)
    torch.cuda.empty_cache = MagicMock()
    torch.cuda.get_device_name = MagicMock(return_value="MockGPU")
    torch.cuda.get_device_capability = MagicMock(return_value=(0, 0))
    torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    torch.no_grad = MagicMock(return_value=MagicMock(
        __enter__=MagicMock(return_value=None),
        __exit__=MagicMock(return_value=False),
    ))
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.Tensor = type("Tensor", (), {})
    torch.device = lambda x: x
    sys.modules["torch"] = torch

# Now we can safely import the source modules
from src.ner.xlmr_extractor import NERSpan
from src.ner.llm_classifier import LLMSpanClassifier, ClassifiedSpan
from src.ner.shared_model import SharedModelManager


# ============================================================
# Helpers
# ============================================================

def make_spans(n: int) -> list[NERSpan]:
    """Create n test NER spans with distinct offsets."""
    return [
        NERSpan(
            text=f"Person{i}",
            start=i * 20,
            end=i * 20 + 8,
            entity_type="PER",
            confidence=0.9,
        )
        for i in range(n)
    ]


def make_valid_response(n: int) -> str:
    """Build a valid compact JSON response for n spans."""
    entries = []
    for i in range(n):
        entries.append({
            "span_index": i,
            "is_person": True,
            "role": "mentioned",
            "all_roles": ["mentioned"],
            "drop": False,
            "reason": None,
        })
    return json.dumps(entries, separators=(",", ":"))


def make_classifier(**kwargs) -> LLMSpanClassifier:
    """Create an LLMSpanClassifier that skips model loading.

    Uses a mock shared_model so _load_model() is a no-op.
    """
    defaults = dict(shared_model=MagicMock(), device="cpu")
    defaults.update(kwargs)
    classifier = LLMSpanClassifier(**defaults)
    return classifier


def make_llm_classification(span: NERSpan) -> ClassifiedSpan:
    """Create a classification that looks like it came from the LLM (confidence=0.8).

    This avoids triggering the half-batch retry logic which activates when >50% of
    spans have default confidence (0.5).
    """
    return ClassifiedSpan(
        span=span,
        is_person=True,
        role="mentioned",
        all_roles=["mentioned"],
        drop=False,
        drop_reason=None,
        classification_confidence=0.8,
    )


def make_low_confidence_spans(n: int) -> list[NERSpan]:
    """Create n NER spans with confidence below the 0.85 override threshold.

    Spans with NER confidence > 0.85 will have their LLM drop decision overridden.
    Use these when testing drop behavior.
    """
    return [
        NERSpan(
            text=f"Person{i}",
            start=i * 20,
            end=i * 20 + 8,
            entity_type="PER",
            confidence=0.7,
        )
        for i in range(n)
    ]


# ============================================================
# 1. Token budget constants (regression guard)
# ============================================================

class TestTokenBudgetConstants:
    """Verify that the constants reflect the truncation fix, not the old values."""

    def test_tokens_per_span_output_is_50(self):
        """TOKENS_PER_SPAN_OUTPUT must be 50 (was 30 before the fix)."""
        assert LLMSpanClassifier.TOKENS_PER_SPAN_OUTPUT == 50

    def test_max_output_tokens_is_4096(self):
        """MAX_OUTPUT_TOKENS must be 4096 (was 2048 before the fix)."""
        assert LLMSpanClassifier.MAX_OUTPUT_TOKENS == 4096

    def test_max_input_tokens_unchanged(self):
        """MAX_INPUT_TOKENS should remain 24000 (not part of the fix)."""
        assert LLMSpanClassifier.MAX_INPUT_TOKENS == 24000

    def test_budget_for_80_spans_exceeds_max(self):
        """80 spans at 50 tokens each + 100 overhead = 4100, which exceeds
        MAX_OUTPUT_TOKENS. The batch-splitting logic should reduce the batch."""
        budget = LLMSpanClassifier.TOKENS_PER_SPAN_OUTPUT * 80 + 100
        assert budget > LLMSpanClassifier.MAX_OUTPUT_TOKENS, (
            "80 spans should exceed MAX_OUTPUT_TOKENS, triggering a batch split"
        )

    def test_budget_for_25_spans_fits(self):
        """Regression test: 25 spans was the failure point with old constants.
        With new constants (50 tokens/span, 4096 max), 25 spans = 1350 tokens,
        which must fit comfortably within MAX_OUTPUT_TOKENS."""
        budget = LLMSpanClassifier.TOKENS_PER_SPAN_OUTPUT * 25 + 100
        assert budget <= LLMSpanClassifier.MAX_OUTPUT_TOKENS, (
            f"25 spans should fit: {budget} <= {LLMSpanClassifier.MAX_OUTPUT_TOKENS}"
        )

    def test_budget_for_50_spans_fits(self):
        """50 spans at 50 tokens each + 100 = 2600, should fit in 4096."""
        budget = LLMSpanClassifier.TOKENS_PER_SPAN_OUTPUT * 50 + 100
        assert budget <= LLMSpanClassifier.MAX_OUTPUT_TOKENS, (
            f"50 spans should fit: {budget} <= {LLMSpanClassifier.MAX_OUTPUT_TOKENS}"
        )

    def test_max_spans_per_batch_calculation(self):
        """Verify the max spans formula: (MAX_OUTPUT_TOKENS - 100) // TOKENS_PER_SPAN_OUTPUT."""
        max_spans = (LLMSpanClassifier.MAX_OUTPUT_TOKENS - 100) // LLMSpanClassifier.TOKENS_PER_SPAN_OUTPUT
        assert max_spans == 79, (
            f"Max spans per batch should be 79 with current constants, got {max_spans}"
        )

    def test_old_constants_would_truncate_large_batches(self):
        """Verify old constants (30 tokens/span, 2048 max) were insufficient for large batches."""
        old_tokens_per_span = 30
        old_max_output = 2048

        # Old budget for 80 spans: 30*80 + 100 = 2500 > 2048 (overflow)
        old_budget_80 = old_tokens_per_span * 80 + 100
        assert old_budget_80 > old_max_output, "80 spans overflows old budget"

        # New constants still enforce splitting at 80 spans (correct behavior)
        new_budget_80 = LLMSpanClassifier.TOKENS_PER_SPAN_OUTPUT * 80 + 100
        assert new_budget_80 > LLMSpanClassifier.MAX_OUTPUT_TOKENS, (
            "80 spans still overflows new budget (triggering correct split)"
        )

        # But the per-span estimate is more accurate (50 vs 30)
        assert LLMSpanClassifier.TOKENS_PER_SPAN_OUTPUT > old_tokens_per_span


# ============================================================
# 2. Batch splitting behavior
# ============================================================

class TestBatchSplitting:
    """Verify classify_spans correctly splits batches when output tokens exceed budget."""

    def test_100_spans_split_into_batches(self):
        """100 spans should be split into multiple batches that each fit the output budget."""
        classifier = make_classifier()

        batch_sizes = []

        def tracking_classify_batch(document_text, spans, start_index=0):
            batch_sizes.append(len(spans))
            # Return confidence 0.8 to avoid triggering half-batch retry
            return [make_llm_classification(s) for s in spans]

        classifier._classify_batch = tracking_classify_batch

        spans = make_spans(100)
        result = classifier.classify_spans(
            document_text="Some document text " * 50,
            spans=spans,
            source_file="test.pdf",
            batch_size=50,
        )

        assert result.total_spans == 100
        max_spans_per_batch = (
            (LLMSpanClassifier.MAX_OUTPUT_TOKENS - 100) // LLMSpanClassifier.TOKENS_PER_SPAN_OUTPUT
        )
        for bs in batch_sizes:
            assert bs <= max_spans_per_batch, (
                f"Batch of {bs} spans exceeds max {max_spans_per_batch}"
            )
        assert sum(batch_sizes) == 100

    def test_small_batch_not_split(self):
        """A batch of 10 spans should NOT be split (well within budget)."""
        classifier = make_classifier()

        batch_sizes = []

        def tracking_classify_batch(document_text, spans, start_index=0):
            batch_sizes.append(len(spans))
            return [make_llm_classification(s) for s in spans]

        classifier._classify_batch = tracking_classify_batch

        spans = make_spans(10)
        classifier.classify_spans(
            document_text="Short doc",
            spans=spans,
            source_file="test.pdf",
            batch_size=50,
        )

        assert batch_sizes == [10], f"10 spans should go in one batch, got {batch_sizes}"

    def test_79_spans_fits_in_one_batch(self):
        """79 spans is the max that fits: 79*50 + 100 = 4050 <= 4096."""
        classifier = make_classifier()

        batch_sizes = []

        def tracking_classify_batch(document_text, spans, start_index=0):
            batch_sizes.append(len(spans))
            return [make_llm_classification(s) for s in spans]

        classifier._classify_batch = tracking_classify_batch

        spans = make_spans(79)
        classifier.classify_spans(
            document_text="Doc text",
            spans=spans,
            source_file="test.pdf",
            batch_size=79,
        )

        assert batch_sizes == [79], f"79 spans should fit in one batch, got {batch_sizes}"

    def test_80_spans_gets_split(self):
        """80 spans: 80*50 + 100 = 4100 > 4096, so it must split."""
        classifier = make_classifier()

        batch_sizes = []

        def tracking_classify_batch(document_text, spans, start_index=0):
            batch_sizes.append(len(spans))
            return [make_llm_classification(s) for s in spans]

        classifier._classify_batch = tracking_classify_batch

        spans = make_spans(80)
        classifier.classify_spans(
            document_text="Doc text",
            spans=spans,
            source_file="test.pdf",
            batch_size=80,
        )

        assert len(batch_sizes) >= 2, f"80 spans must split into 2+ batches, got {batch_sizes}"
        assert sum(batch_sizes) == 80

    def test_output_budget_capped_at_max(self):
        """_build_classify_prompt max_new_tokens must not exceed MAX_OUTPUT_TOKENS."""
        classifier = make_classifier()
        spans = make_spans(200)
        _, max_new_tokens = classifier._build_classify_prompt("doc text", spans)
        assert max_new_tokens <= LLMSpanClassifier.MAX_OUTPUT_TOKENS, (
            f"max_new_tokens {max_new_tokens} exceeds "
            f"MAX_OUTPUT_TOKENS {LLMSpanClassifier.MAX_OUTPUT_TOKENS}"
        )

    def test_output_budget_scales_with_spans(self):
        """For small batches, max_new_tokens should be proportional to span count."""
        classifier = make_classifier()

        _, tokens_10 = classifier._build_classify_prompt("doc", make_spans(10))
        _, tokens_20 = classifier._build_classify_prompt("doc", make_spans(20))

        expected_10 = 10 * LLMSpanClassifier.TOKENS_PER_SPAN_OUTPUT + 100
        expected_20 = 20 * LLMSpanClassifier.TOKENS_PER_SPAN_OUTPUT + 100

        assert tokens_10 == expected_10
        assert tokens_20 == expected_20
        assert tokens_20 > tokens_10


# ============================================================
# 3. _try_repair_json()
# ============================================================

class TestTryRepairJson:
    """Test the JSON truncation repair mechanism."""

    def setup_method(self):
        self.classifier = make_classifier()
        self.spans = make_spans(5)

    def test_valid_complete_json_returns_all(self):
        """Fully valid JSON array should recover all entries."""
        response = make_valid_response(5)
        result = self.classifier._try_repair_json(response, self.spans)

        assert result is not None
        assert len(result) == 5
        for cs in result:
            assert isinstance(cs, ClassifiedSpan)
            assert cs.is_person is True
            assert cs.role == "mentioned"

    def test_truncated_json_recovers_partial(self):
        """JSON truncated mid-entry should recover complete entries before the cut."""
        full = json.dumps([
            {"span_index": i, "is_person": True, "role": "mentioned",
             "all_roles": ["mentioned"], "drop": False, "reason": None}
            for i in range(5)
        ])
        # Cut after the 3rd complete object
        brace_positions = [i for i, c in enumerate(full) if c == '}']
        assert len(brace_positions) >= 3
        truncated = full[:brace_positions[2] + 1] + ', {"span_index": 3, "is_pers'

        result = self.classifier._try_repair_json(truncated, self.spans)

        assert result is not None
        recovered_count = sum(1 for cs in result if cs.classification_confidence == 0.8)
        assert recovered_count >= 3, f"Expected >=3 recovered, got {recovered_count}"

    def test_no_json_at_all_returns_none(self):
        """Response with no JSON array should return None."""
        result = self.classifier._try_repair_json(
            "This is just plain text with no JSON.", self.spans
        )
        assert result is None

    def test_only_opening_bracket_returns_none(self):
        """Response with just '[' should return None (no complete object)."""
        result = self.classifier._try_repair_json("[", self.spans)
        assert result is None

    def test_empty_array_returns_none(self):
        """Empty JSON array '[]' should return None (0 entries is not useful)."""
        result = self.classifier._try_repair_json("[]", self.spans)
        assert result is None

    def test_json_cut_mid_key(self):
        """JSON cut in the middle of a key name should recover preceding objects."""
        entry0 = (
            '{"span_index":0,"is_person":true,"role":"mentioned",'
            '"all_roles":["mentioned"],"drop":false,"reason":null}'
        )
        truncated = f'[{entry0},{{"span_in'

        result = self.classifier._try_repair_json(truncated, self.spans)

        assert result is not None
        recovered = [cs for cs in result if cs.classification_confidence == 0.8]
        assert len(recovered) >= 1

    def test_json_cut_mid_value(self):
        """JSON cut in the middle of a value should recover preceding objects."""
        entry0 = (
            '{"span_index":0,"is_person":true,"role":"mentioned",'
            '"all_roles":["mentioned"],"drop":false,"reason":null}'
        )
        entry1 = (
            '{"span_index":1,"is_person":true,"role":"sender",'
            '"all_roles":["sender"],"drop":false,"reason":null}'
        )
        truncated = (
            f'[{entry0},{entry1},'
            '{"span_index":2,"is_person":true,"role":"menti'
        )

        result = self.classifier._try_repair_json(truncated, self.spans)

        assert result is not None
        recovered = [cs for cs in result if cs.classification_confidence == 0.8]
        assert len(recovered) >= 2

    def test_trailing_comma_handled(self):
        """Trailing comma after last complete object should not prevent parsing."""
        entry0 = (
            '{"span_index":0,"is_person":true,"role":"mentioned",'
            '"all_roles":["mentioned"],"drop":false,"reason":null}'
        )
        entry1 = (
            '{"span_index":1,"is_person":true,"role":"sender",'
            '"all_roles":["sender"],"drop":false,"reason":null}'
        )
        truncated = f'[{entry0},{entry1},'

        result = self.classifier._try_repair_json(truncated, self.spans)

        assert result is not None
        recovered = [cs for cs in result if cs.classification_confidence == 0.8]
        assert len(recovered) == 2

    def test_repaired_entries_have_correct_spans(self):
        """Recovered entries should be correctly mapped to their NER spans."""
        entry0 = (
            '{"span_index":0,"is_person":true,"role":"sender",'
            '"all_roles":["sender"],"drop":false,"reason":null}'
        )
        entry1 = (
            '{"span_index":1,"is_person":false,"role":"mentioned",'
            '"all_roles":[],"drop":true,"reason":"organization"}'
        )
        truncated = f'[{entry0},{entry1},{{"span_in'

        result = self.classifier._try_repair_json(truncated, self.spans)

        assert result is not None
        # Entry 0: Person0, sender
        assert result[0].span.text == "Person0"
        assert result[0].role == "sender"
        assert result[0].is_person is True
        # Entry 1: Person1, dropped
        assert result[1].span.text == "Person1"
        assert result[1].drop is True
        assert result[1].drop_reason == "organization"

    def test_unrecovered_spans_get_defaults(self):
        """Spans beyond the recovered entries should get default classification."""
        entry0 = (
            '{"span_index":0,"is_person":true,"role":"sender",'
            '"all_roles":["sender"],"drop":false,"reason":null}'
        )
        truncated = f'[{entry0},{{"span_in'

        result = self.classifier._try_repair_json(truncated, self.spans)

        assert result is not None
        # Entry 0: recovered
        assert result[0].classification_confidence == 0.8
        assert result[0].role == "sender"
        # Entries 1-4: defaults
        for i in range(1, 5):
            assert result[i].classification_confidence == 0.5, (
                f"Span {i} should have default confidence 0.5, "
                f"got {result[i].classification_confidence}"
            )
            assert result[i].is_person is True
            assert result[i].role == "mentioned"

    def test_text_before_json_is_ignored(self):
        """Preamble text before the JSON array should be skipped."""
        entry0 = (
            '{"span_index":0,"is_person":true,"role":"mentioned",'
            '"all_roles":["mentioned"],"drop":false,"reason":null}'
        )
        response = f'Here are the classifications:\n[{entry0}]'

        result = self.classifier._try_repair_json(response, self.spans)

        assert result is not None
        recovered = [cs for cs in result if cs.classification_confidence == 0.8]
        assert len(recovered) == 1

    def test_nested_braces_in_response(self):
        """Response with nested braces should still parse correctly."""
        entry = (
            '{"span_index":0,"is_person":true,"role":"mentioned",'
            '"all_roles":["mentioned"],"drop":false,"reason":null}'
        )
        truncated = f'[{entry},{{"span_index":1,"is_per'

        result = self.classifier._try_repair_json(truncated, self.spans)
        assert result is not None


# ============================================================
# 4. _parse_classifications() with various inputs
# ============================================================

class TestParseClassifications:
    """Test _parse_classifications with valid, truncated, and missing JSON."""

    def setup_method(self):
        self.classifier = make_classifier()
        self.spans = make_spans(3)

    def test_valid_json_parsed_correctly(self):
        """Complete valid JSON response should yield correct classifications."""
        # Use low-confidence spans so that the high-confidence override (>0.85)
        # does not interfere with the drop=True assertion on span 2.
        low_conf_spans = make_low_confidence_spans(3)

        response = json.dumps([
            {"span_index": 0, "is_person": True, "role": "sender",
             "all_roles": ["sender"], "drop": False, "reason": None},
            {"span_index": 1, "is_person": True, "role": "recipient",
             "all_roles": ["recipient"], "drop": False, "reason": None},
            {"span_index": 2, "is_person": False, "role": None,
             "all_roles": [], "drop": True, "reason": "organization"},
        ])

        result = self.classifier._parse_classifications(response, low_conf_spans)

        assert len(result) == 3
        assert result[0].role == "sender"
        assert result[1].role == "recipient"
        assert result[2].drop is True
        assert result[2].drop_reason == "organization"

        # No error counters should have incremented
        assert self.classifier.parse_fail_count == 0
        assert self.classifier.json_repair_used_count == 0
        assert self.classifier.defaults_count == 0

    def test_no_json_increments_parse_fail_and_defaults(self):
        """Response with no JSON should increment parse_fail_count and defaults_count."""
        response = "I cannot classify these spans because I am confused."

        result = self.classifier._parse_classifications(response, self.spans)

        assert len(result) == 3
        for cs in result:
            assert cs.classification_confidence == 0.5
            assert cs.role == "mentioned"
            assert cs.is_person is True

        assert self.classifier.parse_fail_count == 1
        assert self.classifier.defaults_count == 3

    def test_truncated_json_triggers_repair(self):
        """Truncated JSON should trigger repair, incrementing json_repair_used_count."""
        entry0 = (
            '{"span_index":0,"is_person":true,"role":"sender",'
            '"all_roles":["sender"],"drop":false,"reason":null}'
        )
        entry1 = (
            '{"span_index":1,"is_person":true,"role":"recipient",'
            '"all_roles":["recipient"],"drop":false,"reason":null}'
        )
        # The regex r'\[[\s\S]*\]' needs a closing bracket to match.
        # Include one so the regex matches, but json.loads fails on the malformed content.
        response_with_bracket = f'[{entry0},{entry1},{{"span_index":2,"is_pers]'

        result = self.classifier._parse_classifications(
            response_with_bracket, self.spans
        )

        # The regex finds [...] but json.loads fails, triggering JSONDecodeError.
        # Then _try_repair_json is called and recovers entries 0 and 1.
        assert self.classifier.parse_fail_count == 1
        assert self.classifier.json_repair_used_count == 1
        assert len(result) == 3
        assert result[0].role == "sender"
        assert result[0].classification_confidence == 0.8
        assert result[1].role == "recipient"
        assert result[1].classification_confidence == 0.8
        # Third entry should be default
        assert result[2].classification_confidence == 0.5

    def test_repair_fails_returns_all_defaults(self):
        """When JSON repair also fails, all spans get defaults."""
        response = "[not_json_at_all_but_has_brackets]"

        result = self.classifier._parse_classifications(response, self.spans)

        assert len(result) == 3
        for cs in result:
            assert cs.classification_confidence == 0.5

        assert self.classifier.parse_fail_count == 1
        assert self.classifier.json_repair_used_count == 0  # repair failed
        assert self.classifier.defaults_count == 3

    def test_mismatched_count_uses_positional_fallback(self):
        """LLM returning fewer entries than spans should use positional fallback."""
        response = json.dumps([
            {"span_index": 0, "is_person": True, "role": "sender",
             "all_roles": ["sender"], "drop": False, "reason": None},
        ])

        result = self.classifier._parse_classifications(response, self.spans)

        assert len(result) == 3
        assert result[0].role == "sender"
        assert result[0].classification_confidence == 0.8
        assert result[1].classification_confidence == 0.5  # default
        assert result[2].classification_confidence == 0.5  # default
        assert self.classifier.defaults_count == 2

    def test_empty_response_returns_defaults(self):
        """Empty string response should return defaults for all spans."""
        result = self.classifier._parse_classifications("", self.spans)

        assert len(result) == 3
        for cs in result:
            assert cs.classification_confidence == 0.5
        assert self.classifier.parse_fail_count == 1

    def test_valid_json_with_preamble_text(self):
        """Response with text before the JSON array should still parse."""
        json_data = json.dumps([
            {"span_index": 0, "is_person": True, "role": "sender",
             "all_roles": ["sender"], "drop": False, "reason": None},
            {"span_index": 1, "is_person": True, "role": "mentioned",
             "all_roles": ["mentioned"], "drop": False, "reason": None},
            {"span_index": 2, "is_person": True, "role": "mentioned",
             "all_roles": ["mentioned"], "drop": False, "reason": None},
        ])
        response = f"Here are the classifications:\n{json_data}"

        result = self.classifier._parse_classifications(response, self.spans)

        assert len(result) == 3
        assert result[0].role == "sender"
        assert self.classifier.parse_fail_count == 0

    def test_high_confidence_ner_span_not_dropped(self):
        """NER spans with confidence > 0.85 should not be dropped even if LLM says to."""
        spans = [
            NERSpan(
                text="Jeffrey Epstein", start=0, end=15,
                entity_type="PER", confidence=0.95,
            )
        ]
        response = json.dumps([{
            "span_index": 0, "is_person": False, "role": None,
            "all_roles": [], "drop": True, "reason": "not_a_person",
        }])

        result = self.classifier._parse_classifications(
            response, spans, document_text="Jeffrey Epstein was here."
        )

        assert len(result) == 1
        # Drop should be overridden because NER confidence > 0.85
        assert result[0].drop is False
        assert result[0].drop_reason is None


# ============================================================
# 5. Diagnostic counter accumulation
# ============================================================

class TestDiagnosticCounters:
    """Verify diagnostic counters initialize correctly and accumulate across calls."""

    def test_fresh_classifier_counters_at_zero(self):
        """A newly created classifier should have all counters at 0."""
        classifier = make_classifier()
        assert classifier.parse_fail_count == 0
        assert classifier.json_repair_used_count == 0
        assert classifier.defaults_count == 0

    def test_counters_accumulate_across_calls(self):
        """Multiple _parse_classifications calls should accumulate counters."""
        classifier = make_classifier()
        spans = make_spans(2)

        # Call 1: valid response, no errors
        valid = json.dumps([
            {"span_index": 0, "is_person": True, "role": "mentioned",
             "all_roles": ["mentioned"], "drop": False, "reason": None},
            {"span_index": 1, "is_person": True, "role": "mentioned",
             "all_roles": ["mentioned"], "drop": False, "reason": None},
        ])
        classifier._parse_classifications(valid, spans)
        assert classifier.parse_fail_count == 0
        assert classifier.defaults_count == 0

        # Call 2: no JSON, all defaults
        classifier._parse_classifications("garbage text", spans)
        assert classifier.parse_fail_count == 1
        assert classifier.defaults_count == 2

        # Call 3: no JSON again, counters should accumulate
        classifier._parse_classifications("more garbage", spans)
        assert classifier.parse_fail_count == 2
        assert classifier.defaults_count == 4

    def test_repair_counter_accumulates(self):
        """json_repair_used_count should increment each time repair succeeds."""
        classifier = make_classifier()
        spans = make_spans(3)

        entry0 = (
            '{"span_index":0,"is_person":true,"role":"mentioned",'
            '"all_roles":["mentioned"],"drop":false,"reason":null}'
        )
        # Truncated JSON that triggers JSONDecodeError, then successful repair
        truncated = f'[{entry0},{{"span_index":1,"is]'

        classifier._parse_classifications(truncated, spans)
        assert classifier.parse_fail_count == 1
        assert classifier.json_repair_used_count == 1

        # Again
        classifier._parse_classifications(truncated, spans)
        assert classifier.parse_fail_count == 2
        assert classifier.json_repair_used_count == 2

    def test_defaults_count_from_missing_span_indices(self):
        """When LLM skips span indices, defaults_count increments per missing span."""
        classifier = make_classifier()
        spans = make_spans(5)

        # Only provide entries for indices 0 and 2 (skip 1, 3, 4)
        response = json.dumps([
            {"span_index": 0, "is_person": True, "role": "sender",
             "all_roles": ["sender"], "drop": False, "reason": None},
            {"span_index": 2, "is_person": True, "role": "mentioned",
             "all_roles": ["mentioned"], "drop": False, "reason": None},
        ])

        result = classifier._parse_classifications(response, spans)

        assert len(result) == 5
        assert classifier.defaults_count == 3
        assert result[0].classification_confidence == 0.8  # from LLM
        assert result[1].classification_confidence == 0.5  # default
        assert result[2].classification_confidence == 0.8  # from LLM
        assert result[3].classification_confidence == 0.5  # default
        assert result[4].classification_confidence == 0.5  # default

    def test_counters_survive_mixed_success_failure(self):
        """Counters accumulate correctly through a mix of successes and failures."""
        classifier = make_classifier()
        spans2 = make_spans(2)
        spans3 = make_spans(3)

        # Success (0 errors)
        valid = make_valid_response(2)
        classifier._parse_classifications(valid, spans2)
        assert classifier.parse_fail_count == 0
        assert classifier.json_repair_used_count == 0
        assert classifier.defaults_count == 0

        # Parse failure with no repair possible
        classifier._parse_classifications("[broken json here]", spans3)
        assert classifier.parse_fail_count == 1
        assert classifier.defaults_count == 3

        # Success again (counters should not reset)
        valid3 = make_valid_response(3)
        classifier._parse_classifications(valid3, spans3)
        assert classifier.parse_fail_count == 1
        assert classifier.defaults_count == 3

        # Another failure
        classifier._parse_classifications("no json", spans2)
        assert classifier.parse_fail_count == 2
        assert classifier.defaults_count == 5  # 3 + 2


# ============================================================
# 6. GPU validation (shared_model.py)
# ============================================================

class TestGPUValidation:
    """Test _validate_gpu() in SharedModelManager."""

    def test_validate_gpu_fails_for_low_compute(self):
        """GPU with compute capability < 8.0 should raise RuntimeError."""
        manager = SharedModelManager(model_name="test", device="cuda")

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_name", return_value="NVIDIA GeForce Tier 1 GPU Ti"), \
             patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)):
            with pytest.raises(RuntimeError, match="below minimum 8.0"):
                manager._validate_gpu()

    def test_validate_gpu_passes_for_h200(self):
        """Tier 2 GPU (compute 9.0) should pass validation."""
        manager = SharedModelManager(model_name="test", device="cuda")

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_name", return_value="NVIDIA Tier 2 GPU"), \
             patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            manager._validate_gpu()

    def test_validate_gpu_passes_for_a100(self):
        """A100 (compute 8.0) should pass validation (minimum threshold)."""
        manager = SharedModelManager(model_name="test", device="cuda")

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_name", return_value="NVIDIA A100-SXM4-80GB"), \
             patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
            manager._validate_gpu()

    def test_validate_gpu_passes_for_l40s(self):
        """L40s (compute 8.9) should pass validation."""
        manager = SharedModelManager(model_name="test", device="cuda")

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_name", return_value="NVIDIA L40S"), \
             patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)):
            manager._validate_gpu()

    def test_validate_gpu_skipped_on_cpu(self):
        """CPU mode should skip GPU validation entirely."""
        manager = SharedModelManager(model_name="test", device="cpu")

        with patch.object(torch.cuda, "is_available", return_value=False):
            manager._validate_gpu()

    def test_validate_gpu_fails_for_rtx_2080(self):
        """Tier 1 GPU (compute 7.5) is explicitly too low for LLM workloads."""
        manager = SharedModelManager(model_name="test", device="cuda")

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_name", return_value="NVIDIA GeForce Tier 1 GPU"), \
             patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)):
            with pytest.raises(RuntimeError, match="Tier 1 GPU"):
                manager._validate_gpu()

    def test_validate_gpu_error_message_includes_gpu_name(self):
        """Error message should include the GPU name for diagnostics."""
        manager = SharedModelManager(model_name="test", device="cuda")

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_name", return_value="NVIDIA Tesla V100"), \
             patch.object(torch.cuda, "get_device_capability", return_value=(7, 0)):
            with pytest.raises(RuntimeError, match="Tesla V100"):
                manager._validate_gpu()


# ============================================================
# 7. Compact JSON in prompts (no indent)
# ============================================================

class TestCompactJsonInPrompts:
    """Verify prompts use compact JSON (no indent=2) to save output tokens."""

    def test_build_prompt_uses_compact_json(self):
        """Spans JSON in the prompt should be compact (no indentation)."""
        classifier = make_classifier()
        spans = make_spans(3)

        messages, _ = classifier._build_classify_prompt("Document text", spans)

        user_content = messages[1]["content"]
        # Extract the spans JSON section between the markers
        spans_section = user_content.split("NER spans to classify:")[1].split(
            "Return ONLY"
        )[0]
        # Should NOT contain 4-space indentation (compact JSON)
        assert "    " not in spans_section, (
            "Prompt should use compact JSON for spans, not indented"
        )


# ============================================================
# 8. Default classification properties
# ============================================================

class TestDefaultClassification:
    """Verify the default classification used as fallback."""

    def test_default_classification_values(self):
        """Default classification: is_person=True, role=mentioned, confidence=0.5."""
        classifier = make_classifier()
        span = make_spans(1)[0]

        default = classifier._default_classification(span)

        assert default.span is span
        assert default.is_person is True
        assert default.role == "mentioned"
        assert default.all_roles == ["mentioned"]
        assert default.drop is False
        assert default.drop_reason is None
        assert default.classification_confidence == 0.5

    def test_default_confidence_distinguishable_from_llm(self):
        """Default confidence (0.5) must differ from LLM confidence (0.8)
        so callers can detect defaults."""
        classifier = make_classifier()
        span = make_spans(1)[0]

        default = classifier._default_classification(span)

        assert default.classification_confidence != 0.8
        assert default.classification_confidence == 0.5


# ============================================================
# 9. Edge cases for classify_spans
# ============================================================

class TestClassifySpansEdgeCases:
    """Edge cases for the top-level classify_spans method."""

    def test_empty_spans_returns_empty_result(self):
        """classify_spans with no spans should return an empty result immediately."""
        classifier = make_classifier()

        result = classifier.classify_spans(
            document_text="Some text",
            spans=[],
            source_file="empty.pdf",
        )

        assert result.total_spans == 0
        assert result.person_spans == 0
        assert result.dropped_spans == 0
        assert result.classified_spans == []
        assert result.source_file == "empty.pdf"

    def test_single_span_classified(self):
        """classify_spans with a single span should work without splitting."""
        classifier = make_classifier()
        spans = make_spans(1)

        def mock_batch(doc_text, batch_spans, start_index=0):
            return [ClassifiedSpan(
                span=batch_spans[0],
                is_person=True,
                role="sender",
                all_roles=["sender"],
                drop=False,
                drop_reason=None,
                classification_confidence=0.8,
            )]

        classifier._classify_batch = mock_batch

        result = classifier.classify_spans(
            document_text="From: Person0",
            spans=spans,
            source_file="single.pdf",
        )

        assert result.total_spans == 1
        assert result.person_spans == 1
        assert result.classified_spans[0].role == "sender"


# ============================================================
# 10. Stress test: many spans
# ============================================================

class TestStressScenarios:
    """Verify correct behavior with large span counts."""

    def test_200_spans_all_classified(self):
        """200 spans should all be classified (via multiple batches)."""
        classifier = make_classifier()

        call_count = [0]

        def mock_batch(doc_text, batch_spans, start_index=0):
            call_count[0] += 1
            return [make_llm_classification(s) for s in batch_spans]

        classifier._classify_batch = mock_batch

        spans = make_spans(200)
        result = classifier.classify_spans(
            document_text="Large document " * 100,
            spans=spans,
            source_file="large.pdf",
            batch_size=50,
        )

        assert result.total_spans == 200
        assert len(result.classified_spans) == 200
        # batch_size=50 fits output budget (50*50+100=2600 < 4096) so 4 batches of 50.
        assert call_count[0] == 4, f"Expected 4 batches, got {call_count[0]}"

    def test_500_spans_completes_without_error(self):
        """500 spans should complete without error."""
        classifier = make_classifier()

        classifier._classify_batch = lambda doc, spans, start_index=0: [
            make_llm_classification(s) for s in spans
        ]

        spans = make_spans(500)
        result = classifier.classify_spans(
            document_text="Very large document " * 200,
            spans=spans,
            source_file="huge.pdf",
            batch_size=50,
        )

        assert result.total_spans == 500
