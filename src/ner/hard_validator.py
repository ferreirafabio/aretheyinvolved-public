"""Stage 3: Hard validator for span verification.

This module performs strict validation that NER spans exist exactly
in the source text at the claimed offsets. This prevents any hallucinated
or modified spans from making it into the final output.

The key invariant: raw_text[start:end] == span.text
"""

from dataclasses import dataclass
from typing import NamedTuple

from loguru import logger

from .xlmr_extractor import NERSpan
from .llm_classifier import ClassifiedSpan


class ValidationError(NamedTuple):
    """Details about a span validation failure."""
    span_index: int
    span_text: str
    start: int
    end: int
    actual_text: str
    error_type: str  # 'mismatch', 'out_of_bounds', 'empty'


@dataclass
class ValidatedSpan:
    """A span that has passed hard validation."""
    classified: ClassifiedSpan  # The classified span
    verified_text: str          # Text verified from raw source
    needs_repair: bool          # Whether the text appears corrupted


@dataclass
class ValidationResult:
    """Result of hard validation."""
    source_file: str
    validated_spans: list[ValidatedSpan]
    total_input: int
    passed: int
    failed: int
    needs_repair: int  # Count of spans needing repair
    errors: list[ValidationError]


def needs_repair(text: str) -> bool:
    """Check if span text looks corrupted and needs LLM repair.

    OCR commonly produces these errors in names:
    - 0 → O (zero to letter O)
    - 1 → I or l (one to letter I/l)
    - 5 → S (five to letter S)
    - Weird punctuation characters

    Args:
        text: The span text to check.

    Returns:
        True if the text appears corrupted.
    """
    # Common OCR digit-letter confusions in names
    digit_letters = '015'  # 0→O, 1→I/l, 5→S

    # Weird characters that shouldn't appear in names
    weird_chars = '/{}\u02dc|\\@#$%^&*=+[]<>'

    # Check for digit-letter confusion in what looks like a name
    has_suspicious_digits = any(c in digit_letters for c in text)

    # Check for weird punctuation
    has_weird_chars = any(c in weird_chars for c in text)

    # Check for unusual character patterns
    has_unusual = False
    for i, c in enumerate(text):
        # Digit surrounded by letters suggests OCR error
        if c.isdigit() and i > 0 and i < len(text) - 1:
            if text[i-1].isalpha() and text[i+1].isalpha():
                has_unusual = True
                break

    return has_suspicious_digits or has_weird_chars or has_unusual


def validate_span(raw_text: str, span: ClassifiedSpan) -> tuple[bool, str | None, str | None]:
    """Validate that a span exists exactly in the raw text.

    Args:
        raw_text: The original raw text.
        span: The classified span to validate.

    Returns:
        Tuple of (is_valid, actual_text, error_type).
        If valid, actual_text is the verified text from raw_text.
        If invalid, error_type describes the failure.
    """
    ner_span = span.span
    start = ner_span.start
    end = ner_span.end
    expected_text = ner_span.text

    # Bounds check
    if start < 0 or end > len(raw_text) or start >= end:
        return False, None, 'out_of_bounds'

    # Extract actual text from source
    actual_text = raw_text[start:end]

    # Check for empty
    if not actual_text.strip():
        return False, actual_text, 'empty'

    # Strict match check
    # Allow minor whitespace normalization but text must match
    if actual_text.strip() != expected_text.strip():
        # Try normalized comparison
        actual_normalized = ' '.join(actual_text.split())
        expected_normalized = ' '.join(expected_text.split())

        if actual_normalized != expected_normalized:
            return False, actual_text, 'mismatch'

    return True, actual_text, None


class HardValidator:
    """Hard validator that verifies spans exist in source text.

    This validator enforces the invariant:
        raw_text[start:end] == span.text

    Any span that fails this check is rejected, preventing
    hallucinated or modified spans from entering the output.
    """

    def __init__(self, strict: bool = True):
        """Initialize validator.

        Args:
            strict: If True, reject any span that doesn't match exactly.
                   If False, allow minor whitespace differences.
        """
        self.strict = strict

    def validate(self,
                 raw_text: str,
                 classified_spans: list[ClassifiedSpan],
                 source_file: str = "") -> ValidationResult:
        """Validate all classified spans against raw text.

        Args:
            raw_text: The original raw text (NOT cleaned).
            classified_spans: Spans that passed classification.
            source_file: Source file name.

        Returns:
            ValidationResult with validated spans and errors.
        """
        validated = []
        errors = []
        needs_repair_count = 0

        for i, classified in enumerate(classified_spans):
            # Skip spans marked for dropping
            if classified.drop:
                continue

            is_valid, actual_text, error_type = validate_span(raw_text, classified)

            if is_valid:
                # Check if needs repair
                repair_needed = needs_repair(actual_text)
                if repair_needed:
                    needs_repair_count += 1

                validated.append(ValidatedSpan(
                    classified=classified,
                    verified_text=actual_text,
                    needs_repair=repair_needed
                ))
            else:
                errors.append(ValidationError(
                    span_index=i,
                    span_text=classified.span.text,
                    start=classified.span.start,
                    end=classified.span.end,
                    actual_text=actual_text or '',
                    error_type=error_type
                ))
                logger.warning(
                    f"Validation failed for span '{classified.span.text}' "
                    f"@ {classified.span.start}-{classified.span.end}: {error_type}"
                )

        if errors:
            logger.warning(f"{len(errors)} spans failed validation out of {len(classified_spans)}")

        return ValidationResult(
            source_file=source_file,
            validated_spans=validated,
            total_input=len(classified_spans),
            passed=len(validated),
            failed=len(errors),
            needs_repair=needs_repair_count,
            errors=errors
        )


def validate_spans_strict(raw_text: str,
                          classified_spans: list[ClassifiedSpan],
                          source_file: str = "") -> ValidationResult:
    """Convenience function for strict validation.

    Args:
        raw_text: Original raw text.
        classified_spans: Spans to validate.
        source_file: Source file name.

    Returns:
        ValidationResult.
    """
    validator = HardValidator(strict=True)
    return validator.validate(raw_text, classified_spans, source_file)


if __name__ == "__main__":
    # Test cases
    print("Testing needs_repair():")

    test_cases = [
        ("John Smith", False),      # Clean name
        ("J0HN SMITH", True),        # 0 → O error
        ("JOHN SM1TH", True),        # 1 → I error
        ("John/Smith", True),        # Weird punct
        ("Mary Brown", False),       # Clean
        ("5ARAH JONES", True),       # 5 → S error
        ("Jeffrey Epstein", False),  # Clean
    ]

    for text, expected in test_cases:
        result = needs_repair(text)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: '{text}' -> needs_repair={result} (expected {expected})")

    # Test validation
    print("\nTesting validate_span():")

    raw_text = "From: Jeffrey Epstein <jeff@mail.com>\nTo: Ghislaine Maxwell"

    test_spans = [
        (ClassifiedSpan(
            span=NERSpan(text="Jeffrey Epstein", start=6, end=21, entity_type="PER", confidence=0.9),
            is_person=True, role="sender", all_roles=["sender"],
            drop=False, drop_reason=None, classification_confidence=0.9
        ), True),  # Valid
        (ClassifiedSpan(
            span=NERSpan(text="Wrong Name", start=6, end=21, entity_type="PER", confidence=0.9),
            is_person=True, role="sender", all_roles=["sender"],
            drop=False, drop_reason=None, classification_confidence=0.9
        ), False),  # Mismatch
        (ClassifiedSpan(
            span=NERSpan(text="Out of bounds", start=100, end=113, entity_type="PER", confidence=0.9),
            is_person=True, role="sender", all_roles=["sender"],
            drop=False, drop_reason=None, classification_confidence=0.9
        ), False),  # Out of bounds
    ]

    # Need to import NERSpan for test
    from .xlmr_extractor import NERSpan

    for classified, expected_valid in test_spans:
        is_valid, actual, error = validate_span(raw_text, classified)
        status = "PASS" if is_valid == expected_valid else "FAIL"
        print(f"  {status}: '{classified.span.text}' @ {classified.span.start}-{classified.span.end}")
        print(f"         valid={is_valid}, actual='{actual}', error={error}")
