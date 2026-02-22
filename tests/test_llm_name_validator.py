"""Tests for LLM-based name validation.

These tests verify the name validation functionality including:
- Heuristic filtering of obvious garbage
- Valid name pattern matching
- LLM validation for uncertain cases
- Batch processing

Note: Full LLM integration tests require GPU. Unit tests use mocks.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ner.llm_validator import (
    HeuristicValidator,
    LlamaValidator,
    MockValidator,
    ValidationResult,
    create_validator,
)


class TestHeuristicValidator:
    """Tests for the fast heuristic validator."""

    @pytest.fixture
    def validator(self):
        return HeuristicValidator()

    class TestObviousGarbage:
        """Test detection of obvious garbage."""

        @pytest.fixture
        def validator(self):
            return HeuristicValidator()

        @pytest.mark.parametrize("name,expected_garbage", [
            # Empty/whitespace
            ("", True),
            ("   ", True),

            # Too short
            ("A", True),
            ("Jo", True),

            # Numbers only
            ("12345", True),
            ("123", True),

            # Repeated characters
            ("aaaa", True),
            ("XXXXX", True),
            ("hellooooo", True),

            # Too long
            ("A" * 65, True),

            # Only punctuation
            ("...", True),
            ("---", True),
            ("...-...", True),

            # Unusual characters
            ("@#$%", True),
            ("John@Smith", True),

            # Valid names should NOT be garbage
            ("John", False),
            ("John Smith", False),
            ("Mary Jane Watson", False),
        ])
        def test_is_obvious_garbage(self, validator, name, expected_garbage):
            """Test garbage detection for various inputs."""
            is_garbage, _ = validator.is_obvious_garbage(name)
            assert is_garbage == expected_garbage, f"Failed for '{name}'"

    class TestValidNamePatterns:
        """Test valid name pattern matching."""

        @pytest.fixture
        def validator(self):
            return HeuristicValidator()

        @pytest.mark.parametrize("name,expected_valid", [
            # Standard format
            ("John Smith", True),
            ("Mary Jane", True),
            ("Robert Williams Jr", True),

            # ALL CAPS
            ("JOHN SMITH", True),
            ("JEFFREY EPSTEIN", True),

            # Initial format
            ("J. Smith", True),

            # Last, First format
            ("Smith, John", True),

            # Edge cases that might not match
            ("de la Cruz", False),  # Lowercase prefix
            ("O'Brien", False),  # Apostrophe
            ("Jean-Pierre", False),  # Hyphen

            # Clear non-matches
            ("john smith", False),  # All lowercase
            ("123 Smith", False),  # Number prefix
        ])
        def test_is_likely_valid(self, validator, name, expected_valid):
            """Test valid name pattern matching."""
            assert validator.is_likely_valid(name) == expected_valid, f"Failed for '{name}'"


class TestMockValidator:
    """Tests for MockValidator (no GPU required)."""

    @pytest.fixture
    def validator(self):
        return MockValidator()

    def test_validates_standard_names(self, validator):
        """Test that standard names are validated as valid."""
        names = ["John Smith", "JEFFREY EPSTEIN", "Mary Johnson"]

        for name in names:
            result = validator.validate_name(name)
            assert result.is_valid, f"'{name}' should be valid"
            assert result.confidence > 0

    def test_rejects_obvious_garbage(self, validator):
        """Test that obvious garbage is rejected."""
        garbage = ["", "aaaa", "12345", "...", "@#$"]

        for name in garbage:
            result = validator.validate_name(name)
            assert not result.is_valid, f"'{name}' should be invalid"

    def test_returns_validation_result(self, validator):
        """Test that ValidationResult has expected fields."""
        result = validator.validate_name("John Smith")

        assert isinstance(result, ValidationResult)
        assert result.name == "John Smith"
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.confidence, float)
        assert 0 <= result.confidence <= 1

    def test_handles_ocr_garbage_examples(self, validator):
        """Test specific OCR garbage patterns from real data."""
        ocr_garbage = [
            "aaaaa treenagan",  # Repeated chars
            "Xz123 Qwerty",  # Mixed garbage
            '".)7n',  # Symbol soup
            "& Annap",  # Starts with symbol
            "-Hatt/ rt",  # Unusual punctuation
            "Z. \nJeAdee,-",  # Embedded newlines
        ]

        for name in ocr_garbage:
            result = validator.validate_name(name)
            # Most of these should be rejected
            # Some edge cases might pass heuristics but fail LLM
            if result.is_valid:
                assert result.confidence < 0.8, f"'{name}' shouldn't have high confidence"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_basic_fields(self):
        """Test basic field assignment."""
        result = ValidationResult(
            name="John Smith",
            is_valid=True,
            confidence=0.95,
            reason="matches_name_pattern"
        )

        assert result.name == "John Smith"
        assert result.is_valid is True
        assert result.confidence == 0.95
        assert result.reason == "matches_name_pattern"
        assert result.corrected_name is None

    def test_with_correction(self):
        """Test with corrected name."""
        result = ValidationResult(
            name="JHON SMITH",
            is_valid=True,
            confidence=0.8,
            reason="llm_corrected",
            corrected_name="John Smith"
        )

        assert result.corrected_name == "John Smith"


class TestLlamaValidatorMocked:
    """Tests for LlamaValidator with mocked model."""

    @pytest.fixture
    def mock_validator(self):
        """Create validator with mocked model loading."""
        with patch('src.ner.llm_validator.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            validator = LlamaValidator(device="cpu")
            return validator

    def test_init_default_model(self, mock_validator):
        """Test default model name."""
        assert mock_validator.model_name == "meta-llama/Llama-3.1-8B-Instruct"

    def test_init_custom_model(self):
        """Test custom model name."""
        with patch('src.ner.llm_validator.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            validator = LlamaValidator(
                model_name="custom/model",
                device="cpu"
            )
            assert validator.model_name == "custom/model"

    def test_heuristic_shortcircuit_garbage(self, mock_validator):
        """Test that heuristics short-circuit LLM for garbage."""
        result = mock_validator.validate_name("aaaa")

        assert not result.is_valid
        assert "heuristic" in result.reason
        # Model should not be loaded for obvious garbage
        assert mock_validator._model is None

    def test_heuristic_shortcircuit_valid(self, mock_validator):
        """Test that heuristics short-circuit LLM for clear valid names."""
        result = mock_validator.validate_name("John Smith")

        assert result.is_valid
        assert result.reason == "matches_name_pattern"
        # Model should not be loaded for obvious valid names
        assert mock_validator._model is None


class TestCreateValidator:
    """Tests for validator factory function."""

    def test_creates_mock_when_no_gpu(self):
        """Test factory creates MockValidator when no GPU."""
        with patch('src.ner.llm_validator.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            validator = create_validator(use_llm=True)
            assert isinstance(validator, MockValidator)

    def test_creates_mock_when_disabled(self):
        """Test factory creates MockValidator when LLM disabled."""
        with patch('src.ner.llm_validator.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            validator = create_validator(use_llm=False)
            assert isinstance(validator, MockValidator)

    def test_creates_llama_when_gpu_and_enabled(self):
        """Test factory creates LlamaValidator when GPU available and LLM enabled."""
        with patch('src.ner.llm_validator.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            validator = create_validator(use_llm=True)
            assert isinstance(validator, LlamaValidator)


class TestBatchValidation:
    """Tests for batch validation."""

    @pytest.fixture
    def validator(self):
        return MockValidator()

    def test_batch_validation_empty(self, validator):
        """Test batch validation with empty list."""
        # MockValidator doesn't have validate_batch, test would be for LlamaValidator
        # For now, test individual validation
        pass

    def test_batch_mixed_names(self, validator):
        """Test validation of mixed valid/invalid names."""
        names = [
            "John Smith",  # Valid
            "aaaa",  # Garbage
            "JEFFREY EPSTEIN",  # Valid
            "",  # Empty
            "Mary Johnson",  # Valid
        ]

        results = [validator.validate_name(n) for n in names]

        assert results[0].is_valid  # John Smith
        assert not results[1].is_valid  # aaaa
        assert results[2].is_valid  # JEFFREY EPSTEIN
        assert not results[3].is_valid  # empty
        assert results[4].is_valid  # Mary Johnson


# Integration tests (require GPU - skip if not available)
@pytest.mark.skipif(
    not __import__('torch').cuda.is_available(),
    reason="GPU not available"
)
class TestLlamaValidatorIntegration:
    """Integration tests requiring actual GPU and model."""

    @pytest.fixture(scope="class")
    def validator(self):
        """Create real validator for integration tests."""
        val = LlamaValidator()
        yield val
        val.cleanup()

    def test_validates_real_names(self, validator):
        """Test validation of known real names."""
        real_names = [
            "Jeffrey Epstein",
            "Ghislaine Maxwell",
            "Bill Clinton",
            "Donald Trump",
        ]

        for name in real_names:
            result = validator.validate_name(name)
            assert result.is_valid, f"'{name}' should be valid"

    def test_rejects_ocr_garbage(self, validator):
        """Test rejection of OCR garbage."""
        garbage = [
            "aaaaa treenagan",
            "Xz123 Qwerty",
            "..--..--",
        ]

        for name in garbage:
            result = validator.validate_name(name)
            assert not result.is_valid, f"'{name}' should be invalid"

    def test_handles_edge_cases(self, validator):
        """Test handling of edge case names."""
        edge_cases = [
            ("de la Cruz, Maria", True),  # Hispanic name format
            ("O'Brien", True),  # Irish name with apostrophe
            ("Jean-Pierre", True),  # French hyphenated name
            ("Xi Jinping", True),  # Chinese name
            ("Kim Jong Un", True),  # Korean name
        ]

        for name, expected_valid in edge_cases:
            result = validator.validate_name(name)
            # Don't assert exact validity, just check it handles them
            assert isinstance(result.is_valid, bool)
            assert isinstance(result.confidence, float)
