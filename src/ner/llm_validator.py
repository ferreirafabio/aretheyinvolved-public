"""LLM-based name validation to filter OCR garbage.

Uses a local LLM (Llama 3.1 or similar) to validate whether extracted
text represents a real person name or OCR artifact.

This is crucial for cleaning up names extracted from scanned documents
where OCR errors produce gibberish like "aaaaa treenagan".
"""

import re
from dataclasses import dataclass
from typing import Protocol

import torch
from loguru import logger


@dataclass
class ValidationResult:
    """Result of name validation."""
    name: str
    is_valid: bool
    confidence: float
    reason: str | None = None
    corrected_name: str | None = None


class LLMBackend(Protocol):
    """Protocol for LLM backends."""
    def validate_name(self, name: str, context: str | None = None) -> ValidationResult:
        """Validate if text is a real person name."""
        ...


class HeuristicValidator:
    """Fast heuristic-based validator for obvious cases.

    This runs first before LLM to quickly filter obvious garbage
    without needing expensive LLM calls.
    """

    # Common name patterns
    NAME_PATTERN = re.compile(
        r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$"  # Standard capitalization
        r"|^[A-Z]+(?:\s+[A-Z]+)*$"  # ALL CAPS
        r"|^[A-Z]\.\s*[A-Z][a-z]+$"  # J. Smith
        r"|^[A-Z][a-z]+,\s*[A-Z][a-z]+$"  # Smith, John
    )

    # Garbage patterns
    GARBAGE_PATTERNS = [
        r"^[^a-zA-Z]*$",  # No letters
        r"(.)\1{3,}",  # 4+ repeated characters
        r"^.{1,2}$",  # Too short
        r"^.{60,}$",  # Too long for a name
        r"[\d]{3,}",  # 3+ consecutive digits
        r"[^\w\s\-\.\',]",  # Unusual characters
        r"^[\s\.\-\',]+$",  # Only punctuation/whitespace
    ]

    def __init__(self):
        self._garbage_regex = [re.compile(p) for p in self.GARBAGE_PATTERNS]

    def is_obvious_garbage(self, name: str) -> tuple[bool, str | None]:
        """Check if name is obviously not valid.

        Returns:
            Tuple of (is_garbage, reason).
        """
        name = name.strip()

        if not name:
            return True, "empty"

        for i, pattern in enumerate(self._garbage_regex):
            if pattern.search(name):
                return True, f"matches_garbage_pattern_{i}"

        return False, None

    def is_likely_valid(self, name: str) -> bool:
        """Check if name matches common name patterns."""
        return bool(self.NAME_PATTERN.match(name.strip()))


class LlamaValidator(LLMBackend):
    """Validate names using Llama 3.1 model.

    Uses the model in classification mode for fast validation.
    """

    DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    PROMPT_TEMPLATE = """You are validating whether text extracted from a document is a real person's name.

Text: "{name}"
{context_line}

Is this a REAL person's name? You must reject:
- OCR garbage (random characters, symbols, fragments)
- Timestamps or dates (00:07:46, March 15)
- Partial words or fragments ("& Annap", "se '")
- Company names (AT&T, IBM)
- Single letters or very short text
- Numbers or alphanumeric codes

Real names:
- Follow standard naming conventions (First Last, LAST FIRST, etc.)
- May have titles (Dr. Smith, Mr. Jones)
- May have OCR errors but are recognizable (JOHN 5MITH = John Smith)

Answer with exactly one word: YES or NO"""

    def __init__(self,
                 model_name: str | None = None,
                 device: str | None = None,
                 quantize: bool = True,
                 use_heuristics: bool = False):
        """Initialize Llama validator.

        Args:
            model_name: HuggingFace model name.
            device: Device to use ('cuda', 'cpu', or None for auto).
            quantize: Use 4-bit quantization for memory efficiency.
            use_heuristics: Use heuristic pre-filtering (default: False, let LLM decide).
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.quantize = quantize
        self.use_heuristics = use_heuristics

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None
        self._tokenizer = None
        self._heuristic = HeuristicValidator() if use_heuristics else None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            logger.info(f"Loading LLM: {self.model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if self.quantize and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                if self.device == "cpu":
                    self._model = self._model.to(self.device)

            self._model.eval()
            logger.info("LLM loaded successfully")

    def validate_name(self, name: str, context: str | None = None) -> ValidationResult:
        """Validate if text is a real person name.

        Args:
            name: Text to validate.
            context: Optional surrounding text for context.

        Returns:
            ValidationResult with validation details.
        """
        # Optionally check heuristics first (disabled by default)
        if self._heuristic:
            is_garbage, reason = self._heuristic.is_obvious_garbage(name)
            if is_garbage:
                return ValidationResult(
                    name=name,
                    is_valid=False,
                    confidence=0.95,
                    reason=f"heuristic_{reason}"
                )

            # If it looks like a standard name, probably valid
            if self._heuristic.is_likely_valid(name):
                return ValidationResult(
                    name=name,
                    is_valid=True,
                    confidence=0.85,
                    reason="matches_name_pattern"
                )

        # Use LLM for all validation (or uncertain cases if heuristics enabled)
        self._load_model()

        context_line = f"Context: ...{context[:100]}..." if context else ""
        prompt = self.PROMPT_TEMPLATE.format(name=name, context_line=context_line)

        messages = [{"role": "user", "content": prompt}]

        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id
            )

        response = self._tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        ).strip().upper()

        is_valid = response.startswith("YES")

        return ValidationResult(
            name=name,
            is_valid=is_valid,
            confidence=0.75,  # LLM validation has moderate confidence
            reason="llm_validation"
        )

    def validate_batch(self,
                       names: list[str],
                       contexts: list[str | None] | None = None) -> list[ValidationResult]:
        """Validate multiple names efficiently.

        Args:
            names: List of names to validate.
            contexts: Optional list of contexts (parallel to names).

        Returns:
            List of ValidationResult objects.
        """
        if contexts is None:
            contexts = [None] * len(names)

        results = []
        llm_batch = []  # Names that need LLM validation
        llm_indices = []  # Original indices

        # First pass: heuristics
        for i, (name, ctx) in enumerate(zip(names, contexts)):
            is_garbage, reason = self._heuristic.is_obvious_garbage(name)
            if is_garbage:
                results.append(ValidationResult(
                    name=name,
                    is_valid=False,
                    confidence=0.95,
                    reason=f"heuristic_{reason}"
                ))
            elif self._heuristic.is_likely_valid(name):
                results.append(ValidationResult(
                    name=name,
                    is_valid=True,
                    confidence=0.85,
                    reason="matches_name_pattern"
                ))
            else:
                results.append(None)  # Placeholder
                llm_batch.append((name, ctx))
                llm_indices.append(i)

        # Second pass: LLM for uncertain cases
        if llm_batch:
            logger.info(f"Validating {len(llm_batch)} uncertain names with LLM")
            for (name, ctx), idx in zip(llm_batch, llm_indices):
                result = self.validate_name(name, ctx)
                results[idx] = result

        return results

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

        logger.info("LLM unloaded")


class MockValidator(LLMBackend):
    """Mock validator for testing without GPU."""

    def __init__(self):
        self._heuristic = HeuristicValidator()

    def validate_name(self, name: str, context: str | None = None) -> ValidationResult:
        """Validate using only heuristics."""
        is_garbage, reason = self._heuristic.is_obvious_garbage(name)
        if is_garbage:
            return ValidationResult(
                name=name,
                is_valid=False,
                confidence=0.95,
                reason=f"heuristic_{reason}"
            )

        if self._heuristic.is_likely_valid(name):
            return ValidationResult(
                name=name,
                is_valid=True,
                confidence=0.85,
                reason="matches_name_pattern"
            )

        # For uncertain cases, default to valid in mock
        return ValidationResult(
            name=name,
            is_valid=True,
            confidence=0.5,
            reason="mock_default"
        )


def create_validator(use_llm: bool = True, **kwargs) -> LLMBackend:
    """Factory function to create appropriate validator.

    Args:
        use_llm: Whether to use LLM (requires GPU).
        **kwargs: Arguments to pass to validator.

    Returns:
        Validator instance.
    """
    if use_llm and torch.cuda.is_available():
        return LlamaValidator(**kwargs)
    else:
        logger.warning("Using mock validator (no GPU or use_llm=False)")
        return MockValidator()


if __name__ == "__main__":
    # Test the validator
    test_names = [
        "John Smith",
        "JEFFREY EPSTEIN",
        "J. Maxwell",
        "aaaaa treenagan",  # OCR garbage
        "Xz123 Qwerty",  # Garbage
        "Mr.",  # Incomplete
        "",  # Empty
        "Smith, John",
        "de la Cruz, Maria",
        "O'Brien",
        "Jean-Pierre",
        "1234",  # Numbers
        "...---...",  # Punctuation
    ]

    validator = create_validator(use_llm=False)  # Use mock for quick test

    print("Name Validation Results:")
    print("=" * 60)

    for name in test_names:
        result = validator.validate_name(name)
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        print(f"{status:12} | {name:25} | {result.reason}")
