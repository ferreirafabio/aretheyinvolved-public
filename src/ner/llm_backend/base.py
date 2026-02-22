"""Abstract base class for LLM backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class BatchStats:
    """Metrics for batched generation calls."""
    total_batches: int = 0
    total_prompts: int = 0
    oom_splits: int = 0
    padding_waste_tokens: int = 0


class LLMBackend(ABC):
    """Abstract interface for LLM generation backends.

    Two families of methods:
    - generate/generate_batch: Chat messages (system+user roles), used by classifier and repair
    - generate_raw/generate_raw_batch: Raw string prompts (no chat template), used by recovery
    """

    @abstractmethod
    def generate(self, messages: list[dict], max_new_tokens: int = 2048,
                 temperature: float = 0.0, do_sample: bool = False) -> str:
        """Generate from chat messages (single prompt).

        Args:
            messages: Chat messages in OpenAI format [{role, content}, ...].
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            do_sample: Whether to use sampling.

        Returns:
            Generated text response.
        """
        ...

    @abstractmethod
    def generate_batch(self, messages_list: list[list[dict]], max_new_tokens: int = 2048,
                       temperature: float = 0.0, do_sample: bool = False) -> list[str]:
        """Generate from multiple chat message sets (batched).

        Args:
            messages_list: List of chat message lists.
            max_new_tokens: Maximum tokens to generate per response.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.

        Returns:
            List of generated text responses (positional correspondence).
        """
        ...

    @abstractmethod
    def generate_raw(self, prompt: str, max_new_tokens: int = 500,
                     temperature: float = 0.1, do_sample: bool = True) -> str:
        """Generate from a raw string prompt (no chat template).

        Used by recovery stage which builds its own prompt format.

        Args:
            prompt: Raw text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.

        Returns:
            Generated text response.
        """
        ...

    @abstractmethod
    def generate_raw_batch(self, prompts: list[str], max_new_tokens: int = 500,
                           temperature: float = 0.1, do_sample: bool = True) -> list[str]:
        """Generate from multiple raw string prompts (batched).

        Args:
            prompts: List of raw text prompts.
            max_new_tokens: Maximum tokens to generate per response.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.

        Returns:
            List of generated text responses (positional correspondence).
        """
        ...

    @property
    @abstractmethod
    def tokenizer(self):
        """Get the tokenizer (for token counting/estimation)."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the backend model is loaded."""
        ...

    @property
    @abstractmethod
    def batch_stats(self) -> BatchStats:
        """Get cumulative batch statistics."""
        ...

    @abstractmethod
    def cleanup(self):
        """Release GPU memory and resources."""
        ...
