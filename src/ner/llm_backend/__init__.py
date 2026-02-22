"""LLM backend abstraction for NER pipeline.

Supports two backends:
- transformers: BitsAndBytes 4-bit quantization (default, backwards compatible)
- vllm: AWQ+Marlin quantization via vLLM offline API (faster, better GPU utilization)
"""

from .base import LLMBackend, BatchStats
from .factory import create_backend

__all__ = ["LLMBackend", "BatchStats", "create_backend"]
