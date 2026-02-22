"""Factory for creating LLM backends."""

from loguru import logger

from .base import LLMBackend


def create_backend(backend: str = "transformers",
                   model_name: str | None = None,
                   quantize_4bit: bool = True) -> LLMBackend:
    """Create an LLM backend instance.

    Args:
        backend: Backend type - "transformers" or "vllm".
        model_name: Model name/path. Defaults differ by backend:
            - transformers: Qwen/Qwen3-32B
            - vllm: Qwen/Qwen3-32B-AWQ
        quantize_4bit: Use 4-bit quantization (transformers only).

    Returns:
        LLMBackend instance.
    """
    if backend == "vllm":
        from .vllm_backend import VLLMBackend
        logger.info(f"Creating vLLM backend (model={model_name or VLLMBackend.DEFAULT_MODEL})")
        return VLLMBackend(model_name=model_name)
    elif backend == "transformers":
        from .transformers_backend import TransformersBackend
        logger.info(f"Creating transformers backend (model={model_name or TransformersBackend.DEFAULT_MODEL})")
        return TransformersBackend(model_name=model_name, quantize_4bit=quantize_4bit)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Must be 'transformers' or 'vllm'.")
