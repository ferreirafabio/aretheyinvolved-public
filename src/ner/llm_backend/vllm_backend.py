"""vLLM backend using offline LLM API with AWQ+Marlin quantization.

Uses continuous batching and PagedAttention for better GPU utilization.
No left-padding waste, no OOM fallback needed (vLLM manages KV cache).
"""

import re

import torch
from loguru import logger

# Pattern to strip Qwen3 <think>...</think> blocks from raw output
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.DOTALL)

from .base import BatchStats, LLMBackend


class VLLMBackend(LLMBackend):
    """LLM backend using vLLM offline API with AWQ quantization."""

    DEFAULT_MODEL = "Qwen/Qwen3-32B-AWQ"

    def __init__(self,
                 model_name: str | None = None,
                 max_model_len: int = 32768,
                 gpu_memory_utilization: float = 0.90):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

        self._llm = None
        self._tokenizer_obj = None
        self._loaded = False
        self._batch_stats = BatchStats()

    def _validate_gpu(self):
        """Fail fast if GPU is not Tier 2 GPU-class for LLM workloads."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FATAL: vLLM backend requires CUDA GPU. No GPU detected."
            )

        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)

        logger.info(f"GPU: {gpu_name}")
        logger.info(f"Compute capability: {capability[0]}.{capability[1]}")

        MIN_COMPUTE_CAPABILITY = 8.0
        if capability[0] + capability[1] / 10 < MIN_COMPUTE_CAPABILITY:
            raise RuntimeError(
                f"FATAL: GPU '{gpu_name}' (compute {capability[0]}.{capability[1]}) "
                f"is below minimum {MIN_COMPUTE_CAPABILITY} for vLLM workloads. "
                f"vLLM + AWQ requires Tier 2 GPU/A100-class GPU. "
                f"Tier 1 GPU (compute 7.5) is only for XLM-R NER (--ner-only)."
            )

    def _load_model(self):
        """Load the vLLM model."""
        if self._loaded:
            return

        self._validate_gpu()

        try:
            from vllm import LLM
        except ImportError:
            raise RuntimeError(
                "vLLM is not installed. Install with: pip install vllm"
            )

        logger.info(f"Loading vLLM backend: {self.model_name}")
        logger.info(f"max_model_len={self.max_model_len}, "
                     f"gpu_memory_utilization={self.gpu_memory_utilization}")

        # Determine quantization from model name
        quantization = None
        model_lower = self.model_name.lower()
        if "awq" in model_lower:
            quantization = "awq_marlin"
            logger.info("Using AWQ+Marlin quantization")

        self._llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            quantization=quantization,
            trust_remote_code=True,
        )

        self._tokenizer_obj = self._llm.get_tokenizer()
        self._loaded = True

        logger.info("vLLM backend loaded")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    def _make_sampling_params(self, max_new_tokens: int, temperature: float,
                              do_sample: bool):
        """Create vLLM SamplingParams."""
        from vllm import SamplingParams

        if not do_sample or temperature == 0.0:
            return SamplingParams(
                temperature=0,
                max_tokens=max_new_tokens,
            )
        else:
            return SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=1.0,
            )

    def _apply_chat_template(self, messages: list[dict]) -> str:
        """Apply chat template to messages with thinking disabled."""
        kwargs = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        # Qwen3 supports enable_thinking to suppress <think> blocks
        try:
            return self._tokenizer_obj.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            # Older models (Qwen2.5) don't have enable_thinking
            return self._tokenizer_obj.apply_chat_template(
                messages, **kwargs
            )

    def generate(self, messages: list[dict], max_new_tokens: int = 2048,
                 temperature: float = 0.0, do_sample: bool = False) -> str:
        if not self._loaded:
            self._load_model()

        prompt = self._apply_chat_template(messages)
        params = self._make_sampling_params(max_new_tokens, temperature, do_sample)

        self._batch_stats.total_batches += 1
        self._batch_stats.total_prompts += 1

        outputs = self._llm.generate([prompt], params)
        return self._strip_think_blocks(outputs[0].outputs[0].text)

    def generate_batch(self, messages_list: list[list[dict]], max_new_tokens: int = 2048,
                       temperature: float = 0.0, do_sample: bool = False) -> list[str]:
        if not messages_list:
            return []

        if not self._loaded:
            self._load_model()

        prompts = [self._apply_chat_template(msgs) for msgs in messages_list]
        params = self._make_sampling_params(max_new_tokens, temperature, do_sample)

        self._batch_stats.total_batches += 1
        self._batch_stats.total_prompts += len(prompts)

        outputs = self._llm.generate(prompts, params)
        return [self._strip_think_blocks(out.outputs[0].text) for out in outputs]

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        """Strip Qwen3 <think>...</think> blocks from raw output."""
        return _THINK_RE.sub("", text).strip()

    def generate_raw(self, prompt: str, max_new_tokens: int = 500,
                     temperature: float = 0.1, do_sample: bool = True) -> str:
        if not self._loaded:
            self._load_model()

        params = self._make_sampling_params(max_new_tokens, temperature, do_sample)

        self._batch_stats.total_batches += 1
        self._batch_stats.total_prompts += 1

        outputs = self._llm.generate([prompt], params)
        return self._strip_think_blocks(outputs[0].outputs[0].text)

    def generate_raw_batch(self, prompts: list[str], max_new_tokens: int = 500,
                           temperature: float = 0.1, do_sample: bool = True) -> list[str]:
        if not prompts:
            return []

        if not self._loaded:
            self._load_model()

        params = self._make_sampling_params(max_new_tokens, temperature, do_sample)

        self._batch_stats.total_batches += 1
        self._batch_stats.total_prompts += len(prompts)

        outputs = self._llm.generate(prompts, params)
        return [self._strip_think_blocks(out.outputs[0].text) for out in outputs]

    @property
    def tokenizer(self):
        if not self._loaded:
            self._load_model()
        return self._tokenizer_obj

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def batch_stats(self) -> BatchStats:
        return self._batch_stats

    def cleanup(self):
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._tokenizer_obj = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("vLLM backend unloaded")

    def __del__(self):
        self.cleanup()
