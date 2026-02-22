"""Transformers backend wrapping existing SharedModelManager logic.

Uses BitsAndBytes 4-bit quantization (NF4) for memory efficiency.
This is the default backend for backwards compatibility.
"""

import re

import torch
from loguru import logger

# Pattern to strip Qwen3 <think>...</think> blocks from output
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.DOTALL)

from .base import BatchStats, LLMBackend


class TransformersBackend(LLMBackend):
    """LLM backend using HuggingFace transformers + BitsAndBytes 4-bit."""

    DEFAULT_MODEL = "Qwen/Qwen3-32B"

    def __init__(self,
                 model_name: str | None = None,
                 device: str | None = None,
                 quantize_4bit: bool = True):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.quantize_4bit = quantize_4bit

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._batch_stats = BatchStats()

    def _validate_gpu(self):
        """Fail fast if GPU is not Tier 2 GPU-class for LLM workloads."""
        if not torch.cuda.is_available():
            return

        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)

        logger.info(f"GPU: {gpu_name}")
        logger.info(f"Compute capability: {capability[0]}.{capability[1]}")

        MIN_COMPUTE_CAPABILITY = 8.0
        if capability[0] + capability[1] / 10 < MIN_COMPUTE_CAPABILITY:
            raise RuntimeError(
                f"FATAL: GPU '{gpu_name}' (compute {capability[0]}.{capability[1]}) "
                f"is below minimum {MIN_COMPUTE_CAPABILITY} for LLM workloads. "
                f"Qwen 32B requires Tier 2 GPU/A100-class GPU. "
                f"Tier 1 GPU (compute 7.5) is only for XLM-R NER (--ner-only)."
            )

    def _load_model(self):
        """Load the model and tokenizer."""
        if self._loaded:
            return

        self._validate_gpu()

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading transformers backend: {self.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Requesting flash_attention_2")

        if self.quantize_4bit and self.device == "cuda":
            logger.info("Using 4-bit quantization")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        except (ImportError, ValueError) as e:
            if "flash" in str(e).lower():
                logger.warning(f"Flash attention unavailable ({e}), falling back to default")
                model_kwargs.pop("attn_implementation", None)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            else:
                raise

        self._loaded = True
        logger.info("Transformers backend loaded")

        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Compute capability: {torch.cuda.get_device_capability(0)}")
            logger.info(f"Attention implementation: {getattr(self._model.config, '_attn_implementation', 'unknown')}")
            logger.info(f"Quantization: {'4-bit' if self.quantize_4bit else 'none'}")

            try:
                import flash_attn
                logger.info(f"flash_attn: v{flash_attn.__version__}")
            except ImportError:
                logger.warning("flash_attn: NOT INSTALLED")

    def _apply_chat_template(self, messages: list[dict]) -> str:
        """Apply chat template with thinking disabled."""
        kwargs = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        # Qwen3 supports enable_thinking to suppress <think> blocks
        try:
            return self._tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            # Older models (Qwen2.5) don't have enable_thinking
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def generate(self, messages: list[dict], max_new_tokens: int = 2048,
                 temperature: float = 0.0, do_sample: bool = False) -> str:
        if not self._loaded:
            self._load_model()

        text = self._apply_chat_template(messages)

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        }
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        response = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return _THINK_RE.sub("", response).strip()

    def generate_batch(self, messages_list: list[list[dict]], max_new_tokens: int = 2048,
                       temperature: float = 0.0, do_sample: bool = False) -> list[str]:
        if not messages_list:
            return []

        if len(messages_list) == 1:
            return [self.generate(messages_list[0], max_new_tokens=max_new_tokens,
                                  temperature=temperature, do_sample=do_sample)]

        if not self._loaded:
            self._load_model()

        texts = [self._apply_chat_template(msgs) for msgs in messages_list]

        return self._batched_generate_texts(texts, max_new_tokens, temperature, do_sample)

    def generate_raw(self, prompt: str, max_new_tokens: int = 500,
                     temperature: float = 0.1, do_sample: bool = True) -> str:
        if not self._loaded:
            self._load_model()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        }
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        response = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return _THINK_RE.sub("", response).strip()

    def generate_raw_batch(self, prompts: list[str], max_new_tokens: int = 500,
                           temperature: float = 0.1, do_sample: bool = True) -> list[str]:
        if not prompts:
            return []

        if len(prompts) == 1:
            return [self.generate_raw(prompts[0], max_new_tokens=max_new_tokens,
                                      temperature=temperature, do_sample=do_sample)]

        if not self._loaded:
            self._load_model()

        return self._batched_generate_texts(prompts, max_new_tokens, temperature, do_sample)

    def _batched_generate_texts(self, texts: list[str], max_new_tokens: int,
                                temperature: float, do_sample: bool) -> list[str]:
        """Shared batched generation for both chat and raw prompts."""
        original_padding_side = self._tokenizer.padding_side
        original_pad_token_id = self._tokenizer.pad_token_id
        try:
            self._tokenizer.padding_side = 'left'
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            ).to(self._model.device)

            input_lengths = inputs.attention_mask.sum(dim=1).tolist()
            padded_input_len = inputs.input_ids.shape[1]
            total_tokens = inputs.input_ids.numel()
            non_pad_tokens = sum(input_lengths)
            self._batch_stats.padding_waste_tokens += total_tokens - non_pad_tokens
            self._batch_stats.total_batches += 1
            self._batch_stats.total_prompts += len(texts)

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self._tokenizer.pad_token_id,
            }
            if do_sample and temperature > 0:
                gen_kwargs["temperature"] = temperature

            try:
                with torch.no_grad():
                    outputs = self._model.generate(**inputs, **gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                self._batch_stats.oom_splits += 1
                logger.warning(
                    f"OOM with batch size {len(texts)}, splitting in half"
                )
                torch.cuda.empty_cache()
                mid = len(texts) // 2
                left = self._batched_generate_texts(texts[:mid], max_new_tokens, temperature, do_sample)
                right = self._batched_generate_texts(texts[mid:], max_new_tokens, temperature, do_sample)
                return left + right

            assert outputs.shape[1] >= padded_input_len, (
                f"generate() output shorter than input: "
                f"{outputs.shape[1]} < {padded_input_len}"
            )
            results = []
            for i in range(len(texts)):
                new_tokens = outputs[i][padded_input_len:]
                response = self._tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                )
                results.append(_THINK_RE.sub("", response).strip())

            return results

        finally:
            self._tokenizer.padding_side = original_padding_side
            if original_pad_token_id is not None:
                self._tokenizer.pad_token_id = original_pad_token_id

    @property
    def tokenizer(self):
        if not self._loaded:
            self._load_model()
        return self._tokenizer

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def batch_stats(self) -> BatchStats:
        return self._batch_stats

    def cleanup(self):
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Transformers backend unloaded")

    def __del__(self):
        self.cleanup()
