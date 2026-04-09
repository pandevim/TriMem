"""
Unified LLM inference wrapper.

Supports two backends:
  - "vllm"         : fast GPU inference via vLLM (recommended for Colab / GPU)
  - "transformers"  : universal HuggingFace fallback (works with any model)

To swap models, change MODEL_NAME in configs/settings.py to any HuggingFace
model ID (e.g. "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3").
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from configs.settings import (
    MODEL_NAME,
    INFERENCE_BACKEND,
    MAX_TOKENS,
    AGENT_TEMPERATURE,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
)


@dataclass
class LLMResponse:
    """Standardised response from any backend."""
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Backend: vLLM
# ---------------------------------------------------------------------------
class VLLMBackend:
    def __init__(self):
        from vllm import LLM
        print(f"[LLM] Initializing vLLM engine (model={MODEL_NAME}, "
              f"gpu_mem={GPU_MEMORY_UTILIZATION}, max_len={MAX_MODEL_LEN}) …",
              flush=True)
        t0 = time.time()
        self.llm = LLM(
            model=MODEL_NAME,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
        )
        print(f"[LLM] vLLM engine ready in {time.time() - t0:.1f}s", flush=True)

    def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = MAX_TOKENS,
        temperature: float = AGENT_TEMPERATURE,
    ) -> LLMResponse:
        from vllm import SamplingParams

        full_messages = [{"role": "system", "content": system}] + messages
        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        n_msgs = len(full_messages)
        print(f"[LLM] Generating (vLLM, {n_msgs} messages) …", end=" ", flush=True)
        t0 = time.time()
        outputs = self.llm.chat(full_messages, sampling_params=sampling)
        latency = (time.time() - t0) * 1000

        out = outputs[0]
        tokens_in = len(out.prompt_token_ids)
        tokens_out = len(out.outputs[0].token_ids)
        raw_text = out.outputs[0].text
        # vLLM may strip <think>/</think> via skip_special_tokens.
        # Decode with token IDs to preserve </think> for parse_action.
        if "</think>" not in raw_text:
            tokenizer = self.llm.get_tokenizer()
            raw_text = tokenizer.decode(
                out.outputs[0].token_ids, skip_special_tokens=False
            )
        print(f"done in {latency:.0f}ms "
              f"(in={tokens_in}, out={tokens_out})", flush=True)
        return LLMResponse(
            text=raw_text.strip(),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# Backend: HuggingFace transformers
# ---------------------------------------------------------------------------
class TransformersBackend:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[LLM] Loading tokenizer for {MODEL_NAME} …", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        print(f"[LLM] Loading model weights (dtype=bfloat16, device_map=auto) …",
              flush=True)
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[LLM] Model loaded in {time.time() - t0:.1f}s", flush=True)

    def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = MAX_TOKENS,
        temperature: float = AGENT_TEMPERATURE,
    ) -> LLMResponse:
        import torch

        full_messages = [{"role": "system", "content": system}] + messages

        prompt_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        tokens_in = inputs["input_ids"].shape[1]

        n_msgs = len(full_messages)
        print(f"[LLM] Generating (transformers, {n_msgs} messages, "
              f"{tokens_in} prompt tokens) …", end=" ", flush=True)
        t0 = time.time()
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        latency = (time.time() - t0) * 1000

        new_tokens = generated[0][tokens_in:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        tokens_out = len(new_tokens)

        print(f"done in {latency:.0f}ms "
              f"(in={tokens_in}, out={tokens_out})", flush=True)
        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# Singleton accessor — model loads once, reused across all agents
# ---------------------------------------------------------------------------
_backend_instance = None


def get_llm():
    """Return the shared LLM backend (lazy-loaded singleton)."""
    global _backend_instance
    if _backend_instance is not None:
        return _backend_instance

    if INFERENCE_BACKEND == "vllm":
        print(f"[LLM] Loading {MODEL_NAME} via vLLM …", flush=True)
        _backend_instance = VLLMBackend()
    elif INFERENCE_BACKEND == "transformers":
        print(f"[LLM] Loading {MODEL_NAME} via transformers …", flush=True)
        _backend_instance = TransformersBackend()
    else:
        raise ValueError(
            f"Unknown INFERENCE_BACKEND={INFERENCE_BACKEND!r}. "
            "Use 'vllm' or 'transformers'."
        )

    print(f"[LLM] Ready.", flush=True)
    return _backend_instance
