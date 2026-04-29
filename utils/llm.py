"""
Unified LLM inference wrapper.

Supports two backends:
  - "vllm"         : fast GPU inference via vLLM (recommended for Colab / GPU)
  - "transformers"  : universal HuggingFace fallback (works with any model)

To swap models, change MODEL_NAME in configs/settings.py to any HuggingFace
model ID (e.g. "meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3").
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from configs.settings import (
    MODEL_NAME,
    INFERENCE_BACKEND,
    MAX_TOKENS,
    AGENT_TEMPERATURE,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    ENABLE_PREFIX_CACHING,
)


@dataclass
class LLMResponse:
    """Standardised response from any backend.

    ``first_token_entropy`` is the Shannon entropy (bits) of the first
    generated token's distribution, computed from the top-K logprobs the
    backend returns. -1.0 means the backend did not emit logprobs for
    this call (caller passed ``logprobs=0`` or the backend doesn't
    support it).
    """
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    first_token_entropy: float = -1.0


# ---------------------------------------------------------------------------
# Entropy helpers — backend-specific because each library returns logprobs
# in a different shape. Both produce Shannon entropy (bits) of the first
# generated token's top-K distribution, with the residual mass folded into
# a single "other" bucket so the distribution sums to 1.
# ---------------------------------------------------------------------------
def _entropy_from_vllm_logprobs(step_logprobs) -> float:
    """vLLM returns logprobs as a list (per generated token) of dicts
    mapping token_id -> Logprob(logprob=float, ...). We use only step 0.
    """
    if not step_logprobs:
        return -1.0
    first = step_logprobs[0]
    if not first:
        return -1.0
    # vLLM Logprob objects have a .logprob attr; older versions used floats.
    probs = []
    for entry in first.values():
        lp = getattr(entry, "logprob", entry)
        probs.append(math.exp(float(lp)))
    return _shannon_with_residual(probs)


def _entropy_from_transformers_scores(step_logits, top_k: int) -> float:
    """transformers `output_scores` are pre-softmax logits per step.
    We take the top-K, softmax over all logits to get the true tail mass,
    then add a residual bucket so the entropy reflects the full distribution.
    """
    import torch

    if step_logits is None:
        return -1.0
    logits = step_logits[0]  # batch index 0
    probs_full = torch.softmax(logits.float(), dim=-1)
    k = max(1, top_k)
    top = torch.topk(probs_full, k=min(k, probs_full.shape[-1]))
    return _shannon_with_residual([float(p) for p in top.values.tolist()])


def _shannon_with_residual(probs: list[float]) -> float:
    """Shannon entropy in bits over a top-K distribution, with leftover
    probability mass folded into a single residual bucket."""
    probs = [p for p in probs if p > 0.0]
    if not probs:
        return -1.0
    total = sum(probs)
    residual = max(0.0, 1.0 - total)
    h = 0.0
    for p in probs:
        h -= p * math.log2(p)
    if residual > 0.0:
        h -= residual * math.log2(residual)
    return round(h, 4)


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
            enable_prefix_caching=ENABLE_PREFIX_CACHING,
        )
        print(f"[LLM] vLLM engine ready in {time.time() - t0:.1f}s", flush=True)

    def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = MAX_TOKENS,
        temperature: float = AGENT_TEMPERATURE,
        logprobs: int = 0,
    ) -> LLMResponse:
        from vllm import SamplingParams

        full_messages = [{"role": "system", "content": system}] + messages
        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs if logprobs > 0 else None,
        )
        n_msgs = len(full_messages)
        print(f"[LLM] Generating (vLLM, {n_msgs} messages) …", end=" ", flush=True)
        t0 = time.time()
        outputs = self.llm.chat(full_messages, sampling_params=sampling)
        latency = (time.time() - t0) * 1000

        out = outputs[0]
        tokens_in = len(out.prompt_token_ids)
        tokens_out = len(out.outputs[0].token_ids)

        first_token_entropy = -1.0
        if logprobs > 0:
            first_token_entropy = _entropy_from_vllm_logprobs(out.outputs[0].logprobs)

        print(f"done in {latency:.0f}ms "
              f"(in={tokens_in}, out={tokens_out}, H={first_token_entropy:.3f})",
              flush=True)
        return LLMResponse(
            text=out.outputs[0].text.strip(),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency,
            first_token_entropy=first_token_entropy,
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
        logprobs: int = 0,
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
                output_scores=logprobs > 0,
                return_dict_in_generate=logprobs > 0,
            )
        latency = (time.time() - t0) * 1000

        if logprobs > 0:
            sequences = generated.sequences
            scores = generated.scores  # tuple of (logits[batch, vocab]) per step
        else:
            sequences = generated
            scores = None

        new_tokens = sequences[0][tokens_in:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        tokens_out = len(new_tokens)

        first_token_entropy = -1.0
        if scores is not None and len(scores) > 0:
            first_token_entropy = _entropy_from_transformers_scores(
                scores[0], top_k=logprobs
            )

        print(f"done in {latency:.0f}ms "
              f"(in={tokens_in}, out={tokens_out}, H={first_token_entropy:.3f})",
              flush=True)
        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency,
            first_token_entropy=first_token_entropy,
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
        import torch
        if not torch.cuda.is_available():
            print("[LLM] WARNING: CUDA not available, falling back to transformers backend.", flush=True)
            _backend_instance = TransformersBackend()
            print(f"[LLM] Ready.", flush=True)
            return _backend_instance
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
