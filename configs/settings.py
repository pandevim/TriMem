"""Tri-Mem configuration."""

# --- LLM ---
# Change MODEL_NAME to any HuggingFace model ID to swap models.
MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
INFERENCE_BACKEND = "vllm"  # "vllm" (fast, GPU) or "transformers" (universal fallback)
MAX_TOKENS = 1024
# vLLM-specific settings
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 16384
# Max concurrent sequences vLLM will schedule. We run one task at a time so a
# small number is fine; 256 (vLLM's default) is way too many for Mamba/MoE
# models like Qwen3.5-35B-A3B where each decode seq needs a dedicated Mamba
# cache block, and the engine refuses to start if we don't have enough blocks.
VLLM_MAX_NUM_SEQS = 16

# --- Agent ---
MAX_AGENT_TURNS = 60
AGENT_TEMPERATURE = 0.2

# --- Benchmark ---
DEFAULT_NUM_TASKS = 10

# --- RAG (Phase 2) ---
RAG_COLLECTION = "tri_mem_facts"
RAG_TOP_K = 3

# --- MSA (Phase 3.75, simulated rulebook layer) ---
# Static policy corpus. The system prompt must be byte-identical across
# turns so vLLM's prefix cache holds the rulebook KV once per process.
import os as _os
MSA_RULEBOOK_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "memory", "rulebook", "novacorp_it_policy.md",
)
MSA_COLLECTION = "tri_mem_msa_rulebook"
MSA_TOP_K = 2              # routed sections injected per turn
MSA_INJECT_FULL_RULEBOOK = True  # path 1: freeze full rulebook in system prompt
MSA_INJECT_ROUTED_CHUNKS = True  # path 2: inject Top-k sections into user turn
ENABLE_PREFIX_CACHING = True     # vLLM automatic prefix cache

# --- Visual Bus (Phase 3) ---
OCR_MODEL_NAME = "zai-org/GLM-OCR"
VISUAL_BUS_HISTORY_DIR = "/tmp/trimem_visual_bus"
VISUAL_BUS_IMAGE_WIDTH = 1200     # pixel width of rendered history image
VISUAL_BUS_FONT_SIZE = 14         # pt — monospace font size in rendered image
MAX_VISUAL_TILES = 20             # max turns to keep in the visual history image

# --- Entropy Router (Phase 4) ---
# Thresholds are measured in bits of Shannon entropy on the first
# generated token of a single-token probe pass with no memory injected.
# Distribution is top-K + residual bucket (see utils/llm._shannon_with_residual).
ENTROPY_LOW_THRESHOLD = 0.3   # below: MSA only (frozen rulebook)
ENTROPY_MED_THRESHOLD = 0.7   # below: + Visual Bus; above: + RAG
# Top-K considered when computing probe entropy. 20 is enough to capture
# >99% of probability mass for a well-trained model on a structured task.
ROUTER_PROBE_TOP_K = 20
# Whether to enable the entropy router for the TriMemAgent. Flipping this
# off makes the agent always inject all three memory layers — useful for
# ablations.
ROUTER_ENABLED = True

# --- Frontend ---
FRONTEND_PORT = 5000
