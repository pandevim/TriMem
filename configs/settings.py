"""Tri-Mem configuration."""

# --- LLM ---
# Change MODEL_NAME to any HuggingFace model ID to swap models.
MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
INFERENCE_BACKEND = "vllm"  # "vllm" (fast, GPU) or "transformers" (universal fallback)
MAX_TOKENS = 2048
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
# Qwen3.x thinking-mode toggle, set per call site. Empirically:
#   - Probe pass: OFF — entropy reading must be on the answer-token
#     distribution, not on a <think> opening token.
#   - Answer pass: ON — reasoning gives ~+30pp on temporal questions on
#     the LongMemEval slice; the LLM judge already tolerates the verbose
#     preamble so we don't pay a grading penalty.
#   - Judge pass: OFF — single yes/no verdict; reasoning is wasteful.
ENABLE_THINKING_PROBE = False
ENABLE_THINKING_ANSWER = True
ENABLE_THINKING_JUDGE = False

# --- LLM judge (LongMemEval scoring) ---
# Temperature for the judge call. Keep at 0 for determinism — the judge
# emits a single yes/no token.
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 8

# --- Benchmark ---
DEFAULT_NUM_TASKS = 10

# --- RAG (Phase 2) ---
RAG_COLLECTION = "tri_mem_facts"
RAG_TOP_K = 5

# --- MSA (Phase 3.75, simulated rulebook layer) ---
# Static policy corpus. The system prompt must be byte-identical across
# turns so vLLM's prefix cache holds the rulebook KV once per process.
import os as _os
MSA_RULEBOOK_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "memory", "rulebook", "novacorp_it_policy.md",
)
MSA_COLLECTION = "tri_mem_msa_rulebook"
MSA_TOP_K = 4              # routed sections injected per turn
MSA_INJECT_FULL_RULEBOOK = True  # path 1: freeze full rulebook in system prompt
MSA_INJECT_ROUTED_CHUNKS = True  # path 2: inject Top-k sections into user turn
ENABLE_PREFIX_CACHING = True     # vLLM automatic prefix cache

# --- Visual Bus (Phase 3) ---
OCR_MODEL_NAME = "zai-org/GLM-OCR"
VISUAL_BUS_HISTORY_DIR = "/tmp/trimem_visual_bus"
VISUAL_BUS_IMAGE_WIDTH = 1200     # pixel width of rendered history image
VISUAL_BUS_FONT_SIZE = 14         # pt — monospace font size in rendered image
# Bumped 4 → 16 once OCR moved to its own GPU (cuda:1, OCR_DEVICE below).
# At 4 we lost most session context on the s_cleaned split; at 16 we cover
# oracle (2–3 sessions/task) with margin and capture the most recent 16
# sessions on s_cleaned. Going higher pressures the OCR generate cap at
# memory/visual_bus.py:229 (max_new_tokens=8192) without much added recall.
MAX_VISUAL_TILES = 16             # max turns to keep in the visual history image
# Device pin for GLM-OCR. With 2 GPUs allocated (sbatch --gpus=2), vLLM
# owns cuda:0 (0.90 reservation) and OCR gets cuda:1 with all 94 GB to
# itself. memory/visual_bus.py auto-falls back to device_map="auto" if
# the requested device index doesn't exist (e.g. single-GPU host).
OCR_DEVICE = "cuda:1"
# OCR generate token cap. Scales with input image height: at MAX_VISUAL_TILES=4
# and ~150 px/tile, 8192 was sufficient. After bumping tiles to 16 the rendered
# image is ~4× taller and OCR output text grows with it; a hard 8192 cap risks
# silent truncation of the compressed history. 16384 leaves margin.
OCR_MAX_NEW_TOKENS = 16384

# --- Entropy Router (Phase 4) ---
# Thresholds are measured in bits of Shannon entropy on the first
# generated token of a single-token probe pass with no memory injected.
# Distribution is top-K + residual bucket (see utils/llm._shannon_with_residual).
ENTROPY_LOW_THRESHOLD = 0.4   # below: MSA only (frozen rulebook)
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
