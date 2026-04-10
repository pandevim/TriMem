"""Tri-Mem configuration."""

# --- LLM ---
# Change MODEL_NAME to any HuggingFace model ID to swap models.
MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
INFERENCE_BACKEND = "vllm"  # "vllm" (fast, GPU) or "transformers" (universal fallback)
MAX_TOKENS = 1024
# vLLM-specific settings
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 16384

# --- Agent ---
MAX_AGENT_TURNS = 60
AGENT_TEMPERATURE = 0.2

# --- Benchmark ---
DEFAULT_NUM_TASKS = 10

# --- RAG (Phase 2) ---
RAG_COLLECTION = "tri_mem_facts"
RAG_TOP_K = 3

# --- Visual Bus (Phase 3) ---
OCR_MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"
VISUAL_BUS_HISTORY_DIR = "/tmp/trimem_visual_bus"
VISUAL_BUS_BASE_SIZE = 1024       # DeepSeek-OCR-2 base_size param
VISUAL_BUS_IMAGE_SIZE = 768       # DeepSeek-OCR-2 image_size param
VISUAL_BUS_IMAGE_WIDTH = 1200     # pixel width of rendered history image
VISUAL_BUS_FONT_SIZE = 14         # pt — monospace font size in rendered image
MAX_VISUAL_TILES = 20             # max turns to keep in the visual history image

# --- Entropy Router (Phase 4) ---
ENTROPY_LOW_THRESHOLD = 0.3   # rely on MSA/internal
ENTROPY_MED_THRESHOLD = 0.7   # check visual bus
# above 0.7 -> query RAG

# --- Frontend ---
FRONTEND_PORT = 5000
