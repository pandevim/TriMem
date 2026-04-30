#!/bin/bash
#SBATCH --job-name=trimem-longmemeval
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/longmemeval_%j.out
#SBATCH --error=logs/longmemeval_%j.err

set -euo pipefail
mkdir -p logs

# --- Secrets ---
# HF_TOKEN is needed to pull Qwen3.5-35B-A3B, GLM-OCR, and the
# longmemeval-cleaned dataset. Export before sbatch:
#   export HF_TOKEN=hf_...
#   sbatch run_longmemeval.sbatch
: "${HF_TOKEN:?HF_TOKEN not set — export before sbatch}"

# --- Python + CUDA modules ---
# runner.ipynb runs on cp312, but the cluster only exposes 3.11.10. The
# FA2 auto-detect below resolves the cp311 wheel; vLLM nightly publishes
# both cp311 and cp312 builds.
#
# CUDA module is required: vLLM nightly falls back to a source build if
# no prebuilt wheel matches the resolved torch/CUDA/python triple, and
# the source build needs nvcc + CUDA_HOME. Adjust the cuda/* version to
# whatever your cluster exposes (`module avail cuda`).
module purge
module load python/3.11.10
# Cluster modulefile is nested: `cuda12.8/toolkit/12.8.1` (see `module avail cuda`).
# Other 12.x toolkits available: cuda12.{0,1,2,4}/toolkit/... — pick one that
# matches the torch wheel index below (cu128 ↔ 12.8, cu124 ↔ 12.4, etc.).
module load cuda12.8/toolkit/12.8.1
# Git is not available as a module on this cluster. We will bypass the need
# for it by using ZIP downloads for pip installations.
# module load git || { echo "ERROR: no git module — run 'module avail git' to find one"; exit 1; }

# Make CUDA_HOME explicit even if the module didn't set it. nvcc lives
# at $CUDA_HOME/bin/nvcc, so derive it from `which nvcc`.
if [ -z "${CUDA_HOME:-}" ]; then
    NVCC_BIN=$(which nvcc 2>/dev/null || true)
    if [ -n "$NVCC_BIN" ]; then
        export CUDA_HOME=$(dirname "$(dirname "$NVCC_BIN")")
    fi
fi
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"

# Module sets PYTHONHOME, which conflicts with the venv. Drop it.
unset PYTHONHOME

# --- Virtualenv ---
# Recreate venv if it was built with a different interpreter.
if [ -d "venv" ] && ! venv/bin/python -c 'import sys; assert sys.version_info[:2]==(3,11)' 2>/dev/null; then
    rm -rf venv
fi
if [ ! -d "venv" ]; then
    python3.11 -m venv venv
fi
source venv/bin/activate

pip install --quiet --upgrade pip

# --- Tri-Mem dependencies (mirrors runner.ipynb cell-6) ---
# Install torch first with a pinned CUDA wheel index — this lets vLLM
# resolve to a *prebuilt* nightly wheel for that exact torch instead of
# falling back to a source build (which needs nvcc + CUDA_HOME and is
# slow). Match the cu* index to the CUDA module loaded above.
pip install torch --index-url https://download.pytorch.org/whl/cu128

# vLLM nightly tracks the latest CUDA+torch combos and is needed for
# Mamba/MoE support (Qwen3.5-35B-A3B). Bleeding-edge transformers is
# required because GLM-OCR's processor lives on main (not yet released).
# Bleeding-edge transformers is required because GLM-OCR's processor lives 
# on main. We use the .zip URL to bypass the need for a local 'git' binary.
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
pip install chromadb "https://github.com/huggingface/transformers/archive/refs/heads/main.zip"
pip install -U huggingface_hub

# --- FlashAttention-2 ---
# H100 (sm_90) supports both FA2 and FA3. We use the same prebuilt FA2
# wheel as runner.ipynb for parity; vLLM falls back to SDPA if the
# wheel doesn't match the resolved torch/cuda/python triple.
# (FA3 would be ~1.3-1.5x faster on Hopper but needs a separate build.)
PY_V=$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
TORCH_V=$(python -c 'import torch; v=torch.__version__.split("+")[0].split("."); print(".".join(v[:2]))')
CUDA_V=$(python -c 'import torch; print(torch.version.cuda.replace(".",""))')
FA_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu${CUDA_V}torch${TORCH_V}-cp${PY_V}-cp${PY_V}-linux_x86_64.whl"
echo "Trying FA2 wheel: $FA_WHEEL"
pip install "$FA_WHEEL" || \
    echo "WARN: FA2 wheel install failed (cu${CUDA_V}/torch${TORCH_V}/cp${PY_V}) — vLLM will fall back to SDPA"

# --- HF auth (Qwen + GLM-OCR + dataset) ---
huggingface-cli login --token "$HF_TOKEN"

# --- vLLM + telemetry env (mirrors runner.ipynb) ---
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OTEL_SDK_DISABLED=true

# Keep HF cache on the workspace, not $HOME (HPC quota).
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
mkdir -p "$HF_HOME"

# --- LongMemEval data ---
# Three splits ship at huggingface.co/datasets/xiaowu0162/longmemeval-cleaned:
#   longmemeval_oracle.json    — only evidence sessions (smallest, smoke runs)
#   longmemeval_s_cleaned.json — small haystack (~50 sessions / task)
#   longmemeval_m_cleaned.json — medium haystack
# Default to oracle; override with DATA_FILE env var.
DATA_DIR=${DATA_DIR:-./data/longmemeval}
DATA_FILE=${DATA_FILE:-$DATA_DIR/longmemeval_oracle.json}
mkdir -p "$DATA_DIR"
if [ ! -f "$DATA_FILE" ]; then
    SPLIT_NAME=$(basename "$DATA_FILE")
    echo "Downloading $SPLIT_NAME from longmemeval-cleaned..."
    python -c "
import shutil, sys
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='xiaowu0162/longmemeval-cleaned',
    filename='$SPLIT_NAME',
    repo_type='dataset',
)
shutil.copy(p, '$DATA_FILE')
print(f'Saved to $DATA_FILE', flush=True)
"
fi

# --- Run ---
# Override MAX_TASKS / METHOD / OUTPUT in the calling shell:
#   MAX_TASKS=50 METHOD=exact_substring sbatch run_longmemeval.sbatch
MAX_TASKS=${MAX_TASKS:-10}
METHOD=${METHOD:-exact_substring}
OUTPUT=${OUTPUT:-logs/longmemeval_${SLURM_JOB_ID}.json}

python run_longmemeval.py \
    --data "$DATA_FILE" \
    --agent trimem \
    --max-tasks "$MAX_TASKS" \
    --method "$METHOD" \
    --output "$OUTPUT"

echo "DONE — results at $OUTPUT"
