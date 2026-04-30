#!/bin/bash
#SBATCH --job-name=trimem-longmemeval
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

# --- Python module ---
# runner.ipynb runs on cp312 with the prebuilt FA2 wheel for cu128/torch2.10.
# Adjust the module name to whatever your cluster exposes.
module purge
module load python/3.12

# Module sets PYTHONHOME, which conflicts with the venv. Drop it.
unset PYTHONHOME

# --- Virtualenv ---
# Recreate venv if it was built with a different interpreter.
if [ -d "venv" ] && ! venv/bin/python -c 'import sys; assert sys.version_info[:2]==(3,12)' 2>/dev/null; then
    rm -rf venv
fi
if [ ! -d "venv" ]; then
    python3.12 -m venv venv
fi
source venv/bin/activate

pip install --quiet --upgrade pip

# --- Tri-Mem dependencies (mirrors runner.ipynb cell-6) ---
# vLLM nightly tracks the latest CUDA+torch combos and is needed for
# Mamba/MoE support (Qwen3.5-35B-A3B). Bleeding-edge transformers is
# required because GLM-OCR's processor lives on main (not yet released).
pip install --quiet -U vllm --extra-index-url https://wheels.vllm.ai/nightly
pip install --quiet chromadb "git+https://github.com/huggingface/transformers.git"
pip install --quiet -U huggingface_hub

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
pip install --quiet "$FA_WHEEL" || \
    echo "WARN: FA2 wheel install failed (cu${CUDA_V}/torch${TORCH_V}/cp${PY_V}) — vLLM will fall back to SDPA"

# --- HF auth (Qwen + GLM-OCR + dataset) ---
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true

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
