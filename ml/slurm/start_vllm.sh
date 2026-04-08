#!/bin/bash
# HPEMA vLLM isolated runner - No tmux required!
# Run this ON the fal048 GPU node after salloc.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="$PROJECT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
    echo ">>> Loading secrets from .env"
    set -a
    source "$ENV_FILE"
    set +a
fi

MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
VLLM_PORT=8001

echo "=== HPEMA vLLM Launcher ==="
echo "Node:    $(hostname)"
if command -v nvidia-smi &>/dev/null; then
    echo "GPUs:    $(nvidia-smi -L | wc -l) detected"
fi

pip install -r "$PROJECT_DIR/requirements.txt"
if ! python -c "import vllm" 2>/dev/null; then
    pip install vllm
fi

echo ">>> Starting vLLM Server on Port $VLLM_PORT..."
python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --port $VLLM_PORT \
  --max-model-len 8192 \
  --structured-outputs-config.backend outlines \
  --dtype bfloat16 \
  --trust-remote-code
