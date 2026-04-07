#!/bin/bash
# HPEMA tmux workflow — run this ON a GPU node after salloc.
#
# Prerequisites:
#   1. salloc --partition=a100_normal_q --gres=gpu:1 --time=2:00:00
#   2. cd /path/to/Capstone
#   3. bash ml/slurm/start_tmux.sh
#
# This creates a tmux session "hpema" with 2 panes:
#   0: vLLM serving Phi-4
#   1: CLI (interactive REPL, runs orchestrator in-process — no backend needed)
#
# To attach later: tmux attach -t hpema
# To kill:         tmux kill-session -t hpema

set -e

SESSION="hpema"
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="hpema_config.arc.yaml"
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
VLLM_PORT=8001

echo "=== HPEMA tmux launcher ==="
echo "Project: $PROJECT_DIR"
echo "Config:  $CONFIG"
echo "Model:   $MODEL"
echo "Node:    $(hostname)"
if command -v nvidia-smi &>/dev/null; then
    echo "GPUs:    $(nvidia-smi -L 2>/dev/null | wc -l) detected"
fi
echo ""

# -----------------------------------------------------------
# Dependency setup (runs once before tmux session starts)
# -----------------------------------------------------------

# Install app requirements if any are missing
echo ">>> Installing app dependencies..."
pip install  -r "$PROJECT_DIR/requirements.txt"

# Install vLLM if not already present.
# vLLM is GPU-only infrastructure, kept separate from requirements.txt.
if ! python -c "import vllm" 2>/dev/null; then
    echo ">>> vLLM not found — installing (this may take a few minutes)..."
    pip install vllm
    echo ">>> vLLM installed."
else
    echo ">>> vLLM already installed — skipping."
fi
echo ""

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session — pane 0: vLLM
tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:0" "echo '=== PANE 0: vLLM ===' && \
python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --port $VLLM_PORT \
  --max-model-len 8192 \
  --structured-outputs-config.backend outlines \
  --dtype bfloat16 \
  --trust-remote-code" Enter

# Pane 1: CLI (split horizontal)
tmux split-window -h -t "$SESSION:0" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:0.1" "echo '=== PANE 1: HPEMA CLI ===' && \
echo 'Waiting 10s for the sake of 10s...' && \
sleep 10 && \
export HPEMA_CONFIG=$CONFIG && \
export HPEMA_API_KEY=unused && \
export ANONYMIZED_TELEMETRY=False && \
python -m cli.main" Enter

# Attach
tmux attach -t "$SESSION"
