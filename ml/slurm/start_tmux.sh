#!/bin/bash
# HPEMA tmux workflow — run this ON a GPU node after salloc.
#
# Prerequisites:
#   1. salloc --partition=a100_normal_q --gres=gpu:1 --time=2:00:00
#   2. cd /path/to/Capstone
#   3. bash ml/slurm/start_tmux.sh
#
# This creates a tmux session "hpema" with 3 panes:
#   0: vLLM serving Phi-4
#   1: FastAPI backend
#   2: Your working shell (run CLI, smoke tests, etc.)
#
# To attach later: tmux attach -t hpema
# To kill:         tmux kill-session -t hpema

set -e

SESSION="hpema"
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="hpema_config.arc.yaml"
MODEL="microsoft/Phi-4-Reasoning-Plus"
VLLM_PORT=8001
BACKEND_PORT=8000

echo "=== HPEMA tmux launcher ==="
echo "Project: $PROJECT_DIR"
echo "Config:  $CONFIG"
echo "Model:   $MODEL"
echo "Node:    $(hostname)"
if command -v nvidia-smi &>/dev/null; then
    echo "GPUs:    $(nvidia-smi -L 2>/dev/null | wc -l) detected"
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
  --dtype auto \
  --trust-remote-code" Enter

# Pane 1: FastAPI backend (split horizontal)
tmux split-window -h -t "$SESSION:0" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:0.1" "echo '=== PANE 1: Backend ===' && \
echo 'Waiting 60s for vLLM to load model...' && \
sleep 60 && \
export HPEMA_CONFIG=$CONFIG && \
export HPEMA_API_KEY=unused && \
python -m uvicorn backend.main:app --host 0.0.0.0 --port $BACKEND_PORT" Enter

# Pane 2: Working shell (split pane 1 vertically)
tmux split-window -v -t "$SESSION:0.1" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:0.2" "echo '=== PANE 2: Working shell ===' && \
echo '' && \
echo 'Wait for vLLM + backend to be ready, then run:' && \
echo '  python ml/slurm/smoke_test.py           # quick LLM test' && \
echo '  python -m cli.main generate \\\\' && \
echo '    --requirement \"binary search with bounds checking\" \\\\' && \
echo '    --standard DO_178C --language C' && \
echo '' && \
export HPEMA_CONFIG=$CONFIG && \
export HPEMA_API_KEY=unused" Enter

# Attach
tmux attach -t "$SESSION"
