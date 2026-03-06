#!/bin/bash
#SBATCH --job-name=hpema-vllm
#SBATCH --partition=a100_normal_q
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/vllm-%j.out
#SBATCH --error=logs/slurm/vllm-%j.err

# HPEMA Model Serving — launches vLLM instances for all agents
# Usage: sbatch ml/slurm/serve_models.sh
# Or interactively: salloc --partition=a100_normal_q --gres=gpu:4 --time=2:00:00

set -e

echo "=== HPEMA Model Serving ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L)"
echo ""

# Actor: Qwen3-Coder-Next (80B MoE, needs 2x A100)
echo "Starting Actor (Qwen3-Coder-Next) on port 8001..."
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Coder-Next \
    --port 8001 \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --guided-decoding-backend outlines &
ACTOR_PID=$!

# Checker: Phi-4-Reasoning-Plus (14B, 1x A100)
echo "Starting Checker (Phi-4-Reasoning-Plus) on port 8002..."
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-4-Reasoning-Plus \
    --port 8002 \
    --max-model-len 8192 \
    --guided-decoding-backend outlines &
CHECKER_PID=$!

# Policy: Mistral-Small-3.2-24B (1x A100)
echo "Starting Policy (Mistral-Small-3.2-24B) on port 8003..."
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-Small-3.2-24B-Instruct \
    --port 8003 \
    --max-model-len 8192 \
    --guided-decoding-backend outlines &
POLICY_PID=$!

echo ""
echo "All models starting. Waiting for health checks..."
sleep 30

# Health check
for port in 8001 8002 8003; do
    if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
        echo "  Port ${port}: HEALTHY"
    else
        echo "  Port ${port}: NOT READY (may still be loading)"
    fi
done

echo ""
echo "=== Ready for pipeline. Start backend with: ==="
echo "  uvicorn backend.main:app --port 8000"
echo ""
echo "=== Environment variables to set: ==="
echo "  export HPEMA_ACTOR_ENDPOINT=http://localhost:8001/v1"
echo "  export HPEMA_ACTOR_MODEL=Qwen/Qwen3-Coder-Next"
echo "  export HPEMA_CHECKER_ENDPOINT=http://localhost:8002/v1"
echo "  export HPEMA_CHECKER_MODEL=microsoft/Phi-4-Reasoning-Plus"
echo "  export HPEMA_POLICY_ENDPOINT=http://localhost:8003/v1"
echo "  export HPEMA_POLICY_MODEL=mistralai/Mistral-Small-3.2-24B-Instruct"

# Wait for all background processes
wait $ACTOR_PID $CHECKER_PID $POLICY_PID
