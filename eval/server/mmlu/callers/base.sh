#!/bin/bash
# Base Evaluation Sweep Script
# Sweeps across different models using full reasoning base evaluator

set -e
export CUDA_VISIBLE_DEVICES=1

# Configuration arrays
MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
)

# MAX_TOKENS_VALUES=(2048 4096)

# Base directories and config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE="$SCRIPT_DIR/results"
CONFIG_FILE="$SCRIPT_DIR/configs/base.yaml"

echo ">> Starting Base Evaluation Sweep"
echo "=================================="
echo "Models: ${MODELS[*]}"
echo ""

mkdir -p "$RESULTS_BASE"

for model in "${MODELS[@]}"; do
    echo "[*] Running base evaluation for Model=$model"
    python3 "$SCRIPT_DIR/all_base.py" \
        --model "$model" \
        --config "$CONFIG_FILE" \
        || echo "[X] Failed: $model"
    echo "[✓] Completed: $model"
    echo ""
done

echo "== Base evaluation sweep completed! Results in: $RESULTS_BASE"