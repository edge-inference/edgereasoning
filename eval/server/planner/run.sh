#!/bin/bash
# Natural Planner Evaluation Coordinator
# Usage: ./run.sh [evaluation_type]
# 
# Examples:
#   ./run.sh                    # Run default evaluation
#   ./run.sh direct             # Run direct evaluation (no reasoning)
#   ./run.sh budget             # Run budget evaluation
#   ./run.sh scaling            # Run test-time scaling evaluation
#   ./run.sh base               # Run base evaluation (all tasks)

set -e

# Environment setup
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0
export VLLM_ENABLE_METRICS=true
export VLLM_PROFILE=true
export VLLM_DETAILED_METRICS=true
export VLLM_REQUEST_METRICS=true

# Default configuration
DEFAULT_GPUS="0,1,2"
DEFAULT_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Clean up previous logs
rm -f run.log

# Evaluation type configurations
case "${1:-base}" in
    "direct")
        echo " Starting Direct Natural Planner Evaluation (No Reasoning)"
        bash "$SCRIPT_DIR/callers/direct.sh" "${@:2}"
        ;;
    "budget")
        echo " Starting Budget Natural Planner Evaluation"
        bash "$SCRIPT_DIR/callers/budget.sh" "${@:2}"
        ;;
    "scaling"|"tt_scale")
        echo " Starting Test-Time Scaling Natural Planner Evaluation"
        bash "$SCRIPT_DIR/callers/tt_scale.sh" "${@:2}"
        ;;
    "base"|"all"|"")
        echo " Starting Base Natural Planner Evaluation (All Tasks)"
        echo "Model: ${DEFAULT_MODEL}"
        echo "GPUs: ${DEFAULT_GPUS}"
        python -u "$SCRIPT_DIR/planner.py" \
            --task all \
            --model "${DEFAULT_MODEL}" \
            --gpus "${DEFAULT_GPUS}" \
            --output "data/planner/server/base" \
            2>&1 | tee -a run.log
        ;;
    *)
        echo "Error: Unknown evaluation type: $1"
        echo ""
        echo "Available evaluation types:"
        echo "  base       - Base evaluation (all tasks)"
        echo "  direct     - Direct evaluation (no reasoning)"
        echo "  budget     - Budget evaluation"  
        echo "  scaling    - Test-time scaling evaluation"
        exit 1
        ;;
esac

echo ""
echo "Natural Planner evaluation completed!"
