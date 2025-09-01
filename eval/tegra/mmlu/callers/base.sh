#!/bin/bash
# Base Evaluation Script
# 
# Usage:
#   ./base.sh                          # Run all models on GPU with default settings
#   ./base.sh --cpu                    # Run all models on CPU
#   ./base.sh --num-questions 5        # Run with 5 questions per subject
#   ./base.sh --cpu --num-questions 5  # Run on CPU with 5 questions per subject
#   ./base.sh --no-flash-attention     # Disable Flash Attention (use PyTorch native attention)
#   ./base.sh --no-screen              # Run without screen session (direct execution)
#
# The script automatically runs in a screen 

set -e

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
RESULTS_BASE="$(python3 "$SCRIPT_DIR/../scripts/utils.py")"
CONFIG_FILE="$SCRIPT_DIR/../configs/base.yaml"

NO_SCREEN=false
FILTERED_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-screen" ]]; then
        NO_SCREEN=true
    else
        FILTERED_ARGS+=("$arg")
    fi
done

# Auto-start in screen session unless already in one or --no-screen is used
if [[ -z "$STY" && "$NO_SCREEN" == "false" ]]; then
    echo "* Starting base evaluation in screen session: 'base'"
    echo "  Use 'screen -r base' to reattach later"
    echo "  Use 'screen -ls' to list all sessions"
    echo ""
    
    screen -S base -X quit 2>/dev/null || true
    
    # Debug: Log what command we're about to run
    echo "cd '$(pwd)' && '$0' --no-screen ${FILTERED_ARGS[*]}; exec bash" > /tmp/base_screen_command.txt
    
    exec screen -S base -dm bash -c "cd '$(pwd)' && '$0' --no-screen ${FILTERED_ARGS[*]}; exec bash"
fi

if [[ -n "$STY" ]]; then
    echo "* Running in screen session: $STY"
elif [[ "$NO_SCREEN" == "true" ]]; then
    echo "* Running in direct mode (no screen session)"
    echo "* Arguments received: $@"
    echo "* Filtered args: ${FILTERED_ARGS[*]}"
fi

export VLLM_ATTENTION_BACKEND="FLASHINFER"
export VLLM_USE_V1=0
export VLLM_USE_TRITON_FLASH_ATTN=false

for arg in "${FILTERED_ARGS[@]}"; do
    if [[ "$arg" == "--cpu" ]]; then
        echo "* CPU mode detected - setting CPU environment variables"
        export CUDA_VISIBLE_DEVICES=""
        export VLLM_CPU_KVCACHE_SPACE=40
        export VLLM_CPU_NUM_OF_RESERVED_CPU=1
        break
    fi
done

# Clean up previous run log
echo "* Cleaning up previous run.log..."
rm -f run.log

# List of models to evaluate - read from models.txt
MODELS_FILE="$SCRIPT_DIR/../models.txt"
if [[ -f "$MODELS_FILE" ]]; then
    echo "* Reading models from: $MODELS_FILE"
    mapfile -t MODELS < "$MODELS_FILE"
    FILTERED_MODELS=()
    for model in "${MODELS[@]}"; do
        if [[ -n "$model" && ! "$model" =~ ^[[:space:]]*# ]]; then
            FILTERED_MODELS+=("$model")
        fi
    done
    MODELS=("${FILTERED_MODELS[@]}")
else
    echo "! models.txt not found at: $MODELS_FILE, using default models"
    MODELS=(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    )
fi

echo "> Starting Base Evaluation"
echo "=========================="
echo "Models: ${MODELS[*]}"
echo "* All output will be logged to: run.log"
echo ""

mkdir -p "$RESULTS_BASE"

{
    for model in "${MODELS[@]}"; do
        echo "* Running base evaluation for Model=$model"
        stdbuf -o0 -e0 python3 "$SCRIPT_DIR/../scripts/base.py" \
            --model "$model" \
            --config "$CONFIG_FILE" \
            "${FILTERED_ARGS[@]}" \
            || echo "x Failed: $model"
        echo "✓ Completed: $model"
        echo ""
    done

    echo "* Base evaluation completed! Results in: $RESULTS_BASE"
    
    # Screen session management info
    if [[ -n "$STY" ]]; then
        echo ""
        echo "* Screen session info:"
        echo "  Session name: base"
        echo "  You can now detach with: Ctrl+A, then D"
        echo "  To reattach later: screen -r base"
        echo "  To kill this session: screen -S base -X quit"
    fi
} 2>&1 | tee -a run.log

echo ""
echo "* Log file stats:"
echo "  Lines: $(wc -l < run.log)"
echo "  Size: $(du -h run.log | cut -f1)"
echo "  Location: $(pwd)/run.log"

if [[ -n "$STY" ]]; then
    echo ""
    echo "* This screen session will remain active. Use 'screen -r base' to reattach anytime."
fi
