#!/bin/bash
# Budget Evaluation Sweep Script
# Sweeps across different models and max_tokens configurations

set -e

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
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
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    )
fi

MAX_TOKENS_VALUES=(128 256)

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
RESULTS_BASE="$(python3 "$SCRIPT_DIR/../scripts/utils.py")"
CONFIG_TEMPLATE="$SCRIPT_DIR/../configs/budget.yaml"
TEMP_CONFIG="/tmp/budget_temp.yaml"

echo "↑ Starting Budget Evaluation Sweep"
echo "=================================="
echo "Models: ${MODELS[*]}"
echo "Max tokens: ${MAX_TOKENS_VALUES[*]}"
echo ""

# Create results directory
mkdir -p "$RESULTS_BASE"

# Main sweep loop
for model in "${MODELS[@]}"; do
    for max_tokens in "${MAX_TOKENS_VALUES[@]}"; do
        echo "≡ Running: Model=$model, MaxTokens=$max_tokens"
        
        # Create temporary config with substituted values
        sed -e "s/max_tokens: [0-9]*/max_tokens: $max_tokens/" \
            -e "s/MAX_TOKENS_PLACEHOLDER/$max_tokens/" \
            "$CONFIG_TEMPLATE" > "$TEMP_CONFIG"
        
        # Run evaluation
        python3 "$SCRIPT_DIR/../scripts/budget.py" \
            --model "$model" \
            --config "$TEMP_CONFIG" \
            --max-tokens "$max_tokens" \
            || echo "✗ Failed: $model with $max_tokens tokens"
        
        echo "✓ Completed: $model with $max_tokens tokens"
        echo ""
    done
done

# Cleanup
rm -f "$TEMP_CONFIG"

echo "★ Sweep completed! Results in: $RESULTS_BASE"
