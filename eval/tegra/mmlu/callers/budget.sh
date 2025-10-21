#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Budget Evaluation Sweep Script
# Sweeps across different models and max_tokens configurations

set -e

# Configuration arrays
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

MAX_TOKENS_VALUES=(128 256)

# Base directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE="$SCRIPT_DIR/results"
CONFIG_TEMPLATE="$SCRIPT_DIR/../configs/budget.yaml"
TEMP_CONFIG="/tmp/budget_temp.yaml"

echo "🚀 Starting Budget Evaluation Sweep"
echo "=================================="
echo "Models: ${MODELS[*]}"
echo "Max tokens: ${MAX_TOKENS_VALUES[*]}"
echo ""

# Create results directory
mkdir -p "$RESULTS_BASE"

# Main sweep loop
for model in "${MODELS[@]}"; do
    for max_tokens in "${MAX_TOKENS_VALUES[@]}"; do
        echo "📊 Running: Model=$model, MaxTokens=$max_tokens"
        
        # Create temporary config with substituted values
        sed -e "s/max_tokens: [0-9]*/max_tokens: $max_tokens/" \
            -e "s/MAX_TOKENS_PLACEHOLDER/$max_tokens/" \
            "$CONFIG_TEMPLATE" > "$TEMP_CONFIG"
        
        # Run evaluation
        python3 "$SCRIPT_DIR/../scripts/budget.py" \
            --model "$model" \
            --config "$TEMP_CONFIG" \
            --max-tokens "$max_tokens" \
            || echo "❌ Failed: $model with $max_tokens tokens"
        
        echo "✅ Completed: $model with $max_tokens tokens"
        echo ""
    done
done

# Cleanup
rm -f "$TEMP_CONFIG"

echo "🎉 Sweep completed! Results in: $RESULTS_BASE"
