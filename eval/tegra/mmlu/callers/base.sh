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

# Base Evaluation Sweep Script
# Sweeps across different models using full reasoning base evaluator
# 
# Usage:
#   ./sweep_base.sh                    # Run all models on GPU with default settings
#   ./sweep_base.sh --cpu              # Run all models on CPU
#   ./sweep_base.sh --num-questions 5  # Run with 5 questions per subject
#   ./sweep_base.sh --cpu --num-questions 5  # Run on CPU with 5 questions per subject
#   ./sweep_base.sh --no-flash-attention  # Disable Flash Attention (use PyTorch native attention)
#   ./sweep_base.sh --no-screen        # Run without screen session (direct execution)
#
# The script automatically runs in a screen session for persistence unless --no-screen is used.
# Use 'screen -r sweep_base' to reattach to the running session.

set -e

# Check if we should skip screen session
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
    echo "🖥️  Starting sweep in screen session: 'sweep_base'"
    echo "   Use 'screen -r sweep_base' to reattach later"
    echo "   Use 'screen -ls' to list all sessions"
    echo ""
    
    # Kill any existing sweep_base screen session
    screen -S sweep_base -X quit 2>/dev/null || true
    
    # Start new screen session with this script
    exec screen -S sweep_base -dm bash -c "cd '$(pwd)' && '$0' --no-screen ${FILTERED_ARGS[*]}; exec bash"
fi

# If we're here, we're either in screen or --no-screen was used
if [[ -n "$STY" ]]; then
    echo "📺 Running in screen session: $STY"
elif [[ "$NO_SCREEN" == "true" ]]; then
    echo "🔧 Running in direct mode (no screen session)"
fi

export VLLM_ATTENTION_BACKEND="FLASHINFER"
export VLLM_USE_V1=0
export VLLM_USE_TRITON_FLASH_ATTN=false

# Check if --cpu flag is present and set CPU environment variables
for arg in "${FILTERED_ARGS[@]}"; do
    if [[ "$arg" == "--cpu" ]]; then
        echo "🖥️  CPU mode detected - setting CPU environment variables"
        export CUDA_VISIBLE_DEVICES=""
        export VLLM_CPU_KVCACHE_SPACE=40
        export VLLM_CPU_NUM_OF_RESERVED_CPU=1
        break
    fi
done

# Clean up previous run log
echo "🧹 Cleaning up previous run.log..."
rm -f run.log

# List of models to evaluate
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
)

# Base directories and config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE="$SCRIPT_DIR/results"
CONFIG_FILE="$SCRIPT_DIR/../configs/base.yaml"

echo "🚀 Starting Base Evaluation Sweep"
echo "=================================="
echo "Models: ${MODELS[*]}"
echo "📝 All output will be logged to: run.log"
echo ""

# Create results directory
mkdir -p "$RESULTS_BASE"

# Main loop - pass through all additional arguments  
{
    for model in "${MODELS[@]}"; do
        echo "📊 Running base evaluation for Model=$model"
        stdbuf -o0 -e0 python3 "$SCRIPT_DIR/../scripts/base.py" \
            --model "$model" \
            --config "$CONFIG_FILE" \
            "${FILTERED_ARGS[@]}" \
            || echo "❌ Failed: $model"
        echo "✅ Completed: $model"
        echo ""
    done

    echo "🎉 Base evaluation sweep completed! Results in: $RESULTS_BASE"
    
    # Screen session management info
    if [[ -n "$STY" ]]; then
        echo ""
        echo "📺 Screen session info:"
        echo "   Session name: sweep_base"
        echo "   You can now detach with: Ctrl+A, then D"
        echo "   To reattach later: screen -r sweep_base"
        echo "   To kill this session: screen -S sweep_base -X quit"
    fi
} 2>&1 | tee -a run.log

# Show log file stats
echo ""
echo "📊 Log file stats:"
echo "   Lines: $(wc -l < run.log)"
echo "   Size: $(du -h run.log | cut -f1)"
echo "   Location: $(pwd)/run.log"

# Final screen session reminder
if [[ -n "$STY" ]]; then
    echo ""
    echo "💡 This screen session will remain active. Use 'screen -r sweep_base' to reattach anytime."
fi
