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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

show_usage() {
    echo "MMLU Evaluation Launcher"
    echo "======================="
    echo ""
    echo "Usage: ./run.sh [mode] [options]"
    echo ""
    echo "Modes:"
    echo "  base        - Full reasoning evaluation (4096 tokens)"
    echo "  budget      - Budget evaluation (configurable tokens)"
    echo "  noreasoning - Direct answer selection"
    echo "  scale       - Parameter scaling experiments"
    echo ""
    echo "Options:"
    echo "  --model MODEL_NAME     Override default model"
    echo "  --max-tokens N         Override token limit"
    echo "  --help                 Show this help"
    echo ""
}

run_evaluation() {
    local mode="${1:-base}"
    local model="$2"
    local max_tokens="$3"
    
    echo "Starting MMLU Evaluation - Mode: $mode"
    echo "Time: $(date)"
    echo "Working directory: $SCRIPT_DIR"
    echo ""
    
    export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
    OUTPUT_DIR="$REPO_ROOT/data/mmlu/server/"
    mkdir -p "$OUTPUT_DIR"
    
    case "$mode" in
        "base")
            echo "* Running base evaluation..."
            python scripts/base.py ${model:+--model "$model"} ${max_tokens:+--max-tokens "$max_tokens"}
            ;;
        "budget")
            echo "* Running budget evaluation..."
            python scripts/budget.py ${model:+--model "$model"} ${max_tokens:+--max-tokens "$max_tokens"}
            ;;
        "noreasoning")
            echo "* Running no-reasoning evaluation..."
            python scripts/noreasoning.py ${model:+--model "$model"} ${max_tokens:+--max-tokens "$max_tokens"}
            ;;
        "direct")
            echo "* Running direct evaluation..."
            python scripts/direct.py ${model:+--model "$model"} ${max_tokens:+--max-tokens "$max_tokens"}
            ;;
        "scale")
            echo "* Running test time scaling experiments..."
            ./callers/scale.sh
            ;;
        *)
            echo "ERROR: Unknown mode '$mode'"
            show_usage
            exit 1
            ;;
    esac
    
    echo ""
    echo "* Evaluation completed: $(date)"
    echo "* Results saved to: $OUTPUT_DIR"
}

# Parse arguments
MODE="base"
MODEL=""
MAX_TOKENS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -*)
            echo "ERROR: Unknown option $1"
            show_usage
            exit 1
            ;;
        *)
            MODE="$1"
            shift
            ;;
    esac
done

run_evaluation "$MODE" "$MODEL" "$MAX_TOKENS"
