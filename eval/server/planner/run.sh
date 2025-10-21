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
