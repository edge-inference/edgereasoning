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

# cached models
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

rm run.log

# synthetic data scripts
synthetic=(
    "synthetic/decode.py"
    "synthetic/prefill.py"
    "synthetic/tt_scaling.py"
)

case "${1:-base}" in
    "base"|"scripts"|"")
        echo "Running base evaluation..."
        bash callers/base.sh 2>&1 | tee -a run.log
        ;;
    "scaling"|"tt_scaling")
        echo "Running test-time scaling evaluation..."
        bash callers/tt_scaling.sh 2>&1 | tee -a run.log
        ;;
    "budget")
        echo "Running budget evaluation..."
        bash callers/budget.sh 2>&1 | tee -a run.log
        ;;
    "prefill")
        echo "Running prefill evaluation..."
        shift  
        python synthetic/prefill.py "$@" 2>&1 | tee -a run.log
        ;;
    "decode")
        echo "Running decode evaluation..."
        shift  
        python synthetic/decode.py "$@" 2>&1 | tee -a run.log
        ;;
    "synthetic")
        echo "Running synthetic scripts individually..."
        shift
        for file in "${synthetic[@]}"; do
            echo "Running: $file $@"
            python "$file" "$@" 2>&1 | tee -a run.log
            echo "--------------------------------"
        done
        ;;
    "all")
        echo "Running all evaluations..."
        for caller in callers/*.sh; do
            echo "Running: $caller"
            bash "$caller" 2>&1 | tee -a run.log
            echo "--------------------------------"
        done
        ;;
    *)
        echo "Usage: $0 [base|scaling|budget|prefill|decode|synthetic|all] [options]"
        echo "  base:     run base evaluation (default)"
        echo "  scaling:  run test-time scaling evaluation"
        echo "  budget:   run budget evaluation"
        echo "  prefill:  run prefill synthetic evaluation"
        echo "  decode:   run decode synthetic evaluation"
        echo "  synthetic: run synthetic scripts individually"
        echo "  all:      run all main evaluations"
        echo ""
        echo "Options for prefill/decode/synthetic:"
        echo "  --models_file FILE  Model names file (default: models.txt)"
        echo "  --config FILE       Config file (default: configs/prefill.yaml)"
        echo "  --output_dir DIR    Output directory"
        echo "  --warmup_runs N     Warmup runs (default: 1)"
        echo "  --help              Show help for specific script"
        exit 1
        ;;
esac