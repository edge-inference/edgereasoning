#!/usr/bin/env bash

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

# Quick status checker for multi-GPU scale sweep
# Shows current progress and GPU utilization

echo "========================================"
echo "MULTI-GPU SCALE SWEEP STATUS CHECKER"
echo "========================================"
echo "Current time: $(date)"
echo ""

# Check if master launcher is running
MASTER_PID=$(pgrep -f "sweep_scale_master.sh" | head -1)
if [[ -n "$MASTER_PID" ]]; then
    echo "🟢 Master launcher running (PID: $MASTER_PID)"
else
    echo "🔴 Master launcher not running"
fi

# Check individual GPU processes
echo ""
echo "📊 GPU PROCESS STATUS:"
echo "======================"
GPU_PROCESSES=0
for gpu in {0..7}; do
    pid=$(pgrep -f "sweep_scale_gpu${gpu}.sh" | head -1)
    if [[ -n "$pid" ]]; then
        echo "GPU $gpu: 🟢 Running (PID: $pid)"
        ((GPU_PROCESSES++))
    else
        echo "GPU $gpu: 🔴 Not running"
    fi
done

echo ""
echo "Active GPU processes: $GPU_PROCESSES/8"

# Check GPU utilization
echo ""
echo "🚀 GPU UTILIZATION:"
echo "=================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    while IFS=', ' read -r gpu_id name util mem_used mem_total temp; do
        mem_percent=$(( mem_used * 100 / mem_total ))
        echo "GPU $gpu_id: ${util}% util, ${mem_percent}% mem (${mem_used}/${mem_total}MB), ${temp}°C"
    done
else
    echo "nvidia-smi not available"
fi

# Check recent log activity
echo ""
echo "📋 RECENT LOG ACTIVITY:"
echo "======================="
latest_master_log=$(ls -t scale_sweep_master_*.log 2>/dev/null | head -1)
if [[ -n "$latest_master_log" ]]; then
    echo "Master log: $latest_master_log"
    if [[ -f "$latest_master_log" ]]; then
        echo "Last 3 lines:"
        tail -3 "$latest_master_log" | sed 's/^/  /'
    fi
else
    echo "No master log found"
fi

# Check for completed runs in results directory
echo ""
echo "📁 COMPLETED RUNS:"
echo "=================="
if [[ -d "results" ]]; then
    completed_dirs=$(find results -name "scale_all_subjects_*" -type d 2>/dev/null | wc -l)
    echo "Result directories found: $completed_dirs"
    
    # Show recent results
    recent_results=$(find results -name "scale_all_subjects_*" -type d -newermt "1 hour ago" 2>/dev/null | wc -l)
    echo "Recent results (last hour): $recent_results"
    
    # Show progress by model if possible
    echo ""
    echo "Progress by model:"
    for model in "DeepSeek-Qwen-1.5B" "DeepSeek-Qwen-14B" "DeepSeek-Llama-8B" "L1-Qwen-1.5B-Max"; do
        model_results=$(find results -name "*${model}*" -type d 2>/dev/null | wc -l)
        echo "  $model: $model_results runs"
    done
else
    echo "No results directory found yet"
fi

# Estimated completion
if [[ $GPU_PROCESSES -gt 0 ]]; then
    echo ""
    echo "⏱️  ESTIMATED PROGRESS:"
    echo "======================"
    
    # Find oldest GPU log to estimate start time
    oldest_gpu_log=$(ls -t scale_sweep_gpu*_*.log 2>/dev/null | tail -1)
    if [[ -n "$oldest_gpu_log" ]]; then
        # Extract timestamp from filename (YYYYMMDD_HHMMSS format)
        timestamp=$(basename "$oldest_gpu_log" | grep -o '[0-9]\{8\}_[0-9]\{6\}')
        if [[ -n "$timestamp" ]]; then
            start_time=$(date -d "${timestamp:0:8} ${timestamp:9:2}:${timestamp:11:2}:${timestamp:13:2}" +%s 2>/dev/null)
            current_time=$(date +%s)
            if [[ -n "$start_time" ]]; then
                elapsed=$((current_time - start_time))
                echo "Elapsed time: ${elapsed}s ($(($elapsed / 60))m $(($elapsed % 60))s)"
                
                # Rough progress estimate based on completed results
                if [[ $completed_dirs -gt 0 && $elapsed -gt 0 ]]; then
                    runs_per_second=$(echo "scale=4; $completed_dirs / $elapsed" | bc -l 2>/dev/null || echo "0")
                    if [[ -n "$runs_per_second" && "$runs_per_second" != "0" ]]; then
                        remaining_runs=$((420 - completed_dirs))
                        eta=$(echo "scale=0; $remaining_runs / $runs_per_second" | bc -l 2>/dev/null || echo "unknown")
                        if [[ "$eta" != "unknown" ]]; then
                            echo "Estimated remaining: ${eta}s ($(($eta / 60))m $(($eta % 60))s)"
                            completion_time=$(date -d "+${eta} seconds" "+%H:%M:%S")
                            echo "Estimated completion: $completion_time"
                        fi
                    fi
                fi
            fi
        fi
    fi
fi

echo ""
echo "Run './check_sweep_status.sh' again to refresh status"
echo "Use 'tail -f scale_sweep_master_*.log' to follow master log"
echo "Use 'tail -f scale_sweep_gpu{0-7}_*.log' to follow individual GPU logs"
