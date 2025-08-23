#!/bin/bash
GPUS="0,1,2"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

python -u bench/planner_eval.py --task all --model "$MODEL" --gpus "$GPUS"
