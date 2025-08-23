#!/bin/bash
# Sample sweep script for test-time scaling evaluation
# Usage: ./sweep_scaling.sh <model_path> <task> <gpu_id>

MODEL_PATH=${1:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
TASK=${2:-"meeting"}
GPU_ID=${3:-"6"}

# Sample counts to sweep
SAMPLE_COUNTS=(1 2 4 8 16)

# Base directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/../configs/np_scaling.yaml"

echo "🚀 Starting scaling sweep for ${TASK} with ${MODEL_PATH}"
echo "📊 Testing sample counts: ${SAMPLE_COUNTS[@]}"

for num_samples in "${SAMPLE_COUNTS[@]}"; do
    echo ""
    echo "=== Testing ${num_samples} samples ==="
    
    # Create temporary config with specific sample count
    TMP_CONFIG="/tmp/np_scaling_${num_samples}.yaml"
    sed "s/num_samples: 8/num_samples: ${num_samples}/" ${CONFIG} > ${TMP_CONFIG}
    
    # Run evaluation with GPU assignment
    CUDA_VISIBLE_DEVICES=${GPU_ID} python "$SCRIPT_DIR/../planner.py" \
        --task ${TASK} \
        --model ${MODEL_PATH} \
        --config ${TMP_CONFIG} \
        --output ./results/scaling_sweep
    
    # Clean up
    rm ${TMP_CONFIG}
    
    echo "✅ Completed ${num_samples} samples"
done

echo ""
echo "🎯 Scaling sweep completed!"
echo "📁 Results saved in: ./results/scaling_sweep/"