#!/bin/bash
# Test-time scaling evaluation script
# Usage: ./tt_scale.sh [task] [model_size] [gpu_id]
#   ./tt_scale.sh                    # meeting task, 8b model, auto GPU
#   ./tt_scale.sh calendar 14b       # calendar task, 14b model, auto GPU  
#   ./tt_scale.sh trip 1.5b 2        # trip task, 1.5b model, GPU 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

if command -v yq &> /dev/null; then
    BASE_DIR="$(yq eval '.outputs.base_dir' "$REPO_ROOT/files/benchmarks.yaml" 2>/dev/null || echo "data/")"
    PLANNER_SUBDIR="$(yq eval '.outputs.subdirs.agentic' "$REPO_ROOT/files/benchmarks.yaml" 2>/dev/null || echo "planner/")"
    OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-$REPO_ROOT/${BASE_DIR}${PLANNER_SUBDIR}server/scaling}"
else
    OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-$REPO_ROOT/data/planner/server/scaling}"
fi

CONFIG="$SCRIPT_DIR/../configs/np_scaling.yaml"

# Load model configuration
MODELS_CONF="$SCRIPT_DIR/models.conf"
if [[ -f "$MODELS_CONF" ]]; then
    source "$MODELS_CONF"
fi

# Load GPU configuration
GPU_CONF="$SCRIPT_DIR/gpu.conf"
if [[ -f "$GPU_CONF" ]]; then
    source "$GPU_CONF"
fi

# Parse CLI arguments
TASK=${1:-"meeting"}
MODEL_SIZE=${2:-"8b"}
GPU_ID=${3:-""}

# Get model from config
MODEL_SHORT="${MODEL_SIZE,,}"
case "$MODEL_SHORT" in
  14b)
    MODEL="$MODEL_14b" ;;
  8b)
    MODEL="$MODEL_8b" ;;
  1.5b|1_5b|1.5)
    MODEL="$MODEL_1_5b" ; MODEL_SHORT="1.5b" ;;
  *)
    echo "Error: Unsupported MODEL_SIZE '${MODEL_SIZE}'. Use: 14b | 8b | 1.5b" >&2 ; exit 1 ;;
esac

if [[ -z "$MODEL" ]]; then
    echo "Error: Model not defined in models.conf for size: $MODEL_SHORT" >&2 ; exit 1
fi

# Get GPU assignment if not specified
if [[ -z "$GPU_ID" ]]; then
    MODEL_VAR=$(echo "$MODEL_SHORT" | sed 's/\./_/g')
    TASK_GPU_VAR="GPU_MAP_${MODEL_VAR}_${TASK}"
    GPU_ID=${!TASK_GPU_VAR:-0}
fi

# Sample counts to sweep
SAMPLE_COUNTS=(1 2 4 8 16)

# Set output directory
OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_SHORT}/${TASK}"

mkdir -p "$OUTPUT_DIR"

echo "Starting scaling sweep for ${TASK} with ${MODEL} (${MODEL_SHORT})"
echo "GPU: ${GPU_ID}"
echo "Testing sample counts: ${SAMPLE_COUNTS[@]}"
echo "Output: ${OUTPUT_DIR}"

for num_samples in "${SAMPLE_COUNTS[@]}"; do
    echo ""
    echo "=== Testing ${num_samples} samples ==="
    
    TMP_CONFIG="/tmp/np_scaling_${num_samples}.yaml"
    sed "s/num_samples: 8/num_samples: ${num_samples}/" ${CONFIG} > ${TMP_CONFIG}
    
    timestamp=$(date +"%Y%m%d_%H%M%S")
    sample_output_dir="${OUTPUT_DIR}/samples_${num_samples}_${timestamp}"
    
    # Run evaluation with GPU assignment
    CUDA_VISIBLE_DEVICES=${GPU_ID} python "$SCRIPT_DIR/../planner.py" \
        --task ${TASK} \
        --model ${MODEL} \
        --config ${TMP_CONFIG} \
        --output ${sample_output_dir}
    
    # Clean up
    rm ${TMP_CONFIG}
    
    echo "Completed ${num_samples} samples"
done

echo ""
echo "Scaling sweep completed!"
echo "Results saved in: ${OUTPUT_DIR}"