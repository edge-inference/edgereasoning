#!/usr/bin/env bash
# Scale evaluation sweep for GPU 0 - DeepSeek-Qwen-1.5B, Seeds 0-2

# GPU assignment
export CUDA_VISIBLE_DEVICES=0
GPU_ID=0

# Resume functionality
RESUME_SEED=${RESUME_SEED:-""}
RESUME_SAMPLE_SIZE=${RESUME_SAMPLE_SIZE:-""}
CHECKPOINT_FILE="scale_sweep_gpu${GPU_ID}_checkpoint.txt"

# Setup logging
LOG_FILE="scale_sweep_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

echo "========================================" 
echo "SCALE SWEEP GPU ${GPU_ID} STARTED: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Log file: $LOG_FILE"
if [[ -n "$RESUME_SEED" ]]; then
    echo "RESUMING from seed: $RESUME_SEED"
    if [[ -n "$RESUME_SAMPLE_SIZE" ]]; then
        echo "Starting from sample size: $RESUME_SAMPLE_SIZE"
    fi
fi
echo "========================================"

# Model configuration - GPU 0 handles DeepSeek-Qwen-1.5B
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)
NUM_QUESTIONS=${NUM_QUESTIONS:-10}

# GPU 0 handles seeds 0-2 (3 seeds)
SEEDS=(0 1073741823 2147483647)

echo "GPU ${GPU_ID} assigned:"
echo "Models: ${MODELS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Token budgets: 512"
echo "Sample sizes: 1 2 4 8 16 32 64"
echo "Questions per subject: $NUM_QUESTIONS"

# Define token budgets and sample sizes
TOKEN_BUDGETS=(512)
SAMPLE_SIZES=(1 2 4 8 16 32 64)
SUCCESSFUL_RUNS=()
FAILED_RUNS=()

# Resume logic
SKIP_UNTIL_RESUME=false
SKIP_UNTIL_SAMPLE=false
if [[ -n "$RESUME_SEED" ]]; then
    SKIP_UNTIL_RESUME=true
    if [[ -n "$RESUME_SAMPLE_SIZE" ]]; then
        SKIP_UNTIL_SAMPLE=true
    fi
fi

for MODEL_PATH in "${MODELS[@]}"; do
  echo "========================================"
  echo "==== GPU ${GPU_ID}: MODEL: ${MODEL_PATH} ===="
  echo "========================================"
  
  MODEL_NAME=$(basename "$MODEL_PATH")
  
  for TOKEN_BUDGET in "${TOKEN_BUDGETS[@]}"; do
    echo "---- GPU ${GPU_ID}: Token Budget: ${TOKEN_BUDGET} ----"
    
    for SEED in "${SEEDS[@]}"; do
      # Skip seeds until we reach the resume point
      if [[ "$SKIP_UNTIL_RESUME" == "true" ]]; then
        if [[ "$SEED" == "$RESUME_SEED" ]]; then
            SKIP_UNTIL_RESUME=false
            echo "==== GPU ${GPU_ID}: RESUMING at Seed ${SEED} ===="
        else
            echo "==== GPU ${GPU_ID}: SKIPPING Seed ${SEED} (resuming from ${RESUME_SEED}) ===="
            continue
        fi
      else
        echo "==== GPU ${GPU_ID}: Seed ${SEED} ===="
      fi
      
      # Write checkpoint
      echo "$SEED" > "$CHECKPOINT_FILE"
      
      for SAMPLES in "${SAMPLE_SIZES[@]}"; do
        # Skip sample sizes until we reach the resume point
        if [[ "$SKIP_UNTIL_SAMPLE" == "true" ]]; then
          if [[ "$SAMPLES" == "$RESUME_SAMPLE_SIZE" ]]; then
              SKIP_UNTIL_SAMPLE=false
              echo "-> GPU ${GPU_ID}: RESUMING from ${SAMPLES} samples..."
          else
              echo "-> GPU ${GPU_ID}: SKIPPING ${SAMPLES} samples (resuming from ${RESUME_SAMPLE_SIZE})..."
              continue
          fi
        else
          echo "-> GPU ${GPU_ID}: ${SAMPLES} samples..."
        fi
        
        SAMPLE_START_TIME=$(date +%s)
        
        # Set config for this number of samples
        export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
        
        # Create temporary config with token budget
        TEMP_CONFIG="configs/scale_temp_gpu${GPU_ID}_${TOKEN_BUDGET}.yaml"
        sed "s/min_new_tokens: [0-9]*/min_new_tokens: ${TOKEN_BUDGET}/g; s/max_new_tokens: [0-9]*/max_new_tokens: ${TOKEN_BUDGET}/g; s/MAX_TOKENS_PLACEHOLDER/${TOKEN_BUDGET}/g" configs/scale.yaml > "$TEMP_CONFIG"
        
        # Run scale evaluation
        if python all_scale.py \
            --model "$MODEL_PATH" \
            --num-samples "$SAMPLES" \
            --token-budget "$TOKEN_BUDGET" \
            --seed "$SEED" \
            --config "$TEMP_CONFIG"; then
          
          echo -e "\033[1;32m✓ GPU ${GPU_ID}: Scale evaluation completed for ${SAMPLES} samples, ${TOKEN_BUDGET} tokens\033[0m"
          
          # Clean up temp config
          rm -f "$TEMP_CONFIG"
          
          SAMPLE_END_TIME=$(date +%s)
          DURATION=$((SAMPLE_END_TIME - SAMPLE_START_TIME))
          SUCCESSFUL_RUNS+=("GPU${GPU_ID}:${MODEL_NAME}:${SEED}:${SAMPLES}:${TOKEN_BUDGET} (${DURATION}s)")
          echo -e "\033[1;32m✓ GPU ${GPU_ID}: Completed ${SAMPLES} samples, ${TOKEN_BUDGET} tokens in ${DURATION}s\033[0m"
          
        else
            echo -e "\033[1;31m✗ GPU ${GPU_ID}: Scale evaluation failed for ${SAMPLES} samples, ${TOKEN_BUDGET} tokens\033[0m"
            FAILED_RUNS+=("GPU${GPU_ID}:${MODEL_NAME}:${SEED}:${SAMPLES}:${TOKEN_BUDGET} (evaluation failed)")
            echo -e "\033[1;33m⚠ GPU ${GPU_ID}: Continuing with next sample size...\033[0m"
            
            # Clean up temp config on failure too
            rm -f "$TEMP_CONFIG"
        fi
      done
    done
  done
done

# Clear checkpoint on successful completion
rm -f "$CHECKPOINT_FILE"

echo ""
echo "========================================"
echo "GPU ${GPU_ID} SCALE SWEEP COMPLETED: $(date)"
echo "========================================"
echo "SUCCESSFUL RUNS (${#SUCCESSFUL_RUNS[@]}): ${SUCCESSFUL_RUNS[*]}"
echo "FAILED RUNS (${#FAILED_RUNS[@]}): ${FAILED_RUNS[*]}"
echo "Log saved to: $LOG_FILE"

if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo -e "\033[1;32m🎉 GPU ${GPU_ID}: All scale evaluations completed successfully!\033[0m"
    exit 0
else
    echo -e "\033[1;33m⚠ GPU ${GPU_ID}: Sweep completed with ${#FAILED_RUNS[@]} failures\033[0m"
    exit 1
fi
