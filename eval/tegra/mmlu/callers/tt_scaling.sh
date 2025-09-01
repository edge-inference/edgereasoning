#!/usr/bin/env bash
# Scale evaluation sweep for Jetson Orin - Single GPU, 3 Models, 3 Seeds

# Base directories
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"

# GPU assignment (single GPU on Jetson)
export CUDA_VISIBLE_DEVICES=0
GPU_ID=0

# Resume functionality
RESUME_MODEL=${RESUME_MODEL:-""}
RESUME_SEED=${RESUME_SEED:-""}
RESUME_SAMPLE_SIZE=${RESUME_SAMPLE_SIZE:-""}
CHECKPOINT_FILE="scale_sweep_jetson_checkpoint.txt"

LOG_FILE="scale_sweep_jetson_$(date +%Y%m%d_%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

echo "========================================" 
echo "SCALE SWEEP JETSON ORIN STARTED: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Log file: $LOG_FILE"
if [[ -n "$RESUME_MODEL" ]]; then
    echo "RESUMING from model: $RESUME_MODEL"
    if [[ -n "$RESUME_SEED" ]]; then
        echo "Starting from seed: $RESUME_SEED"
        if [[ -n "$RESUME_SAMPLE_SIZE" ]]; then
            echo "Starting from sample size: $RESUME_SAMPLE_SIZE"
        fi
    fi
fi
echo "========================================"

# List of models to evaluate - read from models.txt
MODELS_FILE="$SCRIPT_DIR/../models.txt"
if [[ -f "$MODELS_FILE" ]]; then
    echo "* Reading models from: $MODELS_FILE"
    mapfile -t MODELS < "$MODELS_FILE"
    FILTERED_MODELS=()
    for model in "${MODELS[@]}"; do
        if [[ -n "$model" && ! "$model" =~ ^[[:space:]]*# ]]; then
            FILTERED_MODELS+=("$model")
        fi
    done
    MODELS=("${FILTERED_MODELS[@]}")
else
    echo "! models.txt not found at: $MODELS_FILE, using default models"
    MODELS=(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    )
fi

# 3 seeds for reproducibility
SEEDS=(42 1337 2023)

echo "Jetson Orin Configuration:"
echo "Models: ${MODELS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Token budget: 256"
echo "Sample sizes: 1 2 4 8 16 32"
echo "Questions per subject: 5 (Jetson optimized)"
echo "Max subjects per run: 30"

# Define token budget and sample sizes 
TOKEN_BUDGETS=(256)
SAMPLE_SIZES=(4)  
SUCCESSFUL_RUNS=()
FAILED_RUNS=()

# Resume logic
SKIP_UNTIL_MODEL=false
SKIP_UNTIL_SEED=false
SKIP_UNTIL_SAMPLE=false
if [[ -n "$RESUME_MODEL" ]]; then
    SKIP_UNTIL_MODEL=true
    if [[ -n "$RESUME_SEED" ]]; then
        SKIP_UNTIL_SEED=true
        if [[ -n "$RESUME_SAMPLE_SIZE" ]]; then
            SKIP_UNTIL_SAMPLE=true
        fi
    fi
fi

for MODEL_PATH in "${MODELS[@]}"; do
  echo "========================================"
  echo "==== JETSON: MODEL: ${MODEL_PATH} ===="
  echo "========================================"
  
  MODEL_NAME=$(basename "$MODEL_PATH")
  
  # Skip models until we reach the resume point
  if [[ "$SKIP_UNTIL_MODEL" == "true" ]]; then
    if [[ "$MODEL_PATH" == "$RESUME_MODEL" ]]; then
        SKIP_UNTIL_MODEL=false
        echo "==== JETSON: RESUMING at Model ${MODEL_PATH} ===="
    else
        echo "==== JETSON: SKIPPING Model ${MODEL_PATH} (resuming from ${RESUME_MODEL}) ===="
        continue
    fi
  fi
  
  for TOKEN_BUDGET in "${TOKEN_BUDGETS[@]}"; do
    echo "---- JETSON: Token Budget: ${TOKEN_BUDGET} ----"
    
    for SEED in "${SEEDS[@]}"; do
      # Skip seeds until we reach the resume point
      if [[ "$SKIP_UNTIL_SEED" == "true" ]]; then
        if [[ "$SEED" == "$RESUME_SEED" ]]; then
            SKIP_UNTIL_SEED=false
            echo "==== JETSON: RESUMING at Seed ${SEED} ===="
        else
            echo "==== JETSON: SKIPPING Seed ${SEED} (resuming from ${RESUME_SEED}) ===="
            continue
        fi
      else
        echo "==== JETSON: Seed ${SEED} ===="
      fi
      
      # Write checkpoint
      echo "$MODEL_PATH $SEED" > "$CHECKPOINT_FILE"
      
      for SAMPLES in "${SAMPLE_SIZES[@]}"; do
        # Skip sample sizes until we reach the resume point
        if [[ "$SKIP_UNTIL_SAMPLE" == "true" ]]; then
          if [[ "$SAMPLES" == "$RESUME_SAMPLE_SIZE" ]]; then
              SKIP_UNTIL_SAMPLE=false
              echo "-> JETSON: RESUMING from ${SAMPLES} samples..."
          else
              echo "-> JETSON: SKIPPING ${SAMPLES} samples (resuming from ${RESUME_SAMPLE_SIZE})..."
              continue
          fi
        else
          echo "-> JETSON: ${SAMPLES} samples..."
        fi
        
        SAMPLE_START_TIME=$(date +%s)
        
        export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
        
        TEMP_CONFIG="$SCRIPT_DIR/../configs/scale_temp_jetson_${TOKEN_BUDGET}.yaml"
        sed "s/min_new_tokens: [0-9]*/min_new_tokens: ${TOKEN_BUDGET}/g; s/max_new_tokens: [0-9]*/max_new_tokens: ${TOKEN_BUDGET}/g; s/MAX_TOKENS_PLACEHOLDER/${TOKEN_BUDGET}/g" "$SCRIPT_DIR/../configs/scale.yaml" > "$TEMP_CONFIG"
        
        if python "$SCRIPT_DIR/../scripts/tt_scaling.py" \
            --model "$MODEL_PATH" \
            --num-samples "$SAMPLES" \
            --token-budget "$TOKEN_BUDGET" \
            --seed "$SEED" \
            --max-subjects 30 \
            --config "$TEMP_CONFIG"; then
          
          echo "+ SCALE: Scale evaluation completed for ${SAMPLES} samples, ${TOKEN_BUDGET} tokens"
          
          rm -f "$TEMP_CONFIG"
          
          SAMPLE_END_TIME=$(date +%s)
          DURATION=$((SAMPLE_END_TIME - SAMPLE_START_TIME))
          SUCCESSFUL_RUNS+=("JETSON:${MODEL_NAME}:${SEED}:${SAMPLES}:${TOKEN_BUDGET} (${DURATION}s)")
          echo "+ SCALE: Completed ${SAMPLES} samples, ${TOKEN_BUDGET} tokens in ${DURATION}s"
          
          echo "+ Forcing memory cleanup..."
          python3 -c "
import gc
gc.collect()
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        print('GPU memory cleared')
except ImportError:
    pass
"
          
        else
            echo "✗ SCALE: Scale evaluation failed for ${SAMPLES} samples, ${TOKEN_BUDGET} tokens"
            FAILED_RUNS+=("JETSON:${MODEL_NAME}:${SEED}:${SAMPLES}:${TOKEN_BUDGET} (evaluation failed)")
            echo "! SCALE: Continuing with next sample size..."
            
            rm -f "$TEMP_CONFIG"
            
            echo "Forcing memory cleanup after failure..."
            python3 -c "
import gc
gc.collect()
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        print('GPU memory cleared after failure')
except ImportError:
    pass
"
        fi
      done
    done
  done
done

rm -f "$CHECKPOINT_FILE"

echo ""
echo "========================================"
echo "JETSON SCALE SWEEP COMPLETED: $(date)"
echo "========================================"
echo "SUCCESSFUL RUNS (${#SUCCESSFUL_RUNS[@]}): ${SUCCESSFUL_RUNS[*]}"
echo "FAILED RUNS (${#FAILED_RUNS[@]}): ${FAILED_RUNS[*]}"
echo "Log saved to: $LOG_FILE"

# Final summary
TOTAL_RUNS=$((${#SUCCESSFUL_RUNS[@]} + ${#FAILED_RUNS[@]}))
echo ""
echo "≡ FINAL SUMMARY:"
echo "Total Models: ${#MODELS[@]}"
echo "Total Seeds: ${#SEEDS[@]}"
echo "Total Token Budgets: ${#TOKEN_BUDGETS[@]}"
echo "Total Sample Sizes: ${#SAMPLE_SIZES[@]}"
echo "Total Runs: $TOTAL_RUNS"
echo "Successful: ${#SUCCESSFUL_RUNS[@]}"
echo "Failed: ${#FAILED_RUNS[@]}"
echo "Success Rate: $(( ${#SUCCESSFUL_RUNS[@]} * 100 / TOTAL_RUNS ))%"

# Clear checkpoint on completion
rm -f "$CHECKPOINT_FILE"

if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo "★ SCALE: All scale evaluations completed successfully!"
    exit 0
else
    echo "! SCALE: Sweep completed with ${#FAILED_RUNS[@]} failures"
    exit 1
fi