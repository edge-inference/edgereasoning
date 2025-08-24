#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# np_budget.sh – launch Natural-Plan *budget-limited* evaluation jobs in detached
#                 GNU screen sessions (one per task).
#
# Usage examples
# -----------------------------------------------------------------------------
#   ./np_budget.sh                 # run meeting + calendar in budget mode (14 B)
#   ./np_budget.sh meeting         # run only meeting task  (14 B)
#   ./np_budget.sh meeting 8b      # run meeting task with 8 B model
#   ./np_budget.sh all 1.5b        # run all tasks    with 1 .5 B model
#
# Notes
# -----
# • Each task launches in its own detached screen session so you can
#   re-attach (`screen -r <session>`) and tail the logs live.
# • GPU allocation is static (see GPU_MAP below) but easy to tweak.
# • The script always uses bench/configs/np_budget.yaml and therefore injects the
#   token-budget prompts automatically via planner_eval.py → NaturalPlanEvaluator.
# -----------------------------------------------------------------------------

set -euo pipefail

# Get repository root and read from files/ directly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Read output directory from files/benchmarks.yaml
if command -v yq &> /dev/null; then
    BASE_DIR="$(yq eval '.outputs.base_dir' "$REPO_ROOT/files/benchmarks.yaml" 2>/dev/null || echo "data/")"
    PLANNER_SUBDIR="$(yq eval '.outputs.subdirs.agentic' "$REPO_ROOT/files/benchmarks.yaml" 2>/dev/null || echo "planner/")"
    OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-$REPO_ROOT/${BASE_DIR}${PLANNER_SUBDIR}server/budget}"
else
    # Fallback to hardcoded path if yq not available
    OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-$REPO_ROOT/data/planner/server/budget}"
fi

BASE_CONFIG_DIR="$SCRIPT_DIR/../configs"
BUDGET_CONFIG="${BASE_CONFIG_DIR}/np_budget.yaml"

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

# Check if virtual environment is activated (allow unset)
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Warning: No virtual environment detected. Please run 'source .venv/bin/activate' first."
fi

# 1️⃣ Parse CLI ---------------------------------------------------------------
TASK_ARG=${1:-both}   # meeting | calendar | trip | all | both
MODEL_SIZE=${2:-14b}  # 14b | 8b | 1.5b

# 2️⃣ Get model from config --------------------------------------------------
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

OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_SHORT}"

# 3️⃣ Get GPU assignments from config ----------------------------------------
MODEL_VAR=$(echo "$MODEL_SHORT" | sed 's/\./_/g')

MEETING_GPU_VAR="GPU_MAP_${MODEL_VAR}_meeting"
CALENDAR_GPU_VAR="GPU_MAP_${MODEL_VAR}_calendar"  
TRIP_GPU_VAR="GPU_MAP_${MODEL_VAR}_trip"

MEETING_GPU=${!MEETING_GPU_VAR:-0}
CALENDAR_GPU=${!CALENDAR_GPU_VAR:-1}
TRIP_GPU=${!TRIP_GPU_VAR:-2}

# 4️⃣ Decide which tasks to run ----------------------------------------------
RUN_MEETING=false; RUN_CALENDAR=false; RUN_TRIP=false
case "${TASK_ARG,,}" in
  meeting)   RUN_MEETING=true ;;
  calendar)  RUN_CALENDAR=true ;;
  trip)      RUN_TRIP=true ;;
  all)       RUN_MEETING=true ; RUN_CALENDAR=true ; RUN_TRIP=true ;;
  both|*)    RUN_MEETING=true ; RUN_CALENDAR=true ;; 
esac

# 5️⃣ Helper: GPU availability check -----------------------------------------
check_gpu() {
  local gpu=$1; local task=$2
  local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ' || echo 0)
  if [[ "$util" =~ ^[0-9]+$ ]] && (( util > 20 )); then
    echo "⚠️  GPU $gpu busy ($util %) – skip $task" >&2 ; return 1
  fi
  return 0
}

# 6️⃣ Launch helper -----------------------------------------------------------
launch_task() {
  local task=$1; local gpu=$2; local session=$3; local log=$4
  local cfg="$BUDGET_CONFIG"

  check_gpu "$gpu" "$task" || return

  local timestamp=$(date +"%Y%m%d_%H%M%S")
  local task_out_dir="${OUTPUT_DIR}/${task}_${timestamp}"
  mkdir -p "$task_out_dir"
  
  local full_log_path="${task_out_dir}/${log}"

  # kill stale session
  if screen -ls | grep -q "\.${session}[[:space:]]"; then
    echo "Killing old screen session $session" ; screen -S "$session" -X quit || true ; sleep 1
  fi

  echo "▶️  Starting $task (GPU $gpu) → screen:$session"
  echo "📁 Output: $task_out_dir"
  echo "📝 Log: $full_log_path"
  
  screen -dmS "$session" bash -c "
# Activate virtual environment
cd '$REPO_ROOT'
if [[ -f '.venv/bin/activate' ]]; then
    source .venv/bin/activate
else
    echo 'Warning: .venv/bin/activate not found. Run make venv first.'
fi
cd - > /dev/null
export CUDA_VISIBLE_DEVICES=$gpu
python -u $SCRIPT_DIR/../planner.py \
  --task $task \
  --model '$MODEL' \
  --gpu $gpu \
  --config '$cfg' \
  --output '$task_out_dir' 2>&1 | tee -a '$full_log_path'"

}

# 7️⃣ Start requested tasks ---------------------------------------------------
mkdir -p "$OUTPUT_DIR"

if $RUN_MEETING; then
  launch_task meeting "$MEETING_GPU" "np_budget_meeting_${MODEL_SHORT}" "np_budget_meeting_${MODEL_SHORT}.log"
fi
if $RUN_CALENDAR; then
  launch_task calendar "$CALENDAR_GPU" "np_budget_calendar_${MODEL_SHORT}" "np_budget_calendar_${MODEL_SHORT}.log"
fi
if $RUN_TRIP; then
  launch_task trip "$TRIP_GPU" "np_budget_trip_${MODEL_SHORT}" "np_budget_trip_${MODEL_SHORT}.log"
fi

# 8️⃣ Summary -----------------------------------------------------------------
echo "\n=== np_budget summary ==="
echo "Model: $MODEL (${MODEL_SHORT})"
$RUN_MEETING   && echo "Meeting  → GPU $MEETING_GPU  screen np_budget_meeting_${MODEL_SHORT}"
$RUN_CALENDAR  && echo "Calendar → GPU $CALENDAR_GPU screen np_budget_calendar_${MODEL_SHORT}"
$RUN_TRIP      && echo "Trip     → GPU $TRIP_GPU     screen np_budget_trip_${MODEL_SHORT}"
