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

# Base directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CONFIG_DIR="$SCRIPT_DIR/../configs"
BUDGET_CONFIG="${BASE_CONFIG_DIR}/np_budget.yaml"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-./results/budget}"

# Python environment to activate (same as bypass.sh)
VENV_ACT="/home/lab/modfi/SkyThought/venvsky/bin/activate"

# 1️⃣ Parse CLI ---------------------------------------------------------------
TASK_ARG=${1:-both}   # meeting | calendar | trip | all | both
MODEL_SIZE=${2:-14b}  # 14b | 8b | 1.5b

# 2️⃣ Map model-size shortcut → full HF repo ----------------------------------
case "${MODEL_SIZE,,}" in
  14b)
    MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" ; MODEL_SHORT="14b" ;;
  8b)
    MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" ; MODEL_SHORT="8b" ;;
  1.5b|1_5b|1.5)
    MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" ; MODEL_SHORT="1.5b" ;;
  *)
    echo "❌ Unsupported MODEL_SIZE '${MODEL_SIZE}'. Use: 14b | 8b | 1.5b" >&2 ; exit 1 ;;
esac

# Set output directory now that MODEL_SHORT is defined
OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_SHORT}"

# 3️⃣ Static GPU assignment table --------------------------------------------
#   Key = "${MODEL_SHORT},${TASK}" → value = GPU-ID
declare -A GPU_MAP
GPU_MAP["14b,trip"]=0     ; GPU_MAP["14b,meeting"]=1     ; GPU_MAP["14b,calendar"]=2
GPU_MAP["8b,trip"]=3     ; GPU_MAP["8b,meeting"]=4     ; GPU_MAP["8b,calendar"]=5
GPU_MAP["1.5b,trip"]=5   ; GPU_MAP["1.5b,meeting"]=6   ; GPU_MAP["1.5b,calendar"]=7

MEETING_GPU=${GPU_MAP["${MODEL_SHORT},meeting"]}
CALENDAR_GPU=${GPU_MAP["${MODEL_SHORT},calendar"]}
TRIP_GPU=${GPU_MAP["${MODEL_SHORT},trip"]}

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
if [ ! -f $VENV_ACT ]; then
  echo '❌ Virtual env activation script not found: $VENV_ACT' >&2; exit 1
fi
source $VENV_ACT
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
