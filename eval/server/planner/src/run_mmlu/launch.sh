#!/bin/bash

# No-Reasoning MMLU Evaluation Launch Script
# Uses 4 GPUs and logs all output

echo "🚀 Starting No-Reasoning MMLU Evaluation..."
echo "📅 Start time: $(date)"
echo "🔧 Configuration: 4 GPUs, No-Reasoning mode"
echo "📝 Logging to: .run.log"
echo ""

# Run the evaluation with 4 GPUs and log everything
python all_noreason.py --tensor-parallel-size 1 2>&1 | tee -a .run.log

echo ""
echo "✅ Evaluation completed at: $(date)"
echo "📋 Check .run.log for full details"
