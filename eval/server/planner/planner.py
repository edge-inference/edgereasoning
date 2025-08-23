#!/usr/bin/env python3
"""Convenience entry-point for the Natural-Plan benchmark.

Example usage
-------------
# Evaluate trip-planning task with default config
python planner_eval.py --task trip --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

# Evaluate all three tasks (trip, meeting, calendar)
python planner_eval.py --task all --model my/local/model --base-config-dir bench/configs

Notes
-----
• Per-task YAML files must exist in *base_config_dir* and follow the naming
  `np_<task>.yaml` (e.g. `np_trip.yaml`).
• The config can be overridden with `--config` if you only run one task.
"""
from __future__ import annotations

import argparse
import os, sys, subprocess

# Append bench/src to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluators.plan_evaluator_base import PlanEvaluatorBase
from src.evaluators.plan_evaluator_scaling import PlanEvaluatorScaling
from src.evaluators.plan_evaluator_direct import PlanEvaluatorDirect
from src.evaluators.plan_evaluator_budget import PlanEvaluatorBudget
import yaml

TASKS = ["trip", "meeting", "calendar"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Natural-Plan evaluation")
    p.add_argument("--task", choices=TASKS + ["all"], default="trip",
                   help="Which task to run (trip / meeting / calendar / all)")
    p.add_argument("--model", required=True, help="Model path or HF repo id")
    p.add_argument("--gpus", help="Comma-separated CUDA device IDs to use (e.g. '0,1,2')")
    p.add_argument("--gpu", help="Single GPU id for this subprocess (internal)")
    p.add_argument("--config", help="Path to YAML config (single-task only)")
    p.add_argument("--base-config-dir", default="bench/configs",
                   help="Directory with np_<task>.yaml configs")
    p.add_argument("--output", default="./results",
                   help="Base output directory")
    return p.parse_args()

def run_task(task: str, model_path: str, config_path: str, output_root: str, gpu_id: str | None = None):
    # Pin GPU if requested
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"[planner_eval] Using GPU {gpu_id} for task {task}")
    
    # Determine which evaluator to use based on config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check config type based on exact name
    config_name = config.get('name', '').lower()
    is_direct_config = config_name == 'plan_direct'
    is_budget_config = config_name == 'np_budget'
    is_scaling_config = config_name == 'np_scaling'
    
    # Select evaluator based on config name
    if is_scaling_config:
        print(f"[planner_eval] Using PlanEvaluatorScaling")
        evaluator = PlanEvaluatorScaling(config_path, task)
    elif is_direct_config:
        print(f"[planner_eval] Using PlanEvaluatorDirect (direct mode)")
        evaluator = PlanEvaluatorDirect(config_path, task)
    elif is_budget_config:
        print(f"[planner_eval] Using PlanEvaluatorBudget (no-reasoning mode)")
        evaluator = PlanEvaluatorBudget(config_path, task)
    else:
        print(f"[planner_eval] Using PlanEvaluatorBase (default)")
        evaluator = PlanEvaluatorBase(config_path, task)
    
    os.makedirs(output_root, exist_ok=True)
    result = evaluator.evaluate_task(model_path=model_path, output_dir=output_root)
    evaluator.print_summary(result)


def main() -> None:
    args = parse_args()

    if args.task != "all" and args.config is None:
        # Derive config automatically
        cfg = os.path.join(args.base_config_dir, f"np_{args.task}.yaml")
        if not os.path.exists(cfg):
            raise FileNotFoundError(f"Config not found: {cfg}")
        args.config = cfg

    if args.task == "all":
        # Run each task in a separate subprocess to avoid absl.flags duplicates
        for task in TASKS:
            # Determine GPU for this task
            gpu_list = args.gpus.split(',') if args.gpus else ["0", "1", "2"]
            gpu_id = gpu_list[TASKS.index(task) % len(gpu_list)]
            cmd = [
                sys.executable,
                __file__,
                "--task", task,
                "--model", args.model,
                "--base-config-dir", args.base_config_dir,
                "--output", args.output,
                "--gpu", gpu_id,
            ]
            print("\n===== Running", task, "in subprocess =====")
            subprocess.run(cmd, check=True)
        return
    else:
        gpu_single = args.gpu or (args.gpus.split(',')[0] if args.gpus else None)
        run_task(args.task, args.model, args.config, args.output, gpu_single)


if __name__ == "__main__":
    main() 