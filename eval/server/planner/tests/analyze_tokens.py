#!/usr/bin/env python3
"""Token Analysis Script for Natural-Plan Datasets

Analyzes token counts for prompts and expected outputs across all datasets
and tokenizers to inform budget-limited and test-time scaling configurations.

Usage:
    python analyze_tokens.py --models deepseek-ai/DeepSeek-R1-Distill-Llama-8B deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    python analyze_tokens.py --models all --output token_analysis.csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TokenStats:
    """Token statistics for a single example."""
    task: str
    example_id: str
    model_name: str
    prompt_tokens: int
    golden_tokens: int
    total_tokens: int

# Common model configurations
DEFAULT_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
]

TASKS = ["trip", "meeting", "calendar"]
TASK_FILES = {
    "trip": "trip_planning.json",
    "meeting": "meeting_planning.json", 
    "calendar": "calendar_scheduling.json",
}

def load_dataset(task: str, data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load a single dataset."""
    file_path = data_dir / TASK_FILES[task]
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    logger.info(f"Loading {task} dataset from {file_path}")
    with open(file_path) as f:
        return json.load(f)

def get_golden_text(task: str, record: Dict[str, Any]) -> str:
    """Extract the golden answer text for different tasks."""
    golden = record.get("golden_plan", "")
    if task == "meeting" and isinstance(golden, list):
        return " ".join(golden)
    return str(golden)

def analyze_tokens_for_model(model_name: str, datasets: Dict[str, Dict], sample_size: int = None) -> List[TokenStats]:
    """Analyze token counts for a specific model/tokenizer."""
    logger.info(f"Loading tokenizer for {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {model_name}: {e}")
        return []
    
    stats = []
    
    for task, data in datasets.items():
        logger.info(f"Analyzing {task} dataset for {model_name}")
        
        items = list(data.items())
        if sample_size:
            items = items[:sample_size]
            logger.info(f"Sampling {sample_size} examples from {len(data)} total")
        
        for example_id, record in items:
            try:
                # Get prompt and golden answer
                prompt = record.get("prompt_5shot", "")
                golden_text = get_golden_text(task, record)
                
                # Tokenize
                prompt_tokens = len(tokenizer.encode(prompt))
                golden_tokens = len(tokenizer.encode(golden_text)) if golden_text else 0
                
                stats.append(TokenStats(
                    task=task,
                    example_id=example_id,
                    model_name=model_name.split('/')[-1],  # Use short name
                    prompt_tokens=prompt_tokens,
                    golden_tokens=golden_tokens,
                    total_tokens=prompt_tokens + golden_tokens
                ))
                
            except Exception as e:
                logger.warning(f"Failed to analyze {example_id}: {e}")
                continue
    
    return stats

def generate_summary_stats(stats: List[TokenStats]) -> Dict[str, Any]:
    """Generate summary statistics from token analysis."""
    if not stats:
        return {}
    
    # Group by task and model
    task_model_stats = {}
    
    for stat in stats:
        key = f"{stat.task}_{stat.model_name}"
        if key not in task_model_stats:
            task_model_stats[key] = {
                'task': stat.task,
                'model': stat.model_name,
                'prompt_tokens': [],
                'golden_tokens': [],
                'total_tokens': []
            }
        
        task_model_stats[key]['prompt_tokens'].append(stat.prompt_tokens)
        task_model_stats[key]['golden_tokens'].append(stat.golden_tokens)
        task_model_stats[key]['total_tokens'].append(stat.total_tokens)
    
    # Calculate summary statistics
    summaries = []
    for key, data in task_model_stats.items():
        prompt_tokens = data['prompt_tokens']
        golden_tokens = data['golden_tokens'] 
        total_tokens = data['total_tokens']
        
        summaries.append({
            'task': data['task'],
            'model': data['model'],
            'count': int(len(prompt_tokens)),
            'prompt_min': int(min(prompt_tokens)),
            'prompt_max': int(max(prompt_tokens)),
            'prompt_mean': int(sum(prompt_tokens) / len(prompt_tokens)),
            'golden_min': int(min(golden_tokens)) if golden_tokens else 0,
            'golden_max': int(max(golden_tokens)) if golden_tokens else 0,
            'golden_mean': int(sum(golden_tokens) / len(golden_tokens)) if golden_tokens else 0,
            'total_min': int(min(total_tokens)),
            'total_max': int(max(total_tokens)),
            'total_mean': int(sum(total_tokens) / len(total_tokens)),
        })
    
    return summaries

def main():
    parser = argparse.ArgumentParser(description="Analyze token counts for Natural-Plan datasets")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Model names to analyze (use 'all' for default set)")
    parser.add_argument("--data-dir", type=Path, default="eval/data",
                        help="Directory containing dataset files")
    parser.add_argument("--output", default="token_analysis.csv",
                        help="Output CSV file path")
    parser.add_argument("--summary", default="token_summary.csv", 
                        help="Summary statistics CSV file path")
    parser.add_argument("--sample-size", type=int, 
                        help="Sample only N examples per dataset (for quick analysis)")
    parser.add_argument("--tasks", nargs="+", default=TASKS,
                        choices=TASKS, help="Which tasks to analyze")
    
    args = parser.parse_args()
    
    if args.models == ["all"]:
        args.models = DEFAULT_MODELS
    
    # Load datasets
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    datasets = {}
    for task in args.tasks:
        try:
            datasets[task] = load_dataset(task, data_dir)
        except Exception as e:
            logger.error(f"Failed to load {task} dataset: {e}")
            return 1
    
    # Analyze tokens for each model
    all_stats = []
    for model_name in args.models:
        logger.info(f"Analyzing model: {model_name}")
        model_stats = analyze_tokens_for_model(model_name, datasets, args.sample_size)
        all_stats.extend(model_stats)
    
    if not all_stats:
        logger.error("No statistics generated")
        return 1
    
    # Save detailed results
    logger.info(f"Saving detailed results to {args.output}")
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task', 'example_id', 'model_name', 'prompt_tokens', 'golden_tokens', 'total_tokens'])
        
        for stat in all_stats:
            writer.writerow([
                stat.task, stat.example_id, stat.model_name,
                stat.prompt_tokens, stat.golden_tokens, stat.total_tokens
            ])
    
    # Generate and save summary statistics
    logger.info(f"Saving summary statistics to {args.summary}")
    summaries = generate_summary_stats(all_stats)
    
    with open(args.summary, 'w', newline='') as f:
        if summaries:
            writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
            writer.writeheader()
            writer.writerows(summaries)
    
    # Print quick summary to console
    print(f"\n✅ Analysis complete!")
    print(f"📊 Analyzed {len(all_stats)} examples across {len(args.models)} models")
    print(f"📝 Detailed results: {args.output}")
    print(f"📈 Summary statistics: {args.summary}")
    
    if summaries:
        print(f"\n🔍 Quick Overview:")
        for summary in summaries[:6]: 
            print(f"  {summary['task']} ({summary['model']}): "
                  f"prompt mean {summary['prompt_mean']:.0f} tokens, "
                  f"golden mean {summary['golden_mean']:.0f} tokens")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())