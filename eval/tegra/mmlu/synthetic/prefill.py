#!/usr/bin/env python3
"""
Prefill Script for Input/Output Length Experiments

Runs inference for selected questions with controlled input and output lengths.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List
from datetime import datetime
import gc, time
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_loaders.custom_loader import CustomLoader
from src.evaluators.base_evaluator import BaseEvaluator
from src.evaluators.budget_evaluator import BudgetEvaluator

sys.path.append('/workspace/edgereasoning')
from loaders.benchmarks import get_benchmark_config
from loaders.results import get_results_config

# LLAMA_INPUT_TOKEN_LIST = [1, 16, 32, 64, 96] + list(range(128, 4096 + 128, 128))
LLAMA_INPUT_TOKEN_LIST = list(range(2176, 4096 + 128, 128))
LLAMA_OUTPUT_TOKEN_LIST = [1]  

QWEN_INPUT_TOKEN_LIST = LLAMA_INPUT_TOKEN_LIST.copy()
QWEN_OUTPUT_TOKEN_LIST = [1] 

def is_qwen_model(model_name):
    """Check if the model is from the Qwen family."""
    qwen_indicators = ['qwen', 'Qwen', 'QWEN']
    return any(indicator in model_name for indicator in qwen_indicators)

def get_model_config(model_name):
    """Get the appropriate CSV file and token lists based on model family."""
    config = get_benchmark_config()
    results_config = get_results_config()
    family = config.detect_model_family(model_name) or 'llama'
    
    base_output_dir = results_config.get_synthetic_input_dir('prefill')
    
    csv_relative_path = config._config['datasets']['synthetic']['prefill'][family]
    repo_root = Path(__file__).resolve().parents[4]
    csv_path = (repo_root / csv_relative_path).resolve()
    
    if family == 'qwen':
        return {
            'csv_file': csv_path,
            'input_tokens': QWEN_INPUT_TOKEN_LIST,
            'output_tokens': QWEN_OUTPUT_TOKEN_LIST,
            'family': 'Qwen',
            'output_dir': str(base_output_dir)
        }
    else:
        return {
            'csv_file': csv_path,
            'input_tokens': LLAMA_INPUT_TOKEN_LIST,
            'output_tokens': LLAMA_OUTPUT_TOKEN_LIST,
            'family': 'Llama',
            'output_dir': str(base_output_dir)
        }

def extract_model_suffix(model_name):
    import re
    match = re.search(r'([\d.]+B|Max)$', model_name)
    return match.group(1) if match else model_name.split('/')[-1]


def main():
    parser = argparse.ArgumentParser(description='Controlled input/output sweep')
    parser.add_argument('--models_file', default='models.txt', help='File with model names')
    parser.add_argument('--config', default='configs/prefill.yaml', help='Config file')
    parser.add_argument('--output_dir', default='../../../../data/synthetic/gpu/prefill', help='Output directory')
    parser.add_argument('--warmup_runs', type=int, default=1, help='Number of warm-up predictions (not timed) before each measured run')
    args = parser.parse_args()

    pass

    with open(args.models_file) as f:
        model_names = [line.strip() for line in f if line.strip()]

    for model_name in model_names:
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()  
            torch.cuda.reset_peak_memory_stats()
            time.sleep(10)
            
        print("[INFO] Memory cleanup before loading model succeeded.")
        
        # Get model-specific configuration
        model_config = get_model_config(model_name)
        print(f"\n=== Model: {model_name} ===")
        print(f"Detected family: {model_config['family']}")
        print(f"Using CSV: {model_config['csv_file']}")
        print(f"Input tokens: {model_config['input_tokens']}")
        print(f"Output tokens: {model_config['output_tokens']}")
        
        if not os.path.exists(model_config['csv_file']):
            print(f"ERROR: CSV file not found: {model_config['csv_file']}")
            print(f"Please ensure the synthetic dataset for {model_config['family']} family exists")
            continue
            
        loader = CustomLoader(model_config['csv_file'])
        print("DEBUG: Available input token counts in loader:", sorted(set(q['input_tokens'] for q in loader.questions)))
        
        model_suffix = extract_model_suffix(model_name)
        model_out_dir = os.path.join(model_config['output_dir'], model_suffix)
        os.makedirs(model_out_dir, exist_ok=True)
        print(f"Results will be saved to: {model_out_dir}")
        
        evaluator = None
        
        for input_tokens in model_config['input_tokens']:
            if input_tokens == 1:
                # Synthetic 1-token question
                q = loader.build_custom_dataset([1])[0]
                config_path = 'configs/custom.yaml'
            else:
                q = loader.get_question_by_input_tokens(input_tokens)
                config_path = args.config
            if not q:
                print(f"No question found for input_tokens={input_tokens}, skipping.")
                continue
            subj = getattr(q, 'subject', 'unknown')
            
            if evaluator is None:
                evaluator = BaseEvaluator(config_path)
            
            print(f"\n=== Running input_tokens={input_tokens} (question_id={q.question_id}, subject={subj}) ===")
            for out_tok in model_config['output_tokens']:
                evaluator.config.model['max_tokens'] = out_tok
                run_name = f"{subj}_in{input_tokens}_out{out_tok}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"  -> Output tokens: {out_tok}")
                try:
                    result = evaluator.evaluate_subject_on_questions(
                        model_name, [q], model_out_dir, run_name=run_name, min_tokens=out_tok, warmup_runs=args.warmup_runs
                    )
                    print(f"    Accuracy: {result.accuracy:.3f}, Time: {result.avg_time_per_question:.1f} ms")
                except Exception as e:
                    print(f"    [ERROR] Failed for input_tokens={input_tokens}, output_tokens={out_tok}: {e}")
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
        
        if evaluator:
            evaluator.cleanup()
            del evaluator
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("[INFO] Model and evaluator cleaned up successfully.")

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
                print("[INFO] Process group destroyed successfully.")
        except Exception as e:
            print(f"[DEBUG] destroy_process_group failed or not needed: {e}")
