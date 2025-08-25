#!/usr/bin/env python3
"""
Budget Evaluation Script - Configurable token limits
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

os.environ["VLLM_ENABLE_METRICS"] = "1"
os.environ["VLLM_PROFILE"] = "1"
os.environ["VLLM_DETAILED_METRICS"] = "1"
os.environ["VLLM_REQUEST_METRICS"] = "1"

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(str(Path(__file__).parents[3]))

from src.evaluators.budget_evaluator import BudgetEvaluator
from src.data_loaders.mmlu_loader import MMLULoader
from loaders.benchmarks import get_benchmark_config
from loaders.results import get_results_config


def main():
    parser = argparse.ArgumentParser(description='Budget MMLU Evaluation')
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', help='Model path')
    parser.add_argument('--config', default='configs/budget.yaml', help='Config file path')
    parser.add_argument('--max-tokens', type=int, help='Override max tokens (optional)')
    args = parser.parse_args()
    
    config = get_benchmark_config()
    results_config = get_results_config()
    
    model_name = args.model.split('/')[-1] if '/' in args.model else args.model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directory in the results-defined location with model-specific path
    base_results_dir = results_config.get_result_base_dir('mmlu', model_name=args.model)
    suffix = f'budget_{model_name}'
    if args.max_tokens:
        suffix += f'_{args.max_tokens}tok'
    
    output_base = base_results_dir / f"{timestamp}_{suffix}"
    os.makedirs(output_base, exist_ok=True)
    
    
    print("Starting Budget MMLU Evaluation - ALL SUBJECTS")
    print("================================================")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    if args.max_tokens:
        print(f"Max tokens override: {args.max_tokens}")
    print(f"Output base: {output_base}")
    print(f"Timestamp: {timestamp}")
    print("")
    
    try:
        evaluator = BudgetEvaluator(args.config)
        
        evaluator.config.model['path'] = args.model
        if args.max_tokens:
            evaluator.config.model['max_tokens'] = args.max_tokens
        
        print("* Setting up model...")
        evaluator.setup_model(args.model)
        
        loader = MMLULoader()
        all_subjects = loader.get_available_subjects()
        
        print(f"* Found {len(all_subjects)} subjects to evaluate")
        print(f"Subjects: {', '.join(all_subjects[:5])}..." if len(all_subjects) > 5 else f"Subjects: {', '.join(all_subjects)}")
        print("")
        
        successful_subjects = 0
        total_correct = 0
        total_questions = 0
        
        for i, subject in enumerate(all_subjects, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(all_subjects)}] Evaluating subject: {subject}")
            print(f"{'='*60}")
            
            try:
                subject_output_dir = os.path.join(output_base, subject)
                
                result = evaluator.evaluate_subject(
                    model_path=args.model,
                    subject=subject,
                    output_dir=subject_output_dir
                )
                
                print(f"* {subject} completed!")
                print(f"   Accuracy: {result.accuracy:.2%}")
                print(f"   Correct: {result.correct_answers}/{result.total_questions}")
                print(f"   Avg Time/Question: {result.avg_time_per_question:.1f}ms")
                
                successful_subjects += 1
                total_correct += result.correct_answers
                total_questions += result.total_questions
                
            except Exception as e:
                print(f"ERROR: {subject} failed: {e}")
        
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        print(f"\n{'='*60}")
        print("BUDGET EVALUATION - ALL SUBJECTS SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {args.model}")
        print(f"Total Subjects: {len(all_subjects)}")
        print(f"Successful: {successful_subjects}/{len(all_subjects)}")
        print(f"Overall Accuracy: {overall_accuracy:.2%}")
        print(f"Total Questions: {total_questions}")
        print(f"Total Correct: {total_correct}")
        
        summary = {
            'model': args.model,
            'config': args.config,
            'timestamp': timestamp,
            'total_subjects': len(all_subjects),
            'successful_subjects': successful_subjects,
            'overall_accuracy': overall_accuracy,
            'total_questions': total_questions,
            'total_correct': total_correct,
            'output_base': output_base,
            'config_details': {
                'name': evaluator.config.name,
                'description': evaluator.config.description,
                'model_settings': {
                    'max_tokens': evaluator.config.model.get('max_tokens'),
                    'temperature': evaluator.config.model.get('temperature'),
                    'top_p': evaluator.config.model.get('top_p'),
                    'tensor_parallel_size': evaluator.config.model.get('tensor_parallel_size'),
                    'gpu_memory_utilization': evaluator.config.model.get('gpu_memory_utilization')
                },
                'evaluation_settings': evaluator.config.evaluation,
                'prompting_strategy': {
                    'template_type': evaluator.config.prompting.get('template_type'),
                    'system_prompt': evaluator.config.prompting.get('system_prompt'),
                    'user_template': evaluator.config.prompting.get('user_template')
                },
                'output_settings': evaluator.config.output
            }
        }
        
        summary_file = os.path.join(output_base, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n* Summary saved to: {summary_file}")
        print(f"* All results in: {output_base}")
        
        if successful_subjects == len(all_subjects):
            print(f"\n* All {len(all_subjects)} subjects completed successfully!")
            return True
        else:
            print(f"\n* {successful_subjects}/{len(all_subjects)} subjects completed successfully")
            return False
        
    except Exception as e:
        print(f"ERROR: Full evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
