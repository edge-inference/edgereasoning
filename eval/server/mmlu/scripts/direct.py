#!/usr/bin/env python3
"""
Direct Evaluation Script - Direct answer selection for non-reasoning models
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(str(Path(__file__).parents[3]))

from src.evaluators.noreasoning_evaluator import NoReasoningEvaluator
from src.data_loaders.mmlu_loader import MMLULoader
from loaders.benchmarks import get_benchmark_config
from loaders.results import get_results_config
from loaders.models import get_default_direct_model, get_all_direct_model_paths


def main():
    parser = argparse.ArgumentParser(description='Direct MMLU Evaluation')
    parser.add_argument('--model', help='Specific model path to evaluate')
    parser.add_argument('--sweep', action='store_true', help='Run all direct models from config')
    parser.add_argument('--config', default='configs/direct.yaml', help='Config file path')
    parser.add_argument('--max-tokens', type=int, help='Override max tokens per response')
    parser.add_argument('--temperature', type=float, help='Override temperature for sampling')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Number of GPUs to use (default: 1)')
    args = parser.parse_args()
    
    config = get_benchmark_config()
    results_config = get_results_config()
    
    if args.sweep:
        models_to_evaluate = get_all_direct_model_paths()
        mode_description = "MODEL SWEEP"
    elif args.model:
        models_to_evaluate = [args.model]
        mode_description = "SINGLE MODEL"
    else:
        models_to_evaluate = [get_default_direct_model()]
        mode_description = "DEFAULT MODEL"
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"Starting Direct MMLU Evaluation - {mode_description}")
    print("=" * (35 + len(mode_description)))
    print(f"Models to evaluate: {len(models_to_evaluate)}")
    for i, model in enumerate(models_to_evaluate, 1):
        print(f"  {i}. {model}")
    print(f"Config: {args.config}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size} GPUs")
    if args.max_tokens:
        print(f"Max tokens override: {args.max_tokens}")
    if args.temperature is not None:
        print(f"Temperature override: {args.temperature}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print(f"Timestamp: {timestamp}")
    print("")
    
    all_model_results = {}
    
    for model_idx, model_path in enumerate(models_to_evaluate, 1):
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL {model_idx}/{len(models_to_evaluate)}: {model_path}")
        print(f"{'='*80}")
        
        model_name = model_path.split('/')[-1] if '/' in model_path else model_path
        
        base_results_dir = results_config.get_result_base_dir('mmlu', model_name=model_path)
        output_base = base_results_dir / 'direct'
        os.makedirs(output_base, exist_ok=True)
        
        try:
            evaluator = NoReasoningEvaluator(args.config)
            
            evaluator.config.model['path'] = model_path
            evaluator.config.model['tensor_parallel_size'] = args.tensor_parallel_size
            if args.max_tokens:
                evaluator.config.model['max_tokens'] = args.max_tokens
            if args.temperature is not None:
                evaluator.config.model['temperature'] = args.temperature
            if args.seed is not None:
                evaluator.config.model['seed'] = args.seed
            
            print(f"* Setting up model with {args.tensor_parallel_size} GPUs...")
            evaluator.setup_model(model_path)
            
            loader = MMLULoader()
            all_subjects = loader.get_available_subjects()
            
            print(f"* Found {len(all_subjects)} subjects to evaluate")
            print(f"Subjects: {', '.join(all_subjects[:5])}..." if len(all_subjects) > 5 else f"Subjects: {', '.join(all_subjects)}")
            print("")
            
            successful_subjects = 0
            total_correct = 0
            total_questions = 0
            total_quick_responses = 0
            
            # Run evaluation for each subject
            for i, subject in enumerate(all_subjects, 1):
                print(f"\n{'='*60}")
                print(f"[{i}/{len(all_subjects)}] Evaluating subject: {subject}")
                print(f"{'='*60}")
                
                try:
                    subject_output_dir = os.path.join(output_base, subject)
                    
                    result = evaluator.evaluate_subject(
                        model_path=model_path,
                        subject=subject,
                        output_dir=subject_output_dir
                    )
                    
                    print(f"* {subject} completed!")
                    print(f"   Accuracy: {result.accuracy:.2%}")
                    print(f"   Correct: {result.correct_answers}/{result.total_questions}")
                    print(f"   Avg Time/Question: {result.avg_time_per_question:.1f}ms")
                    print(f"   Avg Tokens/Second: {result.avg_tokens_per_second:.1f}")
                    
                    if hasattr(evaluator, 'quick_responses'):
                        print(f"   Quick Responses: {evaluator.quick_responses}")
                        total_quick_responses += evaluator.quick_responses
                        evaluator.quick_responses = 0
                    
                    successful_subjects += 1
                    total_correct += result.correct_answers
                    total_questions += result.total_questions
                    
                except Exception as e:
                    print(f"Error: {subject} failed: {e}")
            
            overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
            quick_response_rate = total_quick_responses / total_questions if total_questions > 0 else 0.0
            
            model_results = {
                'model': model_path,
                'successful_subjects': successful_subjects,
                'total_subjects': len(all_subjects),
                'overall_accuracy': overall_accuracy,
                'total_questions': total_questions,
                'total_correct': total_correct,
                'quick_response_rate': quick_response_rate,
                'total_quick_responses': total_quick_responses,
                'output_base': str(output_base)
            }
            all_model_results[model_name] = model_results
            
            # Print model summary
            print(f"\n{'='*60}")
            print(f" MODEL SUMMARY: {model_name}")
            print(f"{'='*60}")
            print(f"Successful: {successful_subjects}/{len(all_subjects)}")
            print(f"Overall Accuracy: {overall_accuracy:.2%}")
            print(f"Total Questions: {total_questions}")
            print(f"Total Correct: {total_correct}")
            print(f"Quick Response Rate: {quick_response_rate:.2%}")
            print(f"Total Quick Responses: {total_quick_responses}")
            print(f"Output: {output_base}")
            
            model_summary = {
                'model': model_path,
                'config': args.config,
                'timestamp': timestamp,
                'tensor_parallel_size': args.tensor_parallel_size,
                'total_subjects': len(all_subjects),
                'successful_subjects': successful_subjects,
                'overall_accuracy': overall_accuracy,
                'total_questions': total_questions,
                'total_correct': total_correct,
                'quick_response_rate': quick_response_rate,
                'total_quick_responses': total_quick_responses,
                'output_base': str(output_base),
                'config_details': {
                    'name': evaluator.config.name,
                    'description': evaluator.config.description,
                    'model_settings': {
                        'max_tokens': evaluator.config.model.get('max_tokens'),
                        'max_model_len': evaluator.config.model.get('max_model_len'),
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
                json.dump(model_summary, f, indent=2)
            
            print(f"* Model summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"Error: Model {model_path} failed completely: {e}")
            import traceback
            traceback.print_exc()
            all_model_results[model_name] = {
                'model': model_path,
                'error': str(e),
                'status': 'failed'
            }
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    for model_name, results in all_model_results.items():
        if 'error' in results:
            print(f"✗ {model_name}: FAILED - {results['error']}")
        else:
            print(f"✓ {model_name}: {results['overall_accuracy']:.2%} accuracy ({results['successful_subjects']}/{results['total_subjects']} subjects)")
    
    # Save sweep summary if multiple models
    if len(models_to_evaluate) > 1:
        sweep_summary = {
            'sweep_timestamp': timestamp,
            'config': args.config,
            'tensor_parallel_size': args.tensor_parallel_size,
            'models_evaluated': len(models_to_evaluate),
            'results': all_model_results
        }
        
        server_results_dir = results_config.get_result_base_dir('mmlu')
        sweep_summary_file = server_results_dir / f"{timestamp}_direct_sweep_summary.json"
        with open(sweep_summary_file, 'w') as f:
            json.dump(sweep_summary, f, indent=2)
        
        print(f"\n* Sweep summary saved to: {sweep_summary_file}")
    
    # Final status  
    successful_models = sum(1 for r in all_model_results.values() if 'error' not in r)
    if successful_models == len(models_to_evaluate):
        print(f"\n✓ All {len(models_to_evaluate)} models completed successfully!")
        return True
    else:
        print(f"\n! {successful_models}/{len(models_to_evaluate)} models completed successfully")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)