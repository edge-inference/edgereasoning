#!/usr/bin/env python3
"""
Base Evaluation All Subjects Test Script

Runs full reasoning MMLU evaluation on ALL subjects in the MMLU dataset.
Tests the base evaluation mode across the full dataset.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path


script_dir = Path(__file__).resolve().parent
mmlu_root = script_dir.parent
project_root = script_dir.parents[3]
sys.path.insert(0, str(mmlu_root))
sys.path.insert(0, str(project_root))

from src.evaluators.base_evaluator import BaseEvaluator
from src.data_loaders.mmlu_loader import MMLULoader
from loaders.results import get_results_config


def main():
    """Run base evaluation on all subjects."""
    parser = argparse.ArgumentParser(description='Base MMLU Evaluation')
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B', help='Model path')
    parser.add_argument('--config', default='configs/base.yaml', help='Config file path')
    parser.add_argument('--max-tokens', type=int, help='Override max tokens (optional)')
    parser.add_argument('--num-questions', type=int, default=100, help='Number of questions to evaluate per subject (optional)')
    parser.add_argument('--cpu', action='store_true', help='Run model on CPU instead of GPU (default: GPU)')
    parser.add_argument('--no-flash-attention', action='store_true', help='Disable Flash Attention (use xformers instead)')
    args = parser.parse_args()

    model_path = args.model
    config_path = args.config
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_config = get_results_config()
    results_base_dir = results_config.get_result_base_dir('mmlu', 'tegra')
    
    model_name = model_path.split('/')[-1] if '/' in model_path else model_path
    output_base = results_base_dir / f"base_all_subjects_{timestamp}_{model_name}"
    os.makedirs(output_base, exist_ok=True)

    print("> Starting Base MMLU Evaluation - ALL SUBJECTS")
    print("=================================================")
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    if args.max_tokens:
        print(f"Max tokens override: {args.max_tokens}")
    if args.num_questions:
        print(f"Questions per subject override: {args.num_questions}")
    if args.cpu:
        print(f"Device: CPU (forced)")
    else:
        print(f"Device: GPU (default)")
    print(f"Output base: {output_base}")
    print(f"Timestamp: {timestamp}\n")

    try:
        evaluator = BaseEvaluator(config_path)
        if args.max_tokens:
            evaluator.config.model['max_tokens'] = args.max_tokens
        if args.num_questions:
            evaluator.config.evaluation['num_questions'] = args.num_questions

        # Setup model once
        print("* Setting up model...")
        evaluator.setup_model(model_path)

        loader = MMLULoader()
        all_subjects = loader.get_available_subjects()
        questions_per_subject = args.num_questions if args.num_questions else "all available"
        print(f"* Found {len(all_subjects)} subjects to evaluate")
        print(f"* Questions per subject: {questions_per_subject}")

        successful = 0
        total_correct = 0
        total_questions = 0

        for i, subject in enumerate(all_subjects, 1):
            print(f"\n{'='*60}")
            print(f"> [{i}/{len(all_subjects)}] Evaluating subject: {subject}")
            print(f"{'='*60}")
            try:
                subject_dir = os.path.join(output_base, subject)
                result = evaluator.evaluate_subject(
                    model_path=model_path,
                    subject=subject,
                    output_dir=subject_dir
                )
                print(f"✓ {subject} completed! Questions: {result.total_questions}, Accuracy: {result.accuracy:.2%}")
                successful += 1
                total_correct += result.correct_answers
                total_questions += result.total_questions
            except Exception as e:
                print(f"x {subject} failed: {e}")

        overall_accuracy = total_correct / total_questions if total_questions else 0.0
        print(f"\n{'='*60}")
        print("> BASE EVALUATION - ALL SUBJECTS SUMMARY")
        print(f"Model: {model_path}")
        print(f"Successful: {successful}/{len(all_subjects)}")
        print(f"Overall Accuracy: {overall_accuracy:.2%}")

        summary = {
            'model': model_path,
            'config': config_path,
            'timestamp': timestamp,
            'total_subjects': len(all_subjects),
            'successful_subjects': successful,
            'overall_accuracy': overall_accuracy,
            'total_questions': total_questions,
            'total_correct': total_correct,
            'output_base': output_base,
            'config_details': {
                'name': evaluator.config.name,
                'description': evaluator.config.description,
                'model_settings': evaluator.config.model,
                'evaluation_settings': evaluator.config.evaluation,
                'prompting_strategy': evaluator.config.prompting,
                'output_settings': evaluator.config.output,
            }
        }
        summary_file = os.path.join(output_base, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"* Summary saved to: {summary_file}")

        return successful == len(all_subjects)
    except Exception as e:
        print(f"x Full evaluation failed: {e}")
        import traceback; traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)