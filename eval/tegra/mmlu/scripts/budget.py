#!/usr/bin/env python3
"""
Budget Evaluation All Subjects Test Script

Runs budget MMLU evaluation on ALL subjects in the MMLU dataset.
Tests the budget-optimized evaluation mode across the full dataset.
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluators.budget_evaluator import BudgetEvaluator
from src.data_loaders.mmlu_loader import MMLULoader


def main():
    """Run budget evaluation on all subjects."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Budget MMLU Evaluation')
    parser.add_argument('--model', default='l3lab/L1-Qwen-1.5B-Max', help='Model path')
    parser.add_argument('--config', default='configs/budget.yaml', help='Config file path')
    parser.add_argument('--max-tokens', type=int, help='Override max tokens (optional)')
    args = parser.parse_args()
    
    # Configuration
    model_path = args.model
    config_path = args.config
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Output directory setup
    model_name = model_path.split('/')[-1] if '/' in model_path else model_path
    output_suffix = f"_{model_name}"
    
    if args.max_tokens:
        output_suffix += f"_{args.max_tokens}tok"
    output_base = f"./results/budget_all_subjects_{timestamp}{output_suffix}"
    os.makedirs(output_base, exist_ok=True)
    
    
    print("💰 Starting Budget MMLU Evaluation - ALL SUBJECTS")
    print("=================================================")
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    if args.max_tokens:
        print(f"Max tokens override: {args.max_tokens}")
    print(f"Output base: {output_base}")
    print(f"Timestamp: {timestamp}")
    print("")
    
    try:
        evaluator = BudgetEvaluator(config_path)
        
        # Override model path
        evaluator.config.model['path'] = model_path
        if args.max_tokens:
            evaluator.config.model['max_tokens'] = args.max_tokens
        
        # Setup model ONCE
        print("🔧 Setting up model...")
        evaluator.setup_model(model_path)
        
        # Get all available subjects
        loader = MMLULoader()
        all_subjects = loader.get_available_subjects()
        
        print(f"📚 Found {len(all_subjects)} subjects to evaluate")
        print(f"Subjects: {', '.join(all_subjects[:5])}..." if len(all_subjects) > 5 else f"Subjects: {', '.join(all_subjects)}")
        print("")
        
        
        # Simple counters only
        successful_subjects = 0
        total_correct = 0
        total_questions = 0
        
        # Run evaluation for each subject (MODEL LOADED)
        for i, subject in enumerate(all_subjects, 1):
            print(f"\n{'='*60}")
            print(f"🏃 [{i}/{len(all_subjects)}] Evaluating subject: {subject}")
            print(f"{'='*60}")
            
            try:
                # Create subject-specific output directory
                subject_output_dir = os.path.join(output_base, subject)
                
                # Run evaluation (uses already loaded model)
                result = evaluator.evaluate_subject(
                    model_path=model_path,
                    subject=subject,
                    output_dir=subject_output_dir
                )
                
                # Print subject results
                print(f"✅ {subject} completed!")
                print(f"   Accuracy: {result.accuracy:.2%}")
                print(f"   Correct: {result.correct_answers}/{result.total_questions}")
                print(f"   Avg Time/Question: {result.avg_time_per_question:.1f}ms")
                
                # overall tracking 
                successful_subjects += 1
                total_correct += result.correct_answers
                total_questions += result.total_questions
                
            except Exception as e:
                print(f"❌ {subject} failed: {e}")
        
        # overall metrics
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        # summary
        print(f"\n{'='*60}")
        print("📊 BUDGET EVALUATION - ALL SUBJECTS SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {model_path}")
        print(f"Total Subjects: {len(all_subjects)}")
        print(f"Successful: {successful_subjects}/{len(all_subjects)}")
        print(f"Overall Accuracy: {overall_accuracy:.2%}")
        print(f"Total Questions: {total_questions}")
        print(f"Total Correct: {total_correct}")
        
        # Save summary
        summary = {
            'model': model_path,
            'config': config_path,
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
        
        print(f"\n📋 Summary saved to: {summary_file}")
        print(f"📁 All results in: {output_base}")
        
        # Final status
        if successful_subjects == len(all_subjects):
            print(f"\n🎉 All {len(all_subjects)} subjects completed successfully!")
            return True
        else:
            print(f"\n⚠️  {successful_subjects}/{len(all_subjects)} subjects completed successfully")
            return False
        
    except Exception as e:
        print(f"❌ Full evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
