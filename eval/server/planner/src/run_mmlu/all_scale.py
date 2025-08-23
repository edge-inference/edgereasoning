#!/usr/bin/env python3
"""
Scale Evaluation All Subjects Test Script

Runs scale MMLU evaluation (test-time scaling with multiple samples and majority voting) 
on ALL subjects in the MMLU dataset.
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.evaluators.scale_evaluator import ScaleEvaluator
from src.data_loaders.mmlu_loader import MMLULoader


def main():
    """Run scale evaluation on all subjects."""
    parser = argparse.ArgumentParser(description='Scale MMLU Evaluation')
    parser.add_argument('--model', default='l3lab/L1-Qwen-1.5B-Max', help='Model path')
    parser.add_argument('--config', default='configs/scale.yaml', help='Config file path')
    parser.add_argument('--num-samples', type=int, help='Override number of samples per question')
    parser.add_argument('--token-budget', type=int, help='Override token budget (max tokens per sample)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    model_path = args.model
    config_path = args.config
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_name = model_path.split('/')[-1] if '/' in model_path else model_path
    output_suffix = f"_{model_name}"
    
    if args.num_samples:
        output_suffix += f"_{args.num_samples}samples"
    if args.token_budget:
        output_suffix += f"_{args.token_budget}tokens"
    if args.seed is not None:
        output_suffix += f"_seed{args.seed}"
    output_base = f"./results/scale_all_subjects_{timestamp}{output_suffix}"
    os.makedirs(output_base, exist_ok=True)
    
    print("🚀 Starting Scale MMLU Evaluation - ALL SUBJECTS")
    print("================================================")
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    if args.num_samples:
        print(f"Samples per question override: {args.num_samples}")
    if args.token_budget:
        print(f"Token budget override: {args.token_budget}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print(f"Output base: {output_base}")
    print(f"Timestamp: {timestamp}")
    print("")
    
    try:
        evaluator = ScaleEvaluator(config_path)
        
        evaluator.config.model['path'] = model_path
        if args.num_samples:
            evaluator.config.scaling['num_samples'] = args.num_samples
        if args.token_budget:
            evaluator.config.scaling['max_new_tokens'] = args.token_budget
            evaluator.config.model['max_tokens'] = args.token_budget
        if args.seed:
            evaluator.config.model['seed'] = args.seed
        
        print("🔧 Setting up model...")
        evaluator.setup_model(model_path)
        
        loader = MMLULoader()
        all_subjects = loader.get_available_subjects()
        
        print(f"📚 Found {len(all_subjects)} subjects to evaluate")
        print(f"Subjects: {', '.join(all_subjects[:5])}..." if len(all_subjects) > 5 else f"Subjects: {', '.join(all_subjects)}")
        
        samples_per_question = args.num_samples or evaluator.config.scaling['num_samples']
        total_samples_expected = len(all_subjects) * samples_per_question
        print(f"🎯 Test-time scaling: {samples_per_question} samples per question")
        print(f"🔢 Expected total samples: ~{total_samples_expected} (approximate)")
        print("")
        
        successful_subjects = 0
        total_correct = 0
        total_questions = 0
        total_samples_generated = 0
        voting_confidences = []
        
        for i, subject in enumerate(all_subjects, 1):
            print(f"\n{'='*60}")
            print(f"🏃 [{i}/{len(all_subjects)}] Evaluating subject: {subject}")
            print(f"{'='*60}")
            
            try:
                subject_output_dir = os.path.join(output_base, subject)
                
                result = evaluator.evaluate_subject(
                    model_path=model_path,
                    subject=subject,
                    output_dir=subject_output_dir
                )
                
                print(f"✅ {subject} completed!")
                print(f"   Accuracy: {result.accuracy:.2%}")
                print(f"   Correct: {result.correct_answers}/{result.total_questions}")
                print(f"   Avg Time/Question: {result.avg_time_per_question:.1f}ms")
                
                scaling_metrics = None
                for item in result.question_results:
                    if 'scaling_metrics' in item:
                        scaling_metrics = item['scaling_metrics']
                        break
                
                if scaling_metrics:
                    print(f"   Samples Generated: {scaling_metrics['total_samples_generated']}")
                    print(f"   Avg Voting Confidence: {scaling_metrics['avg_voting_confidence']:.3f}")
                    print(f"   Scaling Efficiency: {scaling_metrics['scaling_efficiency']:.3f}")
                    total_samples_generated += scaling_metrics['total_samples_generated']
                    voting_confidences.append(scaling_metrics['avg_voting_confidence'])
                
                successful_subjects += 1
                total_correct += result.correct_answers
                total_questions += result.total_questions
                
            except Exception as e:
                print(f"❌ {subject} failed: {e}")
        
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        avg_voting_confidence = sum(voting_confidences) / len(voting_confidences) if voting_confidences else 0.0
        
        print(f"\n{'='*60}")
        print("📊 SCALE EVALUATION - ALL SUBJECTS SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {model_path}")
        print(f"Total Subjects: {len(all_subjects)}")
        print(f"Successful: {successful_subjects}/{len(all_subjects)}")
        print(f"Overall Accuracy: {overall_accuracy:.2%}")
        print(f"Total Questions: {total_questions}")
        print(f"Total Correct: {total_correct}")
        print(f"")
        print(f"SCALING METRICS:")
        print(f"Samples per Question: {samples_per_question}")
        print(f"Total Samples Generated: {total_samples_generated}")
        print(f"Avg Voting Confidence: {avg_voting_confidence:.3f}")
        print(f"Scaling Factor: {total_samples_generated / total_questions:.1f}x" if total_questions > 0 else "N/A")
        
        summary = {
            'model': model_path,
            'config': config_path,
            'timestamp': timestamp,
            'seed': args.seed,
            'total_subjects': len(all_subjects),
            'successful_subjects': successful_subjects,
            'overall_accuracy': overall_accuracy,
            'total_questions': total_questions,
            'total_correct': total_correct,
            'output_base': output_base,
            'scaling_metrics': {
                'samples_per_question': samples_per_question,
                'total_samples_generated': total_samples_generated,
                'avg_voting_confidence': avg_voting_confidence,
                'scaling_factor': total_samples_generated / total_questions if total_questions > 0 else 0.0
            },
            'config_details': {
                'name': evaluator.config.name,
                'description': evaluator.config.description,
                'model_settings': {
                    'max_tokens': evaluator.config.model.get('max_tokens'),
                    'temperature': evaluator.config.model.get('temperature'),
                    'top_p': evaluator.config.model.get('top_p'),
                    'top_k': evaluator.config.model.get('top_k'),
                    'tensor_parallel_size': evaluator.config.model.get('tensor_parallel_size'),
                    'gpu_memory_utilization': evaluator.config.model.get('gpu_memory_utilization'),
                    'repetition_penalty': evaluator.config.model.get('repetition_penalty')
                },
                'evaluation_settings': evaluator.config.evaluation,
                'scaling_settings': evaluator.config.scaling,
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
