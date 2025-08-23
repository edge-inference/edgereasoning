#!/usr/bin/env python3
"""
Scale MMLU Evaluation - Jetson Orin Optimized with Synthetic Datasets

Runs scale evaluation using synthetic datasets
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from src.evaluators.scale_evaluator import ScaleEvaluator
from src.data_loaders.custom_loader import CustomLoader
from loaders.benchmarks import get_benchmark_config


def get_model_config(model_name):
    """Get the appropriate CSV file based on model family using benchmark config."""
    config = get_benchmark_config()
    
    # Detect model family
    family = config.detect_model_family(model_name)
    if not family:
        family = 'llama'  # Default fallback
    
    # Get dataset path
    csv_file = str(config.get_model_synthetic_dataset(model_name))
    
    return {
        'csv_file': csv_file,
        'family': family.capitalize()
    }


def main():
    """Run scale evaluation using synthetic datasets."""
    # Load benchmark configuration
    bench_config = get_benchmark_config()
    synthetic_settings = bench_config.get_synthetic_settings()
    eval_defaults = bench_config.get_evaluation_defaults()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TT-Scaling - Synthetic Dataset')
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', help='Model path')
    parser.add_argument('--config', help='Config file path (auto-detected if not provided)')
    parser.add_argument('--num-samples', type=int, default=synthetic_settings.get('default_samples'), help='Override number of samples per question')
    parser.add_argument('--token-budget', type=int, help='Override token budget')
    parser.add_argument('--input-tokens', type=int, default=synthetic_settings.get('default_input_tokens'), help='Input token length to test')
    parser.add_argument('--seed', type=int, default=eval_defaults.get('default_seed'), help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Configuration
    model_path = args.model
    # Auto-detect config path if not provided
    config_path = args.config or bench_config.get_default_config_path(model_path)
    timestamp = datetime.now().strftime(bench_config.get_output_settings()['timestamp_format'])
    
    # Get model-specific configuration
    model_config = get_model_config(model_path)
    print(f"Detected family: {model_config['family']}")
    print(f"Using CSV: {model_config['csv_file']}")
    
    # Check if CSV file exists
    if not os.path.exists(model_config['csv_file']):
        print(f"ERROR: CSV file not found: {model_config['csv_file']}")
        print(f"Please ensure the synthetic dataset for {model_config['family']} family exists")
        print(f"Available token lengths: {bench_config.get_available_synthetic_tokens()}")
        return False
    
    # Output directory setup using benchmark config
    model_name = model_path.split('/')[-1] if '/' in model_path else model_path
    output_suffix = f"_{model_name}"
    
    if args.num_samples:
        output_suffix += f"_{args.num_samples}samples"
    if args.token_budget:
        output_suffix += f"_{args.token_budget}tokens"
    if args.seed != eval_defaults.get('default_seed'):
        output_suffix += f"_seed{args.seed}"
        
    output_base = str(bench_config.get_output_base_dir("synthetic", timestamp, output_suffix))
    os.makedirs(output_base, exist_ok=True)
    
    print("🚀 Starting Scale MMLU Evaluation - Synthetic Dataset")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    print(f"Model Family: {model_config['family']}")
    print(f"Input tokens: {args.input_tokens}")
    print(f"Samples per question: {args.num_samples}")
    if args.token_budget:
        print(f"Token budget override: {args.token_budget}")
    print(f"Random seed: {args.seed}")
    print(f"Output base: {output_base}")
    print("")
    
    try:
        evaluator = ScaleEvaluator(config_path)
        
        # Apply overrides (like in controlled_sweep.py )
        evaluator.config.model['path'] = model_path
        if args.num_samples:
            evaluator.config.scaling['num_samples'] = args.num_samples
        if args.token_budget:
            evaluator.config.scaling['min_new_tokens'] = args.token_budget
            evaluator.config.scaling['max_new_tokens'] = args.token_budget
            evaluator.config.model['max_tokens'] = args.token_budget
        if args.seed:
            evaluator.config.scaling['seed'] = args.seed
        
        # Load synthetic dataset
        loader = CustomLoader(model_config['csv_file'])
        print(f"DEBUG: Available input token counts in loader: {sorted(set(q['input_tokens'] for q in loader.questions))}")
        
        # Get the specific question with desired input tokens
        question = loader.get_question_by_input_tokens(args.input_tokens)
        if not question:
            print(f"ERROR: No question found for input_tokens={args.input_tokens}")
            return False
            
        print(f"✅ Found question: ID {question.question_id}, Subject: {getattr(question, 'subject', 'unknown')}")
        print(f"Question preview: {question.question[:100]}...")
        
        # Setup model ONCE
        print("🔧 Setting up model...")
        evaluator.setup_model(model_path)
        
        # Run scale evaluation on single question
        print(f"\n{'='*60}")
        print(f"🏃 Running Scale Evaluation")
        print(f"{'='*60}")
        
        # Create run name
        run_name = f"scale_synthetic_{getattr(question, 'subject', 'unknown')}_q{question.question_id}_in{args.input_tokens}_{timestamp}"
        
        # Run evaluation (uses already loaded model)
        result = evaluator.evaluate_subject_on_questions(
            model_path=model_path,
            questions=[question],  # Single question list
            output_dir=output_base,
            run_name=run_name
        )
        
        # Print results
        print(f"✅ Scale evaluation completed!")
        print(f"   Accuracy: {result.accuracy:.2%}")
        print(f"   Correct: {result.correct_answers}/{result.total_questions}")
        print(f"   Avg Time/Question: {result.avg_time_per_question:.1f}ms")
        
        # Save summary
        summary = {
            'model': model_path,
            'config': config_path,
            'timestamp': timestamp,
            'input_tokens': args.input_tokens,
            'num_samples': args.num_samples or evaluator.config.scaling.get('num_samples'),
            'token_budget': args.token_budget,
            'seed': args.seed or evaluator.config.scaling.get('seed'),
            'question_id': question.question_id,
            'subject': getattr(question, 'subject', 'unknown'),
            'accuracy': result.accuracy,
            'total_questions': result.total_questions,
            'correct_answers': result.correct_answers,
            'avg_time_per_question': result.avg_time_per_question,
            'output_base': output_base
        }
        
        summary_file = os.path.join(output_base, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📋 Summary saved to: {summary_file}")
        print(f"📁 All results in: {output_base}")
        print(f"\n🎉 Scale evaluation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Scale evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 