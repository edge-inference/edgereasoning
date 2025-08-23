#!/usr/bin/env python3
"""
Simple Scale Evaluation Test

Test the scale evaluator on a single subject to verify functionality.
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluators.scale_evaluator import ScaleEvaluator


def main():
    """Run scale evaluation test on a single subject."""
    parser = argparse.ArgumentParser(description='Test Scale MMLU Evaluation')
    parser.add_argument('--model', default='l3lab/L1-Qwen-1.5B-Max', help='Model path')
    parser.add_argument('--subject', default='computer_science', help='Subject to test')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples per question')
    args = parser.parse_args()
    
    model_path = args.model
    subject = args.subject
    num_samples = args.num_samples
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_dir = f"./results/scale_test_{subject}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🧪 Scale Evaluation Test")
    print("=======================")
    print(f"Model: {model_path}")
    print(f"Subject: {subject}")
    print(f"Samples per question: {num_samples}")
    print(f"Output: {output_dir}")
    print("")
    
    try:
        evaluator = ScaleEvaluator("configs/scale.yaml")
        
        evaluator.config.scaling['num_samples'] = num_samples
        evaluator.config.model['path'] = model_path
        
        print("🔧 Setting up model...")
        evaluator.setup_model(model_path)
        
        print(f"🏃 Running evaluation on {subject}...")
        result = evaluator.evaluate_subject(
            model_path=model_path,
            subject=subject,
            output_dir=output_dir
        )
        
        print("\n📊 RESULTS:")
        print(f"Accuracy: {result.accuracy:.2%}")
        print(f"Correct: {result.correct_answers}/{result.total_questions}")
        print(f"Avg Time/Question: {result.avg_time_per_question:.1f}ms")
        
        scaling_metrics = None
        for item in result.question_results:
            if 'scaling_metrics' in item:
                scaling_metrics = item['scaling_metrics']
                break
        
        if scaling_metrics:
            print(f"\nSCALING METRICS:")
            print(f"Total Samples: {scaling_metrics['total_samples_generated']}")
            print(f"Samples per Question: {scaling_metrics['samples_per_question']}")
            print(f"Avg Voting Confidence: {scaling_metrics['avg_voting_confidence']:.3f}")
            print(f"Scaling Efficiency: {scaling_metrics['scaling_efficiency']:.3f}")
        
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
