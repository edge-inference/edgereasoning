#!/usr/bin/env python3
"""
Simple working evaluation script
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluators.base_evaluator import BaseEvaluator

def main():
    print("🧪 Simple MMLU Evaluation Test")
    print("=" * 40)
    
    # Use base evaluator with config
    evaluator = BaseEvaluator("configs/base.yaml")
    
    # Set up model
    model_path = "l3lab/L1-Qwen-1.5B-Max"
    print(f"Setting up model: {model_path}")
    evaluator.setup_model(model_path)
    print("Model setup complete!")
    
    # Run evaluation on a small subject
    subject = "anatomy"  # Use an available subject
    output_dir = "../results/quick_test"
    
    print(f"\nEvaluating subject: {subject}")
    print(f"Output directory: {output_dir}")
    
    # Override config to limit questions for quick test
    evaluator.config.evaluation['num_questions'] = 10
    
    result = evaluator.evaluate_subject(
        model_path=model_path,
        subject=subject,
        output_dir=output_dir
    )
    
    print(f"\n✅ Evaluation completed!")
    print(f"Accuracy: {result.accuracy:.2%}")
    print(f"Correct: {result.correct_answers}/{result.total_questions}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
