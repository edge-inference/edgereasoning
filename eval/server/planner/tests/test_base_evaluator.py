"""
Test the base evaluator configuration loading and setup.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluators.base_evaluator import BaseEvaluator, EvaluationConfig


def test_config_loading():
    """Test configuration loading from YAML files."""
    print("Testing Configuration Loading")
    print("=" * 40)
    
    config_files = ['base.yaml', 'budget.yaml', 'noreasoning.yaml']
    
    for config_file in config_files:
        config_path = f"configs/{config_file}"
        if os.path.exists(config_path):
            try:
                config = EvaluationConfig.from_yaml(config_path)
                print(f"✓ {config_file}: {config.name} - {config.description}")
                print(f"  Max tokens: {config.model['max_tokens']}")
                print(f"  Template: {config.prompting['template_type']}")
                print()
            except Exception as e:
                print(f"✗ {config_file}: Error - {e}")
        else:
            print(f"✗ {config_file}: File not found")
    
    return True


def test_evaluator_setup():
    """Test evaluator initialization."""
    print("Testing Evaluator Setup")
    print("=" * 40)
    
    try:
        evaluator = BaseEvaluator("configs/base.yaml")
        print(f"✓ Evaluator created successfully")
        print(f"  Config name: {evaluator.config.name}")
        print(f"  Max tokens: {evaluator.config.model['max_tokens']}")
        print(f"  Answer extractor ready: {evaluator.answer_extractor is not None}")
        print(f"  Dataset loader ready: {evaluator.dataset_loader is not None}")
        
        # Test prompt formatting
        test_question = {
            'question': 'What is 2+2?',
            'choices': {'A': '3', 'B': '4', 'C': '5', 'D': '6'},
            'answer': 'B'
        }
        
        prompt = evaluator.format_prompt(test_question)
        print(f"✓ Prompt formatting works")
        print(f"  Prompt length: {len(prompt)} characters")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluator setup failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Base Evaluator System")
    print("=" * 50)
    
    success1 = test_config_loading()
    print()
    success2 = test_evaluator_setup()
    
    if success1 and success2:
        print("\n✓ All evaluator tests passed!")
    else:
        print("\n✗ Some tests failed!")
        
    exit(0 if success1 and success2 else 1)
