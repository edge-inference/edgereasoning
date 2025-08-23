#!/usr/bin/env python3
"""
Lightweight test of prompt templating logic.

Tests the format_prompt method logic without instantiating heavy VLLM components.
"""

import os
import yaml

def test_budget_config_parsing():
    """Test that budget config has correct templating structure."""
    print('=== BUDGET CONFIG PARSING TEST ===')
    
    try:
        budget_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'np_budget.yaml')
        with open(budget_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Loading config from: {budget_config_path}")
        
        # Check required fields exist
        assert 'model' in config, "Missing 'model' section"
        assert 'max_tokens' in config['model'], "Missing 'max_tokens' in model section"
        assert 'prompting' in config, "Missing 'prompting' section" 
        assert 'system_prompt' in config['prompting'], "Missing 'system_prompt'"
        assert 'user_template' in config['prompting'], "Missing 'user_template'"
        
        max_tokens = config['model']['max_tokens']
        system_prompt = config['prompting']['system_prompt']
        user_template = config['prompting']['user_template']
        
        print(f"Max tokens: {max_tokens}")
        print(f"System prompt template: {system_prompt}")
        print(f"User template: {user_template}")
        
        # Check that templates contain {max_tokens} placeholder
        assert '{max_tokens}' in system_prompt, "system_prompt missing {max_tokens} placeholder"
        assert '{max_tokens}' in user_template, "user_template missing {max_tokens} placeholder"
        assert '{prompt}' in user_template, "user_template missing {prompt} placeholder"
        
        print("✅ Budget config structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_scaling_config_parsing():
    """Test that scaling config has quality-focused templating."""
    print('\n=== SCALING CONFIG PARSING TEST ===')
    
    try:
        scaling_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'np_scaling.yaml')
        with open(scaling_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Loading config from: {scaling_config_path}")
        
        # Check scaling-specific fields
        assert 'scaling' in config, "Missing 'scaling' section"
        assert 'num_samples' in config['scaling'], "Missing 'num_samples' in scaling section"
        
        num_samples = config['scaling']['num_samples']
        print(f"Scaling samples: {num_samples}")
        
        # Check prompting section
        if 'prompting' in config:
            system_prompt = config['prompting']['system_prompt']
            user_template = config['prompting']['user_template']
            
            print(f"System prompt: {system_prompt}")
            print(f"User template: {user_template}")
            
            # Check for budget constraints (should NOT be present)
            budget_words = ['budget', 'limit', 'tokens', 'maximum', 'concise']
            combined_text = (system_prompt + ' ' + user_template).lower()
            
            found_budget_words = [word for word in budget_words if word in combined_text]
            
            if found_budget_words:
                print(f"⚠️  WARNING: Found budget-related words in scaling config: {found_budget_words}")
                return False
            else:
                print("✅ Scaling config is quality-focused (no budget constraints)")
                return True
        else:
            print("✅ Scaling config has no custom prompting (will use original prompts)")
            return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_template_replacement_logic():
    """Test the actual template replacement logic."""
    print('\n=== TEMPLATE REPLACEMENT LOGIC TEST ===')
    
    try:
        # Test budget template replacement
        max_tokens = 128
        system_prompt = "Be concise and direct. You must limit your answer to exactly {max_tokens} tokens or fewer."
        user_template = "TOKEN BUDGET: {max_tokens} tokens maximum\n\n{prompt}\n\nRemember: Stay within {max_tokens} tokens total."
        example_prompt = "Plan a meeting for 5 people"
        
        # This simulates what the evaluator should do
        system_formatted = system_prompt.format(max_tokens=max_tokens)
        user_formatted = user_template.format(max_tokens=max_tokens, prompt=example_prompt)
        combined = f"{system_formatted}\n\n{user_formatted}"
        
        print("FORMATTED OUTPUT:")
        print(combined)
        
        # Verify replacement
        assert '{max_tokens}' not in combined, "{max_tokens} placeholder not replaced"
        assert str(max_tokens) in combined, f"max_tokens value {max_tokens} not found in output"
        assert example_prompt in combined, "Original prompt not found in output"
        
        print("✅ Template replacement logic works correctly")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("Testing prompt templating logic...")
    
    budget_ok = test_budget_config_parsing()
    scaling_ok = test_scaling_config_parsing()
    template_ok = test_template_replacement_logic()
    
    print(f"\n{'='*60}")
    print("TEMPLATING TEST SUMMARY:")
    print(f"Budget config: {'✅ PASS' if budget_ok else '❌ FAIL'}")
    print(f"Scaling config: {'✅ PASS' if scaling_ok else '❌ FAIL'}")
    print(f"Template logic: {'✅ PASS' if template_ok else '❌ FAIL'}")
    
    all_passed = budget_ok and scaling_ok and template_ok
    print(f"Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if not all_passed:
        exit(1)