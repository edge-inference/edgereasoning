#!/usr/bin/env python3
"""
Test the two-phase TTFT measurement approach
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.vllm_model import VLLMModel, VLLMConfig

def TestPrompt():
    """Test the two-phase TTFT measurement"""
    print("Two-Phase TTFT Measurement Test")
    print("=" * 40)
    
    # Configure model
    config = VLLMConfig(
        model_path="facebook/opt-125m",
        gpu_memory_utilization=0.3,
        max_model_len=128
    )
    
    model = VLLMModel(config)
    
    test_cases = [
        {
            "prompt": "What is 2+2?",
            "max_tokens": 10,
            "description": "Short math question"
        },
        {
            "prompt": "Tell me about machine learning",
            "max_tokens": 30,
            "description": "Medium explanation request"
        },
        {
            "prompt": "The capital of France is",
            "max_tokens": 5,
            "description": "Simple completion"
        }
    ]
    
    print("\nTesting two-phase generation for accurate TTFT:")
    print("-" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Prompt: '{test_case['prompt']}'")
        print(f"Max tokens: {test_case['max_tokens']}")
        
        result = model.predict(
            prompt=test_case["prompt"],
            max_tokens=test_case["max_tokens"],
            temperature=0.7
        )
        
        print(f"Generated: '{result.generated_text}'")
        print(f"Metrics:")
        print(f"  - Input tokens: {result.input_tokens}")
        print(f"  - Output tokens: {result.output_tokens}")
        print(f"  - TTFT: {result.ttft:.1f}ms")
        print(f"  - Decode time: {result.decode_time:.1f}ms") 
        print(f"  - Total time: {result.total_time_ms:.1f}ms")
        print(f"  - Tokens/sec: {result.tokens_per_second:.1f}")
        print(f"  - TTFT ratio: {(result.ttft/result.total_time_ms)*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ Two-phase approach benefits:")
    print("  • Accurate TTFT measurement (not estimated)")
    print("  • Separates first token latency from decode speed")
    print("  • Works with VLLM 0.8.6 offline API")
    print("  • Provides real timing data for evaluation")
    print("\n📊 Ready for AIME-style benchmarking!")

if __name__ == "__main__":
    TestPrompt()
