#!/usr/bin/env python3
"""
Test VLLM 0.9.1 RequestMetrics - find the right way to enable them
"""
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from vllm import LLM, SamplingParams

def test_request_metrics_v0_engine():
    """Try using V0 engine parameters from 0.8.6"""
    print("=== Testing V0 Engine RequestMetrics (0.8.6 style) ===")
    
    try:
        # Use the exact parameters from your 0.8.6 working code
        llm = LLM(
            model="microsoft/DialoGPT-small", 
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.3,
            disable_log_stats=False,
            show_hidden_metrics_for_version="0.9.0",  # Updated for 0.9.1
            collect_detailed_traces="all",
        )
        
        prompt = "What is the capital of France?"
        params = SamplingParams(temperature=0.7, max_tokens=16, n=1)
        completions = llm.generate([prompt], params)
        
        completion = completions[0]
        print(f"Completion object attributes: {[attr for attr in dir(completion) if not attr.startswith('_')]}")
        print(f"Completion metrics: {completion.metrics}")
        print(f"Metrics type: {type(completion.metrics)}")
        
        # Check outputs for metrics too
        if hasattr(completion, 'outputs') and completion.outputs:
            output = completion.outputs[0]
            print(f"Output attributes: {[attr for attr in dir(output) if not attr.startswith('_')]}")
            output_metrics = getattr(output, "metrics", None)
            print(f"Output metrics: {output_metrics}")
        
        # Check for alternative metric attributes
        for attr_name in ['timing', 'stats', 'request_metrics', 'generation_metrics']:
            attr_val = getattr(completion, attr_name, None)
            if attr_val:
                print(f"Found {attr_name}: {attr_val}")
        
        if completion.metrics:
            print("✅ RequestMetrics found!")
            # Explore the metrics structure (your original code)
            m = completion.metrics
            print("RequestMetrics attributes:")
            for attr in dir(m):
                if not attr.startswith('_'):
                    try:
                        value = getattr(m, attr)
                        print(f"  {attr}: {value}")
                    except:
                        pass
            
            # Calculate the exact metrics from your 0.8.6 code
            if hasattr(m, 'first_token_time') and hasattr(m, 'arrival_time'):
                ttft = m.first_token_time - m.arrival_time
                decode_time = m.last_token_time - m.first_token_time
                total_time = m.finished_time - m.arrival_time
                tokens_generated = len(completion.outputs[0].token_ids)
                tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
                prompt_tokens = len(completion.prompt_token_ids)
                completion_tokens = len(completion.outputs[0].token_ids)
                
                print(f"\n🎯 VLLM Native Metrics (like 0.8.6):")
                print(f"  TTFT: {ttft:.4f} seconds")
                print(f"  Decode time: {decode_time:.4f} seconds")
                print(f"  Total time: {total_time:.4f} seconds")
                print(f"  Tokens generated: {tokens_generated}")
                print(f"  Tokens per second: {tokens_per_second:.2f}")
                print(f"  Prompt tokens: {prompt_tokens}")
                print(f"  Completion tokens: {completion_tokens}")
                
            return True
        else:
            print("❌ Metrics still None")
            return False
            
    except Exception as e:
        print(f"❌ V0 engine test failed: {e}")
        return False

def test_with_detailed_profiling():
    """Try enabling detailed profiling"""
    print("\n=== Testing with Detailed Profiling ===")
    
    try:
        from vllm.config import ObservabilityConfig
        
        # Try with observability config
        obs_config = ObservabilityConfig(
            collect_detailed_traces=True,
            show_hidden_metrics_for_version="0.9.0"  # Show metrics
        )
        
        # This might not work directly with LLM constructor
        print("ObservabilityConfig created")
        
    except Exception as e:
        print(f"ObservabilityConfig failed: {e}")
    
    try:
        # Try with engine args approach
        from vllm.engine.arg_utils import EngineArgs
        
        args = EngineArgs(
            model="microsoft/DialoGPT-small",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.3
        )
        
        # Check if args has observability settings
        print(f"EngineArgs attributes: {[attr for attr in dir(args) if 'observ' in attr.lower() or 'metric' in attr.lower() or 'profile' in attr.lower()]}")
        
        # Try to enable profiling in args
        if hasattr(args, 'enable_profiling'):
            args.enable_profiling = True
        if hasattr(args, 'profile'):
            args.profile = True
            
        return False
        
    except Exception as e:
        print(f"EngineArgs profiling failed: {e}")
        return False

def test_environment_variables():
    """Try environment variables to enable metrics"""
    print("\n=== Testing Environment Variables ===")
    
    # Set various environment variables that might enable metrics
    os.environ['VLLM_ENABLE_METRICS'] = 'true'
    os.environ['VLLM_PROFILE'] = 'true'
    os.environ['VLLM_DETAILED_METRICS'] = 'true'
    os.environ['VLLM_REQUEST_METRICS'] = 'true'
    
    try:
        llm = LLM(
            model="microsoft/DialoGPT-small", 
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.3
        )
        
        prompt = "Hello world"
        params = SamplingParams(temperature=0.5, max_tokens=8, n=1)
        completions = llm.generate([prompt], params)
        
        completion = completions[0]
        print(f"Completion metrics with env vars: {completion.metrics}")
        
        if completion.metrics:
            print("✅ Environment variables worked!")
            return True
        else:
            print("❌ Environment variables didn't work")
            return False
            
    except Exception as e:
        print(f"❌ Environment variables test failed: {e}")
        return False

def test_direct_v0_import():
    """Try importing V0 engine directly"""
    print("\n=== Testing Direct V0 Engine Import ===")
    
    try:
        # Try to import V0 engine classes
        from vllm.engine.llm_engine import LLMEngine
        from vllm.engine.arg_utils import EngineArgs
        
        print("✅ Successfully imported LLMEngine and EngineArgs")
        
        # Create engine args
        args = EngineArgs(
            model="microsoft/DialoGPT-small",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.3,
            disable_log_stats=False,  # Try to enable stats
        )
        
        # Create engine directly
        engine = LLMEngine.from_engine_args(args)
        print("✅ Created LLMEngine directly")
        
        # Check if this gives us metrics
        from vllm.inputs import TextOnlyInput
        from vllm.sampling_params import SamplingParams
        
        # Add request to engine
        request_id = "test_request"
        inputs = TextOnlyInput(prompt="What is machine learning?")
        params = SamplingParams(temperature=0.5, max_tokens=16)
        
        engine.add_request(request_id, inputs, params)
        
        # Step the engine
        request_outputs = engine.step()
        
        if request_outputs:
            for output in request_outputs:
                print(f"Direct engine output metrics: {output.metrics}")
                if output.metrics:
                    print("✅ Direct engine approach worked!")
                    # Explore metrics
                    for attr in dir(output.metrics):
                        if not attr.startswith('_'):
                            try:
                                value = getattr(output.metrics, attr)
                                print(f"  {attr}: {value}")
                            except:
                                pass
                    return True
        
        print("❌ Direct engine approach didn't provide metrics")
        return False
        
    except Exception as e:
        print(f"❌ Direct V0 engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_vllm_help():
    """Check VLLM help and documentation"""
    print("\n=== Checking VLLM Documentation ===")
    
    try:
        # Check if there are command line options we can use
        from vllm.engine.arg_utils import EngineArgs
        import argparse
        
        parser = argparse.ArgumentParser()
        args = EngineArgs.add_cli_args(parser)
        
        # Look for metric-related arguments
        help_text = parser.format_help()
        metric_lines = [line for line in help_text.split('\n') if any(keyword in line.lower() for keyword in ['metric', 'profile', 'observ', 'stat', 'trace'])]
        
        print("Metric-related CLI arguments:")
        for line in metric_lines:
            print(f"  {line}")
            
        if metric_lines:
            return True
        else:
            print("❌ No metric-related CLI arguments found")
            return False
            
    except Exception as e:
        print(f"❌ Help check failed: {e}")
        return False

def test_version_numbers():
    """Try different version numbers for show_hidden_metrics_for_version"""
    print("\n=== Testing Different Version Numbers ===")
    
    versions_to_try = [
        "0.9.0",
        "0.9.1", 
        "0.8.6",
        "latest",
        None  # Try without version
    ]
    
    for version in versions_to_try:
        print(f"\n--- Trying version: {version} ---")
        try:
            kwargs = {
                'model': "microsoft/DialoGPT-small",
                'tensor_parallel_size': 1,
                'gpu_memory_utilization': 0.3,
                'disable_log_stats': False,
                'collect_detailed_traces': "all",
            }
            
            if version is not None:
                kwargs['show_hidden_metrics_for_version'] = version
                
            llm = LLM(**kwargs)
            
            prompt = "Hello"
            params = SamplingParams(temperature=0.5, max_tokens=8, n=1)
            completions = llm.generate([prompt], params)
            
            completion = completions[0]
            if completion.metrics:
                print(f"✅ SUCCESS with version {version}!")
                print(f"Metrics type: {type(completion.metrics)}")
                # Show a few key attributes
                m = completion.metrics
                for attr in ['arrival_time', 'first_token_time', 'finished_time']:
                    if hasattr(m, attr):
                        value = getattr(m, attr)
                        print(f"  {attr}: {value}")
                return version
            else:
                print(f"❌ No metrics with version {version}")
                
        except Exception as e:
            print(f"❌ Failed with version {version}: {e}")
    
    return None

if __name__ == "__main__":
    print("🔍 Searching for VLLM 0.9.1 RequestMetrics activation method...")
    print("=" * 60)
    
    success1 = test_request_metrics_v0_engine()
    success2 = test_with_detailed_profiling()
    success3 = test_environment_variables()
    success4 = test_direct_v0_import()
    success5 = check_vllm_help()
    success6 = test_version_numbers()
    
    print(f"\n=== RESULTS ===")
    print(f"V0 engine approach: {'✅' if success1 else '❌'}")
    print(f"Detailed profiling: {'✅' if success2 else '❌'}")
    print(f"Environment variables: {'✅' if success3 else '❌'}")
    print(f"Direct engine import: {'✅' if success4 else '❌'}")
    print(f"CLI arguments check: {'✅' if success5 else '❌'}")
    print(f"Version numbers test: {'✅' if success6 else '❌'}")
    
    if any([success1, success2, success3, success4, success6]):
        print(f"\n🎉 Found working approach for RequestMetrics!")
    else:
        print(f"\n🔍 Need to investigate further...")
        print(f"The RequestMetrics system exists but needs proper activation.")
