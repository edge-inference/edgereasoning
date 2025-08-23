"""
Minimal tests to confirm VLLM exposes true timing metrics (no estimation, no two-phase)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from vllm import LLM, SamplingParams

def test_single_completion_metrics():
    llm = LLM(model="microsoft/DialoGPT-small", tensor_parallel_size=1, gpu_memory_utilization=0.3)
    prompt = "What is the capital of France?"
    params = SamplingParams(temperature=0.7, max_tokens=16, n=1)
    completions = llm.generate([prompt], params)
    c = completions[0]
    m = getattr(c, "metrics", None)
    assert m is not None, "No metrics found on completion object"
    print("ttft:", m.first_token_time - m.arrival_time)
    print("decode_time:", m.last_token_time - m.first_token_time)
    print("total_time:", m.finished_time - m.arrival_time)
    print("tokens_generated:", len(c.outputs[0].token_ids))
    print("tokens_per_second:", len(c.outputs[0].token_ids)/(m.finished_time - m.arrival_time))
    print("prompt_tokens:", len(c.prompt))
    print("completion_tokens:", len(c.outputs[0].token_ids))
    print("batch_total_time:", m.finished_time - m.arrival_time)

def test_batch_metrics():
    llm = LLM(model="microsoft/DialoGPT-small", tensor_parallel_size=1, gpu_memory_utilization=0.3)
    prompts = ["Q1?", "Q2?", "Q3?"]
    params = SamplingParams(temperature=0.7, max_tokens=8, n=1)
    completions = llm.generate(prompts, params)
    times = [c.metrics for c in completions]
    batch_start = min(m.arrival_time for m in times)
    batch_end = max(m.finished_time for m in times)
    print("batch_total_time:", batch_end - batch_start)
    for i, c in enumerate(completions):
        print(f"completion {i} ttft:", c.metrics.first_token_time - c.metrics.arrival_time)

if __name__ == "__main__":
    print("Single completion metrics:")
    test_single_completion_metrics()
    print("\nBatch metrics:")
    test_batch_metrics()