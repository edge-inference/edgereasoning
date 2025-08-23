"""
VLLM Model Wrapper - model management for performance evaluation.

This module provides a clean interface for VLLM model operations with built-in
performance monitoring and standardized prediction results.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


@dataclass
class VLLMConfig:
    """Configuration for VLLM model initialization."""
    model_path: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    trust_remote_code: bool = True
    dtype: str = "bfloat16"
    max_model_len: Optional[int] = None


@dataclass
class PredictionResult:
    """Standardized prediction result with comprehensive timing metrics."""
    predicted_choice: str
    generated_text: str
    
    input_tokens: int
    output_tokens: int

    ttft: float
    decode_time: float
    total_time_ms: float
    tokens_per_second: float

    generated_texts: Optional[List[str]] = None

    raw_completion: Any = None
    prompt_token_ids: List[int] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    metrics: Any = None
    # aliases for compatibility
    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens
    
    @property
    def completion_tokens(self) -> int:
        return self.output_tokens
    
    @property
    def tokens_generated(self) -> int:
        return self.output_tokens
    
    @property
    def total_time(self) -> float:
        return self.total_time_ms / 1000.0


class VLLMModel:
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.model: Optional[LLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load VLLM model and tokenizer."""
        print(f"Loading tokenizer: {self.config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code
        )
        
        print(f"Loading VLLM model: {self.config.model_path}")
        # Configure model for server GPUs
        model_kwargs = {
            "model": self.config.model_path,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "trust_remote_code": self.config.trust_remote_code,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "dtype": self.config.dtype
        }
        
        if self.config.max_model_len is not None:
            model_kwargs["max_model_len"] = self.config.max_model_len
            
        self.model = LLM(**model_kwargs)
        print(f"Model loaded successfully")
        
    def predict(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs
    ) -> PredictionResult:
        """
        Generate prediction with vLLM metrics.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional sampling parameters (including 'n' for multiple sequences)
            
        Returns:
            PredictionResult with prediction and performance metrics
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
            
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        # Generate with vLLM
        completions = self.model.generate([prompt], sampling_params)
        
        if not completions:
            raise RuntimeError("No completions returned from vLLM model")
        
        completion = completions[0]
        
        if not completion.outputs:
            raise RuntimeError("No outputs in completion from vLLM model")
        
        # Extract ALL generated texts
        generated_texts = [output.text for output in completion.outputs]
        generated_text = generated_texts[0]
        
        total_output_tokens = sum(len(output.token_ids) for output in completion.outputs)
        
        input_tokens = len(completion.prompt_token_ids) if hasattr(completion, 'prompt_token_ids') else 0
        output_tokens = total_output_tokens
        
        metrics = getattr(completion, "metrics", None)
        if metrics is None:
            raise RuntimeError("No metrics found on completion object")
        
        arrival_time = getattr(metrics, 'arrival_time', None) or 0
        first_token_time = getattr(metrics, 'first_token_time', None)
        last_token_time = getattr(metrics, 'last_token_time', None) 
        finished_time = getattr(metrics, 'finished_time', None)
        
        if first_token_time is not None and arrival_time is not None:
            ttft = first_token_time - arrival_time
        else:
            ttft = 0
            
        if last_token_time is not None and first_token_time is not None:
            decode_time = last_token_time - first_token_time
        else:
            decode_time = 0
            
        if finished_time is not None and arrival_time is not None:
            total_time = finished_time - arrival_time
        else:
            total_time = 0
        
        # Calculate tokens per second
        tokens_per_second = output_tokens / total_time if total_time > 0 else 0
        
        ttft_ms = ttft * 1000
        decode_time_ms = decode_time * 1000
        total_time_ms = total_time * 1000
        
        return PredictionResult(
            predicted_choice="", 
            generated_text=generated_text,
            generated_texts=generated_texts,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft=ttft_ms,
            decode_time=decode_time_ms,
            total_time_ms=total_time_ms,
            tokens_per_second=tokens_per_second,
            raw_completion=completion,
            prompt_token_ids=completion.prompt_token_ids if hasattr(completion, 'prompt_token_ids') else [],
            token_ids=completion.outputs[0].token_ids,
            metrics=metrics
        )
    
    def predict_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs
    ) -> List[PredictionResult]:
        """
        Generate batch predictions with vLLM metrics.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional sampling parameters
            
        Returns:
            List of PredictionResult objects with metrics per prompt
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
            
        if len(prompts) == 1:
            return [self.predict(prompts[0], max_tokens, temperature, top_p, **kwargs)]
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        completions = self.model.generate(prompts, sampling_params)
        
        results = []
        for completion in completions:
            metrics = getattr(completion, "metrics", None)
            if metrics is None:
                raise RuntimeError("No metrics found on completion object")
            
            generated_text = completion.outputs[0].text
            output_token_ids = completion.outputs[0].token_ids
            
            ttft = metrics.first_token_time - metrics.arrival_time
            decode_time = metrics.last_token_time - metrics.first_token_time
            total_time = metrics.finished_time - metrics.arrival_time
            
            input_tokens = len(completion.prompt_token_ids) if hasattr(completion, 'prompt_token_ids') else 0
            output_tokens = len(output_token_ids)
            
            tokens_per_second = output_tokens / total_time if total_time > 0 else 0
            
            ttft_ms = ttft * 1000
            decode_time_ms = decode_time * 1000
            total_time_ms = total_time * 1000
            
            results.append(PredictionResult(
                predicted_choice="",
                generated_text=generated_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                ttft=ttft_ms,
                decode_time=decode_time_ms,
                total_time_ms=total_time_ms,
                tokens_per_second=tokens_per_second,
                raw_completion=completion,
                prompt_token_ids=completion.prompt_token_ids,
                token_ids=output_token_ids,
                metrics=metrics
            ))
        print(f"✅ Batch processing: {len(prompts)} prompts, avg TPS={sum(r.tokens_per_second for r in results)/len(results):.1f}")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        return {
            "model_path": self.config.model_path,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "dtype": self.config.dtype
        }
