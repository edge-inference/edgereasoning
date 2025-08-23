"""
VLLM Model Wrapper - Professional model management for performance evaluation.

This module provides a clean interface for VLLM model operations with built-in
performance monitoring and standardized prediction results.
"""

import time
from dataclasses import dataclass
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
    enforce_eager: bool = False


@dataclass
class PredictionResult:
    """Standardized prediction result with comprehensive timing metrics."""
    predicted_choice: str
    generated_text: str
    
    # Token counts
    input_tokens: int  # prompt_tokens
    output_tokens: int  # completion_tokens
    
    ttft: float 
    decode_time: float 
    total_time_ms: float  
    tokens_per_second: float 
    last_token_time: float = 0.0  
    
    # Optional fields (must come last in dataclass)
    generated_texts: Optional[List[str]] = None  #Support multiple sequences from n parameter
    raw_completion: Any = None 
    
    # Legacy aliases for compatibility
    @property
    def prompt_tokens(self) -> int:
        """Alias for input_tokens to match AIME format."""
        return self.input_tokens
    
    @property
    def completion_tokens(self) -> int:
        """Alias for output_tokens to match AIME format."""
        return self.output_tokens
    
    @property
    def tokens_generated(self) -> int:
        """Alias for output_tokens to match AIME format."""
        return self.output_tokens
    
    @property
    def total_time(self) -> float:
        """Total time in seconds (for AIME compatibility)."""
        return self.total_time_ms / 1000.0


class VLLMModel:
    """
    Professional VLLM model wrapper with performance monitoring.
    
    Features:
    - Efficient VLLM integration
    - Automatic token counting
    - Performance metrics collection
    - Standardized result format
    """
    
    def __init__(self, config: VLLMConfig):
        """Initialize VLLM model with configuration."""
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
        model_kwargs = {
            "model": self.config.model_path,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "trust_remote_code": self.config.trust_remote_code,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "dtype": self.config.dtype,
            "disable_log_stats": False,
            "show_hidden_metrics_for_version": "0.8.6", 
            "collect_detailed_traces": "all",
            "enforce_eager": self.config.enforce_eager
        }
        
        if self.config.max_model_len is not None:
            model_kwargs["max_model_len"] = self.config.max_model_len
            
        self.model = LLM(**model_kwargs)
        print(f"Model loaded successfully")
        
    def predict(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        min_tokens: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs
    ) -> PredictionResult:
        """
        Generate prediction with vLLM metrics.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            min_tokens: Minimum tokens to generate
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
            min_tokens=min_tokens,
            top_p=top_p,
            **kwargs
        )
        
        # Generate with vLLM (metrics collection enabled in model loading)
        completions = self.model.generate([prompt], sampling_params)
        
        # Validate that we got completions
        if not completions:
            raise RuntimeError("No completions returned from vLLM model")
        
        completion = completions[0]
        
        # Validate that we got outputs
        if not completion.outputs:
            raise RuntimeError("No outputs in completion from vLLM model")
        
        # Extract ALL generated texts (for n > 1 support)
        generated_texts = [output.text for output in completion.outputs]
        generated_text = generated_texts[0]  # Primary text (backward compatibility)
        
        # Calculate total output tokens across all sequences
        total_output_tokens = sum(len(output.token_ids) for output in completion.outputs)
        
        # Token counts
        input_tokens = len(completion.prompt_token_ids)
        output_tokens = total_output_tokens
        
        # Extract real metrics from vLLM - handle n > 1 case where some metrics may be None
        metrics = completion.metrics
        if metrics is None:
            raise RuntimeError("No metrics available. Ensure model was loaded with correct parameters")
        
        # Calculate real timing metrics (in seconds) - handle None values for n > 1
        arrival_time = getattr(metrics, 'arrival_time', None) or 0
        first_token_time = getattr(metrics, 'first_token_time', None)
        last_token_time = getattr(metrics, 'last_token_time', None) 
        finished_time = getattr(metrics, 'finished_time', None)
        
        # Calculate timing with None checks (common when n > 1)
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
            
        if last_token_time is not None and arrival_time is not None:
            last_token_time = last_token_time - arrival_time
        else:
            last_token_time = 0
        
        # Calculate tokens per second
        tokens_per_second = output_tokens / total_time if total_time > 0 else 0
        
        # Convert to milliseconds for consistency with existing API
        ttft_ms = ttft * 1000
        decode_time_ms = decode_time * 1000
        total_time_ms = total_time * 1000
        last_token_time_ms = last_token_time * 1000
        
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
            last_token_time=last_token_time_ms,
            raw_completion=completion
        )
    
    def predict_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        min_tokens: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs
    ) -> List[PredictionResult]:
        """
        Generate batch predictions with vLLM metrics.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            min_tokens: Minimum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional sampling parameters
            
        Returns:
            List of PredictionResult objects with metrics per prompt
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
            
        # For single prompt, use individual predict for consistency
        if len(prompts) == 1:
            return [self.predict(prompts[0], max_tokens, min_tokens, temperature, top_p, **kwargs)]
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            top_p=top_p,
            **kwargs
        )
        
        # Generate batch with real metrics collection
        completions = self.model.generate(prompts, sampling_params)
        
        # Validate that we got completions
        if not completions:
            raise RuntimeError("No completions returned from vLLM model")
        
        # Process results with vllm metrics
        results = []
        for completion in completions:
            # Validate that we got outputs
            if not completion.outputs:
                raise RuntimeError("No outputs in completion from vLLM model")
            
            # Extract real metrics
            metrics = completion.metrics
            if metrics is None:
                raise RuntimeError("No metrics available. Ensure V0 engine is being used.")
            
            # Extract generated text and tokens
            generated_text = completion.outputs[0].text
            output_token_ids = completion.outputs[0].token_ids
            
            # Calculate real timing metrics (in seconds)
            ttft = metrics.first_token_time - metrics.arrival_time
            decode_time = metrics.last_token_time - metrics.first_token_time
            total_time = metrics.finished_time - metrics.arrival_time
            
            # Token counts
            input_tokens = len(completion.prompt_token_ids)
            output_tokens = len(output_token_ids)
            
            # Calculate tokens per second
            tokens_per_second = output_tokens / total_time if total_time > 0 else 0
            
            # Convert to milliseconds
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
                raw_completion=completion
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
    
    def cleanup(self) -> None:
        """
        Properly cleanup the model and free GPU memory.
        This is important for preventing memory profiling errors in vLLM.
        """
        if self.model is not None:
            try:
                # Destroy the vLLM engine
                if hasattr(self.model, 'llm_engine') and self.model.llm_engine is not None:
                    self.model.llm_engine.model_executor.shutdown()
                del self.model
                self.model = None
            except Exception as e:
                print(f"[WARN] Error during model cleanup: {e}")
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        # Force garbage collection and GPU memory cleanup
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        self.cleanup()
