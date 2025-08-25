"""
Model Cleanup Module
Handles proper cleanup of model resources, distributed processes, and memory
"""

import gc
import logging
from typing import Optional, Any
import warnings

logger = logging.getLogger(__name__)


class ModelCleanupManager:
    """Centralized cleanup manager for model resources"""
    
    def __init__(self):
        self.cleanup_hooks = []
        self.models_to_cleanup = []
    
    def register_model(self, model: Any):
        """Register a model for cleanup"""
        self.models_to_cleanup.append(model)
    
    def register_cleanup_hook(self, cleanup_func):
        """Register a custom cleanup function"""
        self.cleanup_hooks.append(cleanup_func)
    
    def cleanup_distributed(self):
        """Clean up PyTorch distributed processes"""
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                logger.info("Destroying PyTorch distributed process group")
                dist.destroy_process_group()
        except (ImportError, RuntimeError) as e:
            logger.debug(f"No distributed processes to clean up: {e}")
    
    def cleanup_cuda(self):
        """Clean up CUDA memory and cache"""
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Clearing CUDA cache")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            logger.debug("PyTorch not available, skipping CUDA cleanup")
    
    def cleanup_vllm_model(self, model: Any):
        """Clean up vLLM model resources"""
        try:
            if hasattr(model, 'llm_engine') and model.llm_engine is not None:
                logger.info("Cleaning up vLLM engine")
                if hasattr(model.llm_engine, 'model_executor'):
                    del model.llm_engine.model_executor
                del model.llm_engine
            
            if hasattr(model, 'model'):
                del model.model
                
        except Exception as e:
            logger.warning(f"Error during vLLM model cleanup: {e}")
    
    def cleanup_all_models(self):
        """Clean up all registered models"""
        for model in self.models_to_cleanup:
            try:
                if hasattr(model, 'cleanup'):
                    model.cleanup()
                else:
                    if hasattr(model, 'llm_engine') or hasattr(model, 'model'):
                        self.cleanup_vllm_model(model)
            except Exception as e:
                logger.warning(f"Error cleaning up model: {e}")
        
        self.models_to_cleanup.clear()
    
    def run_cleanup_hooks(self):
        """Run all registered cleanup hooks"""
        for hook in self.cleanup_hooks:
            try:
                hook()
            except Exception as e:
                logger.warning(f"Error in cleanup hook: {e}")
    
    def force_garbage_collection(self):
        """Force garbage collection"""
        logger.info("Running garbage collection")
        gc.collect()
    
    def cleanup_all(self):
        """Run complete cleanup sequence"""
        logger.info("Starting comprehensive cleanup")
        
        self.cleanup_all_models()
        
        self.run_cleanup_hooks()
        
        self.cleanup_distributed()
        
        self.cleanup_cuda()
        
        self.force_garbage_collection()
        
        logger.info("Cleanup completed")


_global_cleanup_manager = None

def get_cleanup_manager() -> ModelCleanupManager:
    """Get the global cleanup manager instance"""
    global _global_cleanup_manager
    if _global_cleanup_manager is None:
        _global_cleanup_manager = ModelCleanupManager()
    return _global_cleanup_manager


def register_model_for_cleanup(model: Any):
    """Convenience function to register a model for cleanup"""
    get_cleanup_manager().register_model(model)


def register_cleanup_hook(cleanup_func):
    """Convenience function to register a cleanup hook"""
    get_cleanup_manager().register_cleanup_hook(cleanup_func)


def cleanup_all():
    """Convenience function to run complete cleanup"""
    get_cleanup_manager().cleanup_all()


def setup_cleanup_handlers():
    """Set up automatic cleanup handlers for signals and exit"""
    import signal
    import atexit
    
    def cleanup_handler(signum=None, frame=None):
        """Handle cleanup on exit or signal"""
        try:
            cleanup_all()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    atexit.register(cleanup_handler)
    
    for sig in [signal.SIGINT, signal.SIGTERM]:
        try:
            signal.signal(sig, cleanup_handler)
        except (ValueError, OSError):
            pass


class CleanupContext:
    """Context manager for automatic cleanup"""
    
    def __init__(self, model: Optional[Any] = None):
        self.model = model
        self.cleanup_manager = get_cleanup_manager()
    
    def __enter__(self):
        if self.model:
            self.cleanup_manager.register_model(self.model)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model:
            try:
                if hasattr(self.model, 'cleanup'):
                    self.model.cleanup()
                else:
                    self.cleanup_manager.cleanup_vllm_model(self.model)
            except Exception as e:
                logger.warning(f"Error during context cleanup: {e}")


def suppress_distributed_warnings():
    """Suppress PyTorch distributed warnings"""
    warnings.filterwarnings(
        "ignore", 
        message=".*destroy_process_group.*",
        category=UserWarning
    )
