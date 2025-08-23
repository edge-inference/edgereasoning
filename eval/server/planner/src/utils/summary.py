"""
Summary handler that extends the base Summary class
Handles GPU telemetry and performance metrics in addition to evaluation results
"""
import os
from datetime import datetime
from utils.summary import Summary

class PerfSummary(Summary):
    """Extended summary class for performance evaluations with GPU and performance metrics"""

    def __init__(self, args, model_name):
        super().__init__(args, model_name)
        self.gpu_stats = {}
        self.perf_stats = {}
    
    def add_gpu_telemetry(self, gpu_stats):
        """Add GPU telemetry statistics"""
        self.gpu_stats = gpu_stats
    
    def add_performance_metrics(self, perf_stats):
        """Add performance metrics"""
        self.perf_stats = perf_stats
    
    def print_gpu_summary(self):
        """Print GPU telemetry summary to console"""
        if not self.gpu_stats:
            return
            
        print("\nGPU Telemetry Summary:")
        for gpu_key, stats in self.gpu_stats.items():
            print(f"  {gpu_key.upper()}:")
            print(f"    Avg Temperature: {stats.get('avg_temp_c', 0):.1f}°C")
            print(f"    Avg Power: {stats.get('avg_power_w', 0):.1f}W")
            print(f"    Total Energy: {stats.get('total_energy_j', 0):.1f}J")
            print(f"    Avg GPU Util: {stats.get('avg_gpu_util_pct', 0):.1f}%")
    
    def print_performance_summary(self):
        """Print performance metrics summary to console"""
        if not self.perf_stats:
            return
            
        print(f"\nPerformance Summary:")
        print(f"  Total batches: {self.perf_stats.get('total_batches', 0)}")
        print(f"  Total questions: {self.perf_stats.get('total_questions', 0)}")
        print(f"  Total duration: {self.perf_stats.get('total_duration', 0):.2f}s")
        print(f"  Overall throughput: {self.perf_stats.get('overall_questions_per_second', 0):.2f} questions/s")
        print(f"  Overall token throughput: {self.perf_stats.get('overall_tokens_per_second', 0):.2f} tokens/s")
        print(f"  Avg completion tokens: {self.perf_stats.get('avg_completion_tokens_per_question', 0):.1f}")
    
    def print_all_summaries(self):
        """Print all summary information to console"""
        self.print_gpu_summary()
        self.print_performance_summary()
    
    def save(self, output_file_path):
        """Save extended summary including GPU and performance metrics"""
        summary_dir = os.path.dirname(output_file_path)
        summary_file = os.path.join(summary_dir, 'summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write(f" Evaluation Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            # Configuration
            f.write("Configuration:\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Dataset: {self.args.data_name} ({self.args.split})\n")
            f.write(f"Temperature: {self.args.temperature}\n")
            f.write(f"Max Tokens: {self.args.max_tokens}\n")
            f.write(f"Batch Size: {self.args.batch_size}\n")
            f.write(f"Tensor Parallel Size: {getattr(self.args, 'tensor_parallel_size', 'N/A')}\n")
            f.write(f"N Sampling: {self.args.n_sampling}\n")
            f.write(f"Prompt Type: {self.args.prompt_type}\n\n")
            
            # Evaluation Results
            f.write("Evaluation Results:\n")
            f.write(f"Total Questions: {self.results.get('total_questions', 0)}\n")
            f.write(f"Correct: {self.results.get('correct_count', 0)}\n")
            f.write(f"Accuracy: {self.results.get('accuracy', 0):.4f}\n")
            if self.results.get('pass_at_k_score') is not None:
                f.write(f"Pass@{self.args.k}: {self.results['pass_at_k_score']:.4f}\n")
            f.write("\n")
            
            # Performance Metrics
            if self.perf_stats:
                f.write("Performance Metrics:\n")
                f.write(f"Total Batches: {self.perf_stats.get('total_batches', 0)}\n")
                f.write(f"Total Duration: {self.perf_stats.get('total_duration', 0):.2f}s\n")
                f.write(f"Overall Throughput: {self.perf_stats.get('overall_questions_per_second', 0):.2f} questions/s\n")
                f.write(f"Token Throughput: {self.perf_stats.get('overall_tokens_per_second', 0):.2f} tokens/s\n")
                f.write(f"Avg Completion Tokens: {self.perf_stats.get('avg_completion_tokens_per_question', 0):.1f}\n")
                f.write("\n")
            
            # GPU Telemetry Summary
            if self.gpu_stats:
                f.write("GPU Telemetry Summary:\n")
                for gpu_key, stats in self.gpu_stats.items():
                    f.write(f"{gpu_key.upper()}:\n")
                    f.write(f"  Avg Temperature: {stats.get('avg_temp_c', 0):.1f}°C\n")
                    f.write(f"  Max Temperature: {stats.get('max_temp_c', 0):.1f}°C\n")
                    f.write(f"  Avg Power: {stats.get('avg_power_w', 0):.1f}W\n")
                    f.write(f"  Max Power: {stats.get('max_power_w', 0):.1f}W\n")
                    f.write(f"  Total Energy: {stats.get('total_energy_j', 0):.1f}J\n")
                    f.write(f"  Avg GPU Util: {stats.get('avg_gpu_util_pct', 0):.1f}%\n")
                    f.write(f"  Avg Memory Util: {stats.get('avg_memory_util_pct', 0):.1f}%\n")
                    f.write("\n")
        
        print(f" summary saved to: {summary_file}")