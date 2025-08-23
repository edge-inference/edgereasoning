"""
Professional telemetry monitoring for VLLM performance evaluation.

This module provides comprehensive performance and energy monitoring
using the proven instrumentation from our tegra_monitor system.
"""

import os
import csv
import time
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from contextlib import contextmanager
from .telemetry_proc import TelemetryProcessor

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    model_name: str
    config_name: str
    evaluation_type: str
    
    # Timing metrics
    total_questions: int = 0
    total_time_ms: float = 0.0
    avg_time_per_question_ms: float = 0.0
    
    # Token metrics
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_tokens_per_second: float = 0.0
    
    # Individual question metrics
    question_metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_question_metric(self, question_id: int, result_data: Dict[str, Any]) -> None:
        """Add metrics for a single question."""
        self.question_metrics.append({
            'question_id': question_id,
            'timestamp': datetime.now().isoformat(),
            **result_data
        })
        
        # Update totals
        self.total_questions += 1
        self.total_time_ms += result_data.get('total_time_ms', 0)
        self.total_input_tokens += result_data.get('input_tokens', 0)
        self.total_output_tokens += result_data.get('output_tokens', 0)
        
        # Update averages
        if self.total_questions > 0:
            self.avg_time_per_question_ms = self.total_time_ms / self.total_questions
            
        total_time_seconds = self.total_time_ms / 1000
        if total_time_seconds > 0:
            self.avg_tokens_per_second = self.total_output_tokens / total_time_seconds


class TelemetryMonitor:
    """
    Professional telemetry monitor integrating proven instrumentation.
    
    Features:
    - Tegrastats energy monitoring
    - Performance metrics collection
    - CSV output for analysis
    - Context manager for clean resource handling
    """
    
    def __init__(self, output_dir: str, run_name: str):
        """
        Initialize telemetry monitor.
        
        Args:
            output_dir: Directory for output files
            run_name: Unique name for this run
        """
        self.output_dir = output_dir
        self.run_name = run_name
        self.tegrastats_log = os.path.join(output_dir, f"tegrastats_{run_name}.log")
        self.performance_csv = os.path.join(output_dir, f"performance_{run_name}.csv")
        self.energy_csv = os.path.join(output_dir, f"energy_{run_name}.csv")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Performance tracking
        self.metrics = None
        self._start_time = None
        
    def start_monitoring(self, model_name: str, config_name: str, evaluation_type: str) -> None:
        """Start telemetry monitoring."""
        self.metrics = PerformanceMetrics(
            model_name=model_name,
            config_name=config_name,
            evaluation_type=evaluation_type
        )
        self._start_time = time.time()
        
        # Start tegrastats
        self._start_tegrastats()
        print(f"Started telemetry monitoring: {self.run_name}")
        
    def stop_monitoring(self) -> PerformanceMetrics:
        """Stop monitoring and save results."""
        if not self.metrics:
            raise RuntimeError("Monitoring not started")
            
        # Stop tegrastats
        self._stop_tegrastats()
        
        # Process telemetry data
        self._process_telemetry()
        
        # Save performance metrics
        self._save_performance_metrics()
        
        # Print summary
        self._print_summary()
        
        print(f"Stopped telemetry monitoring: {self.run_name}")
        return self.metrics
        
    def record_question_result(self, question_id: int, prediction_result: Any) -> None:
        """Record comprehensive metrics for a single question."""
        if not self.metrics:
            raise RuntimeError("Monitoring not started")
            
        # Extract comprehensive metrics from prediction result (matching AIME format)
        result_data = {
            # Basic token counts
            'input_tokens': getattr(prediction_result, 'input_tokens', 0),
            'output_tokens': getattr(prediction_result, 'output_tokens', 0),
            
            # Detailed timing metrics (AIME format)
            'ttft': getattr(prediction_result, 'ttft', 0),
            'decode_time': getattr(prediction_result, 'decode_time', 0),
            'total_time_ms': getattr(prediction_result, 'total_time_ms', 0),
            'tokens_per_second': getattr(prediction_result, 'tokens_per_second', 0),
            
            # Additional metrics
            'predicted_choice': getattr(prediction_result, 'predicted_choice', ''),
            'generated_text_length': len(getattr(prediction_result, 'generated_text', '')),
            
            # Aliases for compatibility
            'prompt_tokens': getattr(prediction_result, 'input_tokens', 0),
            'completion_tokens': getattr(prediction_result, 'output_tokens', 0),
            'tokens_generated': getattr(prediction_result, 'output_tokens', 0),
        }
        
        self.metrics.add_question_metric(question_id, result_data)
        
    def _start_tegrastats(self) -> None:
        """Start tegrastats logging."""
        try:
            # Stop any existing tegrastats
            subprocess.run(['tegrastats', '--stop'], capture_output=True, check=False)
            time.sleep(1)
            
            # Start new logging
            subprocess.run([
                'tegrastats', '--start', '--interval', '1000', 
                '--logfile', self.tegrastats_log
            ], check=True)
            time.sleep(2)  # Give it time to start
            
        except Exception as e:
            print(f"Warning: Failed to start tegrastats: {e}")
            
    def _stop_tegrastats(self) -> None:
        """Stop tegrastats logging."""
        try:
            subprocess.run(['tegrastats', '--stop'], capture_output=True, check=True)
            time.sleep(2)  
            
            if os.path.exists(self.tegrastats_log):
                file_size = os.path.getsize(self.tegrastats_log)
                print(f"Tegrastats log created: {file_size} bytes")
            else:
                print("Warning: Tegrastats log file not found")
                
        except Exception as e:
            print(f"Warning: Failed to stop tegrastats: {e}")
            
    def _process_telemetry(self) -> None:
        """Process tegrastats log into energy CSV."""
        if not os.path.exists(self.tegrastats_log):
            print("Warning: No tegrastats log to process")
            return
            
        try:
            processor = TelemetryProcessor(
                self.tegrastats_log,
                self.output_dir,
                self.run_name
            )
            success = processor.process_log()
            
            if success:
                print(f"Telemetry processing completed: {self.energy_csv}")
            else:
                print("Warning: Telemetry processing failed")
                
        except ImportError:
            print("Warning: Telemetry processor not available - skipping energy analysis")
        except Exception as e:
            print(f"Warning: Telemetry processing failed: {e}")
            
    def _save_performance_metrics(self) -> None:
        """Save comprehensive performance metrics to CSV (AIME compatible format)."""
        if not self.metrics or not self.metrics.question_metrics:
            return
            
        with open(self.performance_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header (matching AIME evaluation format)
            writer.writerow([
                'timestamp', 'model_name', 'config_name', 'evaluation_type',
                'question_id', 'ttft', 'decode_time', 'total_time_ms', 
                'tokens_generated', 'tokens_per_second', 'prompt_tokens', 
                'completion_tokens', 'predicted_choice', 'generated_text_length'
            ])
            
            # Data rows
            for metric in self.metrics.question_metrics:
                writer.writerow([
                    metric['timestamp'],
                    self.metrics.model_name,
                    self.metrics.config_name,
                    self.metrics.evaluation_type,
                    metric['question_id'],
                    metric.get('ttft', 0),
                    metric.get('decode_time', 0),
                    metric['total_time_ms'],
                    metric.get('tokens_generated', metric['output_tokens']),
                    metric['tokens_per_second'],
                    metric.get('prompt_tokens', metric['input_tokens']),
                    metric.get('completion_tokens', metric['output_tokens']),
                    metric['predicted_choice'],
                    metric['generated_text_length']
                ])
                
        print(f"Performance metrics saved (AIME format): {self.performance_csv}")
        
    def _print_summary(self) -> None:
        """Print performance summary."""
        if not self.metrics:
            return
            
        print(f"\n=== Performance Summary ===")
        print(f"Model: {self.metrics.model_name}")
        print(f"Config: {self.metrics.config_name}")
        print(f"Evaluation: {self.metrics.evaluation_type}")
        print(f"Total Questions: {self.metrics.total_questions}")
        print(f"Total Time: {self.metrics.total_time_ms:.0f}ms")
        print(f"Avg Time/Question: {self.metrics.avg_time_per_question_ms:.1f}ms")
        print(f"Total Input Tokens: {self.metrics.total_input_tokens}")
        print(f"Total Output Tokens: {self.metrics.total_output_tokens}")
        print(f"Avg Tokens/Second: {self.metrics.avg_tokens_per_second:.1f}")
        print(f"=========================")


@contextmanager
def monitor_evaluation(output_dir: str, run_name: str, model_name: str, 
                      config_name: str, evaluation_type: str):
    """Context manager for telemetry monitoring."""
    monitor = TelemetryMonitor(output_dir, run_name)
    
    try:
        monitor.start_monitoring(model_name, config_name, evaluation_type)
        yield monitor
    finally:
        monitor.stop_monitoring()
