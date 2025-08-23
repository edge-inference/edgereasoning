"""
CSV writer for streaming evaluation results.

This module provides a reusable CSV writer that flushes after each row,
ensuring data safety during long evaluations.
"""

import os
import csv
from typing import List, Dict, Any, Optional
from contextlib import contextmanager


class StreamingCSVWriter:
    """
    Streaming CSV writer that flushes after each row for data safety.
    
    Features:
    - Immediate flush after each row
    - Configurable fieldnames
    - Context manager for clean resource handling
    - Extensible for different evaluation types
    """
    
    def __init__(self, file_path: str, fieldnames: List[str]):
        """
        Initialize streaming CSV writer.
        
        Args:
            file_path: Path to the CSV file
            fieldnames: List of column names
        """
        self.file_path = file_path
        self.fieldnames = fieldnames
        self.csv_file = None
        self.csv_writer = None
        
    def __enter__(self):
        """Enter context manager and open CSV file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
        # Open CSV file
        self.csv_file = open(self.file_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        
        # Write header and flush
        self.csv_writer.writeheader()
        self.csv_file.flush()
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close CSV file."""
        if self.csv_file:
            self.csv_file.close()
            
    def write_row(self, row_data: Dict[str, Any]) -> None:
        """
        Write a single row and flush immediately.
        
        Args:
            row_data: Dictionary with data for the row
        """
        if not self.csv_writer:
            raise RuntimeError("CSV writer not initialized. Use within context manager.")
            
        self.csv_writer.writerow(row_data)
        self.csv_file.flush()


@contextmanager
def evaluation_csv_writer(output_dir: str, run_name: str, subject: str):
    """
    Context manager for evaluation CSV writing with standard fieldnames.
    
    Args:
        output_dir: Output directory
        run_name: Unique run identifier
        subject: Subject being evaluated
        
    Yields:
        Function to write evaluation rows
    """
    csv_file_path = os.path.join(output_dir, f"detailed_results_{run_name}.csv")
    
    fieldnames = [
        'question_id', 'subject', 'question', 'choices', 'correct_answer',
        'predicted_choice', 'is_correct',
        'ttft', 'decode_time', 'total_time_ms', 'tokens_per_second',
        'input_tokens', 'output_tokens', 'generated_text_length'
    ]
    
    with StreamingCSVWriter(csv_file_path, fieldnames) as writer:
        def write_evaluation_row(question_id: int, question_data: Any, prediction: Any, 
                                correct_answer: str, predicted_choice: str, is_correct: bool, formatted_prompt: str = None):
            """Write a standardized evaluation row."""
            question_text = formatted_prompt if formatted_prompt is not None else getattr(question_data, 'question', getattr(question_data, 'prompt', ''))
            
            writer.write_row({
                'question_id': question_id,
                'subject': subject,
                'question': question_text,
                'choices': str(getattr(question_data, 'choices', '')),
                'correct_answer': correct_answer,
                'predicted_choice': predicted_choice,
                'is_correct': is_correct,
                'ttft': getattr(prediction, 'ttft', 0),
                'decode_time': getattr(prediction, 'decode_time', 0),
                'total_time_ms': prediction.total_time_ms,
                'tokens_per_second': prediction.tokens_per_second,
                'input_tokens': prediction.input_tokens,
                'output_tokens': prediction.output_tokens,
                'generated_text_length': len(prediction.generated_text)
            })
            
        yield write_evaluation_row


def get_detailed_csv_path(output_dir: str, run_name: str) -> str:
    """Get the path for detailed results CSV file."""
    return os.path.join(output_dir, f"detailed_results_{run_name}.csv")
