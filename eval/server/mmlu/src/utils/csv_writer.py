# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
        'predicted_choice', 'is_correct', 'generated_text',
        'ttft', 'decode_time', 'total_time_ms', 'tokens_per_second',
        'input_tokens', 'output_tokens', 'generated_text_length'
    ]
    
    with StreamingCSVWriter(csv_file_path, fieldnames) as writer:
        def write_evaluation_row(question_id: int, question_data: Any, prediction: Any, 
                                correct_answer: str, predicted_choice: str, is_correct: bool):
            """Write a standardized evaluation row."""
            writer.write_row({
                'question_id': question_id,
                'subject': subject,
                'question': question_data.question,
                'choices': str(question_data.choices),
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
