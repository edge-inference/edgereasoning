"""
Data Models for Results Processing

Defines data structures for handling evaluation results across models and subjects.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import pandas as pd


@dataclass
class QuestionResult:
    """Individual question result data."""
    question_id: int
    subject: str
    question: str
    choices: List[str]
    correct_answer: str
    predicted_choice: str
    is_correct: bool
    generated_text: str
    
    # Performance metrics
    ttft: float  # Time to first token (ms)
    decode_time: float  # Decode time (ms)
    total_time_ms: float  # Total time (ms)
    tokens_per_second: float  # Generation speed
    input_tokens: int  # Prompt tokens
    output_tokens: int  # Completion tokens
    generated_text_length: int
    
    @classmethod
    def from_csv_row(cls, row: Dict[str, Any]) -> 'QuestionResult':
        """Create QuestionResult from CSV row."""
        # Parse choices string back to list
        choices_str = row.get('choices', '[]')
        try:
            import ast
            choices = ast.literal_eval(choices_str)
        except:
            choices = []
            
        return cls(
            question_id=int(row.get('question_id', 0)),
            subject=str(row.get('subject', '')),
            question=str(row.get('question', '')),
            choices=choices,
            correct_answer=str(row.get('correct_answer', '')),
            predicted_choice=str(row.get('predicted_choice', '')),
            is_correct=bool(row.get('is_correct', False)),
            generated_text=str(row.get('generated_text', '')),
            ttft=float(row.get('ttft', 0.0)),
            decode_time=float(row.get('decode_time', 0.0)),
            total_time_ms=float(row.get('total_time_ms', 0.0)),
            tokens_per_second=float(row.get('tokens_per_second', 0.0)),
            input_tokens=int(row.get('input_tokens', 0)),
            output_tokens=int(row.get('output_tokens', 0)),
            generated_text_length=int(row.get('generated_text_length', 0))
        )


@dataclass
class SubjectResult:
    """Results for a single subject within a model."""
    model_name: str
    subject: str
    total_questions: int
    correct_answers: int
    accuracy: float
    
    # Performance metrics
    avg_ttft: float
    avg_decode_time: float
    avg_total_time: float
    avg_tokens_per_second: float
    total_input_tokens: int
    total_output_tokens: int
    
    # Individual questions
    questions: List[QuestionResult] = field(default_factory=list)
    
    # Metadata
    timestamp: Optional[datetime] = None
    config_name: Optional[str] = None
    
    @classmethod
    def from_questions(cls, model_name: str, subject: str, questions: List[QuestionResult]) -> 'SubjectResult':
        """Create SubjectResult from list of questions."""
        if not questions:
            return cls(
                model_name=model_name,
                subject=subject,
                total_questions=0,
                correct_answers=0,
                accuracy=0.0,
                avg_ttft=0.0,
                avg_decode_time=0.0,
                avg_total_time=0.0,
                avg_tokens_per_second=0.0,
                total_input_tokens=0,
                total_output_tokens=0,
                questions=[]
            )
        
        total_questions = len(questions)
        correct_answers = sum(1 for q in questions if q.is_correct)
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        
        # Calculate averages
        avg_ttft = sum(q.ttft for q in questions) / total_questions
        avg_decode_time = sum(q.decode_time for q in questions) / total_questions
        avg_total_time = sum(q.total_time_ms for q in questions) / total_questions
        avg_tokens_per_second = sum(q.tokens_per_second for q in questions) / total_questions
        total_input_tokens = sum(q.input_tokens for q in questions)
        total_output_tokens = sum(q.output_tokens for q in questions)
        
        return cls(
            model_name=model_name,
            subject=subject,
            total_questions=total_questions,
            correct_answers=correct_answers,
            accuracy=accuracy,
            avg_ttft=avg_ttft,
            avg_decode_time=avg_decode_time,
            avg_total_time=avg_total_time,
            avg_tokens_per_second=avg_tokens_per_second,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            questions=questions
        )


@dataclass
class ModelResult:
    """Complete results for a single model across all subjects."""
    model_name: str
    timestamp: datetime
    config_name: str
    
    # Overall metrics
    total_subjects: int
    successful_subjects: int
    overall_accuracy: float
    total_questions: int
    total_correct: int
    
    # Performance metrics
    avg_ttft: float
    avg_decode_time: float
    avg_total_time: float
    avg_tokens_per_second: float
    total_input_tokens: int
    total_output_tokens: int
    
    # Subject-level results
    subjects: Dict[str, SubjectResult] = field(default_factory=dict)
    
    @classmethod
    def from_subjects(cls, model_name: str, subjects: Dict[str, SubjectResult], 
                     timestamp: Optional[datetime] = None, config_name: str = "unknown") -> 'ModelResult':
        """Create ModelResult from subject results."""
        if not subjects:
            return cls(
                model_name=model_name,
                timestamp=timestamp or datetime.now(),
                config_name=config_name,
                total_subjects=0,
                successful_subjects=0,
                overall_accuracy=0.0,
                total_questions=0,
                total_correct=0,
                avg_ttft=0.0,
                avg_decode_time=0.0,
                avg_total_time=0.0,
                avg_tokens_per_second=0.0,
                total_input_tokens=0,
                total_output_tokens=0,
                subjects={}
            )
        
        total_subjects = len(subjects)
        successful_subjects = len([s for s in subjects.values() if s.total_questions > 0])
        total_questions = sum(s.total_questions for s in subjects.values())
        total_correct = sum(s.correct_answers for s in subjects.values())
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        # Weight averages by number of questions
        if total_questions > 0:
            avg_ttft = sum(s.avg_ttft * s.total_questions for s in subjects.values()) / total_questions
            avg_decode_time = sum(s.avg_decode_time * s.total_questions for s in subjects.values()) / total_questions
            avg_total_time = sum(s.avg_total_time * s.total_questions for s in subjects.values()) / total_questions
            avg_tokens_per_second = sum(s.avg_tokens_per_second * s.total_questions for s in subjects.values()) / total_questions
        else:
            avg_ttft = avg_decode_time = avg_total_time = avg_tokens_per_second = 0.0
            
        total_input_tokens = sum(s.total_input_tokens for s in subjects.values())
        total_output_tokens = sum(s.total_output_tokens for s in subjects.values())
        
        return cls(
            model_name=model_name,
            timestamp=timestamp or datetime.now(),
            config_name=config_name,
            total_subjects=total_subjects,
            successful_subjects=successful_subjects,
            overall_accuracy=overall_accuracy,
            total_questions=total_questions,
            total_correct=total_correct,
            avg_ttft=avg_ttft,
            avg_decode_time=avg_decode_time,
            avg_total_time=avg_total_time,
            avg_tokens_per_second=avg_tokens_per_second,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            subjects=subjects
        )


@dataclass
class ConsolidatedResult:
    """Consolidated results across multiple models."""
    models: Dict[str, ModelResult] = field(default_factory=dict)
    processing_timestamp: datetime = field(default_factory=datetime.now)
    
    def add_model(self, model_result: ModelResult) -> None:
        """Add a model result to the consolidated results."""
        self.models[model_result.model_name] = model_result
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Get a comparison dataframe across all models."""
        rows = []
        for _key, model_result in self.models.items():
            rows.append({
                'model_name': model_result.model_name,
                'overall_accuracy': model_result.overall_accuracy,
                'total_questions': model_result.total_questions,
                'total_correct': model_result.total_correct,
                'total_subjects': model_result.total_subjects,
                'successful_subjects': model_result.successful_subjects,
                'avg_ttft_ms': model_result.avg_ttft,
                'avg_decode_time_ms': model_result.avg_decode_time,
                'avg_total_time_ms': model_result.avg_total_time,
                'avg_tokens_per_second': model_result.avg_tokens_per_second,
                'total_input_tokens': model_result.total_input_tokens,
                'total_output_tokens': model_result.total_output_tokens,
                'config_name': model_result.config_name,
                'timestamp': model_result.timestamp
            })
        return pd.DataFrame(rows)
    
    def get_subject_comparison_dataframe(self) -> pd.DataFrame:
        """Get subject-level comparison across all models."""
        rows = []
        for _key, model_result in self.models.items():
            for subject_name, subject_result in model_result.subjects.items():
                rows.append({
                    'model_name': model_result.model_name,
                    'subject': subject_name,
                    'accuracy': subject_result.accuracy,
                    'total_questions': subject_result.total_questions,
                    'correct_answers': subject_result.correct_answers,
                    'avg_ttft_ms': subject_result.avg_ttft,
                    'avg_decode_time_ms': subject_result.avg_decode_time,
                    'avg_total_time_ms': subject_result.avg_total_time,
                    'avg_tokens_per_second': subject_result.avg_tokens_per_second,
                    'total_input_tokens': subject_result.total_input_tokens,
                    'total_output_tokens': subject_result.total_output_tokens
                })
        return pd.DataFrame(rows)
