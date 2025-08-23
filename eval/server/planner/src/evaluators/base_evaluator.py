"""
Base Evaluator for VLLM Performance Testing.

This module provides the core evaluation framework that integrates model inference,
telemetry monitoring, and result analysis in a clean, modular architecture.
"""

import os
import yaml
import json
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

from ..models import VLLMModel, VLLMConfig, PredictionResult
from ..data_loaders import MMLULoader
from ..utils.csv_writer import evaluation_csv_writer
from ..utils import AnswerExtractor
from ..telemetry import TelemetryMonitor, monitor_evaluation


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    name: str
    description: str
    model: Dict[str, Any]
    evaluation: Dict[str, Any]
    prompting: Dict[str, Any]
    output: Dict[str, Any]
    scaling: Optional[Dict[str, Any]] = None  
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'EvaluationConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)


@dataclass
class EvaluationResult:
    """Results from an evaluation run."""
    config_name: str
    model_name: str
    subject: str
    total_questions: int
    correct_answers: int
    accuracy: float
    avg_time_per_question: float
    avg_tokens_per_second: float
    total_energy_j: Optional[float] = None
    question_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional metrics fields for specific evaluators
    noreasoning_metrics: Optional[Dict[str, Any]] = None
    scaling_metrics: Optional[Dict[str, Any]] = None
    budget_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result_dict = {
            'config_name': self.config_name,
            'model_name': self.model_name,
            'subject': self.subject,
            'total_questions': self.total_questions,
            'correct_answers': self.correct_answers,
            'accuracy': self.accuracy,
            'avg_time_per_question': self.avg_time_per_question,
            'avg_tokens_per_second': self.avg_tokens_per_second,
            'total_energy_j': self.total_energy_j,
            'timestamp': datetime.now().isoformat(),
            'question_results': self.question_results
        }
        
        # Add specific metrics if they exist
        if self.noreasoning_metrics:
            result_dict['noreasoning_metrics'] = self.noreasoning_metrics
        if self.scaling_metrics:
            result_dict['scaling_metrics'] = self.scaling_metrics
        if self.budget_metrics:
            result_dict['budget_metrics'] = self.budget_metrics
            
        return result_dict


class BaseEvaluator:
    """
    Professional evaluation framework for VLLM models.
    
    Features:
    - Configuration-driven evaluation
    - Integrated telemetry monitoring
    - Comprehensive result tracking
    - Modular prompt templates
    - Detailed performance analysis
    """
    
    def __init__(self, config_path: str):
        """
        Initialize evaluator with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = EvaluationConfig.from_yaml(config_path)
        self.dataset_loader = MMLULoader()
        self.answer_extractor = AnswerExtractor()
        self.model: Optional[VLLMModel] = None
        
    def setup_model(self, model_path: str) -> None:
        """
        Setup VLLM model with configuration.
        
        Args:
            model_path: Path to the model
        """
        model_config = VLLMConfig(
            model_path=model_path,
            tensor_parallel_size=self.config.model.get('tensor_parallel_size', 1),
            gpu_memory_utilization=self.config.model.get('gpu_memory_utilization', 0.90),
            max_model_len=self.config.model.get('max_model_len', 4096)
        )
        
        print(f"Setting up model: {model_path}")
        self.model = VLLMModel(model_config)
        print(f"Model ready for evaluation")
        
    def format_prompt(self, question_data: Dict[str, Any]) -> str:
        """
        Format prompt using configuration template.
        
        Args:
            question_data: Question data from dataset
            
        Returns:
            Formatted prompt string
        """
        # Build the prompt using system and user templates
        system_prompt = self.config.prompting['system_prompt']
        user_template = self.config.prompting['user_template']
        
        # Format user prompt with question data
        user_prompt = user_template.format(
            question=question_data.question,
            choice_a=question_data.choices[0],
            choice_b=question_data.choices[1],
            choice_c=question_data.choices[2],
            choice_d=question_data.choices[3]
        )
        
        # Combine system and user prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        return full_prompt
        
    def evaluate_subject(
        self,
        model_path: str,
        subject: str,
        output_dir: str = "./results"
    ) -> EvaluationResult:
        """
        Evaluate model on a specific subject.
        
        Args:
            model_path: Path to the model
            subject: MMLU subject to evaluate
            output_dir: Directory for output files
            
        Returns:
            EvaluationResult with comprehensive metrics
        """
        if not self.model:
            self.setup_model(model_path)
            
        # Load dataset
        print(f"Loading subject: {subject}")
        questions = self.dataset_loader.load_subject(subject)
        
        if not questions:
            raise ValueError(f"No questions loaded for subject: {subject}")
            
        # Limit questions if specified in config
        num_questions = self.config.evaluation.get('num_questions')
        if num_questions and num_questions < len(questions):
            questions = questions[:num_questions]
            
        print(f"Evaluating {len(questions)} questions")
        
        # Setup telemetry monitoring
        run_name = f"{self.config.name}_{subject}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_name = os.path.basename(model_path)
        
        # Run evaluation with telemetry
        with monitor_evaluation(
            output_dir=output_dir,
            run_name=run_name,
            model_name=model_name,
            config_name=self.config.name,
            evaluation_type=f"mmlu_{subject}"
        ) as monitor:
            question_results = []
            correct_count = 0
            
            # Use streaming CSV writer for detailed results
            with evaluation_csv_writer(output_dir, run_name, subject) as write_csv_row:
                for i, question_data in enumerate(questions):
                    print(f"Processing question {i+1}/{len(questions)}")
                    
                    # Format prompt
                    prompt = self.format_prompt(question_data)
                    
                    # Get prediction
                    prediction = self.model.predict(
                        prompt=prompt,
                        max_tokens=self.config.model['max_tokens'],
                        temperature=self.config.model['temperature'],
                        top_p=self.config.model['top_p']
                    )
                    
                    # Extract answer
                    predicted_choice = self.answer_extractor.extract_choice(prediction.generated_text)
                    prediction.predicted_choice = predicted_choice
                    
                    # Check correctness
                    correct_answer = question_data.correct_answer
                    is_correct = predicted_choice == correct_answer
                    if is_correct:
                        correct_count += 1
                        
                    # Record detailed results
                    question_result = {
                        'question_id': i,
                        'question': question_data.question,
                        'choices': question_data.choices,
                        'correct_answer': correct_answer,
                        'predicted_choice': predicted_choice,
                        'is_correct': is_correct,
                        'generated_text': prediction.generated_text,
                        'input_tokens': prediction.input_tokens,
                        'output_tokens': prediction.output_tokens,
                        'time_ms': prediction.total_time_ms,
                        'tokens_per_second': prediction.tokens_per_second,
                        'ttft': prediction.ttft, 
                        'decode_time': prediction.decode_time  
                    }
                    question_results.append(question_result)
                    
                    # Write to CSV immediately using reusable module
                    write_csv_row(i, question_data, prediction, correct_answer, predicted_choice, is_correct)
                    
                    # Record in telemetry
                    monitor.record_question_result(i, prediction)
                
        # Calculate final metrics
        accuracy = correct_count / len(questions) if questions else 0.0
        avg_time = sum(r['time_ms'] for r in question_results) / len(question_results)
        avg_tokens_per_sec = sum(r['tokens_per_second'] for r in question_results) / len(question_results)
        
        # Create result object
        result = EvaluationResult(
            config_name=self.config.name,
            model_name=model_name,
            subject=subject,
            total_questions=len(questions),
            correct_answers=correct_count,
            accuracy=accuracy,
            avg_time_per_question=avg_time,
            avg_tokens_per_second=avg_tokens_per_sec,
            question_results=question_results
        )
        
        # Save detailed results if configured
        if self.config.output.get('save_detailed_responses', True):
            self._save_detailed_results(result, output_dir, run_name)
            
        return result
        
    def _save_detailed_results(self, result: EvaluationResult, output_dir: str, run_name: str) -> None:
        """Save detailed evaluation results summary in JSON format."""
        # Save JSON summary (CSV is already written during evaluation)
        results_file = os.path.join(output_dir, f"results_{run_name}.json")
        with open(results_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
            
        csv_file = os.path.join(output_dir, f"detailed_results_{run_name}.csv")
        print(f"Detailed results saved: {results_file}")
        print(f"Detailed CSV saved: {csv_file}")
        
    def print_summary(self, result: EvaluationResult) -> None:
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Config: {result.config_name}")
        print(f"Model: {result.model_name}")
        print(f"Subject: {result.subject}")
        print(f"Questions: {result.total_questions}")
        print(f"Correct: {result.correct_answers}")
        print(f"Accuracy: {result.accuracy:.4f} ({result.accuracy*100:.2f}%)")
        print(f"Avg Time/Question: {result.avg_time_per_question:.1f}ms")
        print(f"Avg Tokens/Second: {result.avg_tokens_per_second:.1f}")
        if result.total_energy_j:
            print(f"Total Energy: {result.total_energy_j:.2f}J")
        print(f"{'='*60}")