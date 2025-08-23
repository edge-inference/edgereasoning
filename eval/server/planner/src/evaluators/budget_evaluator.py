"""
Budget Evaluator for efficient, resource-constrained evaluation.

This evaluator extends the base evaluator with budget-specific optimizations:
- Aggressive prompt compression
- Early termination for obvious answers
- Minimal token usage
- Enhanced performance monitoring for low-resource scenarios
"""

import os
import re
from typing import Dict, Any, Optional
from datetime import datetime

from .base_evaluator import BaseEvaluator, EvaluationResult
from ..models import PredictionResult
from ..telemetry import monitor_evaluation
from ..utils.csv_writer import evaluation_csv_writer


class BudgetEvaluator(BaseEvaluator):
    """
    Budget-optimized evaluator for resource-constrained scenarios.
    
    Features:
    - Compressed prompt templates
    - Early termination detection
    - Token usage optimization
    - Enhanced efficiency metrics
    """
    
    def __init__(self, config_path: str = "configs/budget.yaml"):
        """Initialize budget evaluator with budget configuration."""
        super().__init__(config_path)
        self.early_termination_count = 0
        self.token_savings = 0
        
    def format_prompt(self, question_data: Dict[str, Any]) -> str:
        """
        Format prompt with budget-optimized compression.
        
        Args:
            question_data: Question data from dataset
            
        Returns:
            Compressed prompt string
        """
        # Use compressed template for budget mode
        system_prompt = self.config.prompting['system_prompt']
        user_template = self.config.prompting['user_template']
        
        # Compress question text if too long
        question = self._compress_text(question_data.question)
        
        # Format with compressed content
        user_prompt = user_template.format(
            question=question,
            choice_a=self._compress_text(question_data.choices[0]),
            choice_b=self._compress_text(question_data.choices[1]),
            choice_c=self._compress_text(question_data.choices[2]),
            choice_d=self._compress_text(question_data.choices[3])
        )
        
        return f"{system_prompt}\n\n{user_prompt}"
    
    def _compress_text(self, text: str, max_words: int = 50) -> str:
        """
        Compress text to reduce token usage.
        
        Args:
            text: Input text to compress
            max_words: Maximum number of words to keep
            
        Returns:
            Compressed text
        """
        words = text.split()
        if len(words) <= max_words:
            return text
            
        # Keep first part and add ellipsis
        compressed = ' '.join(words[:max_words])
        return f"{compressed}..."
    
    def _detect_early_answer(self, partial_response: str) -> Optional[str]:
        """
        Detect if model has provided clear answer early.
        
        Args:
            partial_response: Partial generated text
            
        Returns:
            Detected answer choice or None
        """
        # Look for clear answer patterns
        patterns = [
            r'(?:The )?answer is ([ABCD])',
            r'(?:The )?correct answer is ([ABCD])',
            r'Answer: ([ABCD])',
            r'([ABCD])\)',
            r'\(([ABCD])\)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, partial_response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
                
        return None
    
    def evaluate_subject(
        self,
        model_path: str,
        subject: str,
        output_dir: str = "./results"
    ) -> EvaluationResult:
        """
        Evaluate subject with budget optimizations.
        
        Args:
            model_path: Path to the model
            subject: MMLU subject to evaluate
            output_dir: Directory for output files
            
        Returns:
            EvaluationResult with budget-specific metrics
        """
        # Reset budget-specific counters
        self.early_termination_count = 0
        self.token_savings = 0
        
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
            
        print(f"Evaluating {len(questions)} questions (budget mode)")
        
        # Setup telemetry monitoring
        run_name = f"{self.config.name}_{subject}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_name = os.path.basename(model_path)
        
        # Run evaluation with telemetry
        with monitor_evaluation(
            output_dir=output_dir,
            run_name=run_name,
            model_name=model_name,
            config_name=self.config.name,
            evaluation_type=f"mmlu_{subject}_budget"
        ) as monitor:
            question_results = []
            correct_count = 0
            
            # Use streaming CSV writer for detailed results
            with evaluation_csv_writer(output_dir, run_name, subject) as write_csv_row:
                for i, question_data in enumerate(questions):
                    print(f"Processing question {i+1}/{len(questions)} (budget)")
                    
                    # Format prompt with budget compression
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

        # Add budget-specific metrics
        result.question_results.append({
            'budget_metrics': {
                'early_terminations': self.early_termination_count,
                'token_savings': self.token_savings,
                'efficiency_score': self._calculate_efficiency_score(result)
            }
        })
        
        # Save detailed results if configured
        if self.config.output.get('save_detailed_responses', True):
            self._save_detailed_results(result, output_dir, run_name)
            
        return result
    
    def _calculate_efficiency_score(self, result: EvaluationResult) -> float:
        """
        Calculate efficiency score based on accuracy and resource usage.
        
        Args:
            result: Evaluation result
            
        Returns:
            Efficiency score (0.0 to 1.0)
        """
        # Balance accuracy and efficiency
        accuracy_weight = 0.7
        efficiency_weight = 0.3
        
        # Normalize tokens per second (higher is better)
        # Assume baseline of 50 tokens/sec, scale to 0-1
        efficiency_component = min(result.avg_tokens_per_second / 100.0, 1.0)
        
        efficiency_score = (
            accuracy_weight * result.accuracy +
            efficiency_weight * efficiency_component
        )
        
        return efficiency_score
    
    def print_summary(self, result: EvaluationResult) -> None:
        """Print budget evaluation summary with efficiency metrics."""
        super().print_summary(result)
        
        # Find budget metrics
        budget_metrics = None
        for item in result.question_results:
            if 'budget_metrics' in item:
                budget_metrics = item['budget_metrics']
                break
        
        if budget_metrics:
            print(f"\nBUDGET OPTIMIZATION METRICS:")
            print(f"Early Terminations: {budget_metrics['early_terminations']}")
            print(f"Token Savings: {budget_metrics['token_savings']}")
            print(f"Efficiency Score: {budget_metrics['efficiency_score']:.4f}")
            print(f"{'='*60}")
