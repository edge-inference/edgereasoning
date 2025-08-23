"""
No-Reasoning Evaluator for direct answer evaluation.

This evaluator extends the base evaluator with no-reasoning specific features:
- Direct answer extraction without reasoning steps
- Optimized for speed and directness
- Enhanced confidence analysis for quick decisions
- Minimal prompt overhead
"""

import re
from typing import Dict, Any, List, Tuple
from datetime import datetime

from .base_evaluator import BaseEvaluator, EvaluationResult
from ..models import PredictionResult
from ..telemetry import monitor_evaluation
from ..utils.csv_writer import evaluation_csv_writer


class NoReasoningEvaluator(BaseEvaluator):
    """
    No-reasoning evaluator for direct answer evaluation.
    
    Features:
    - Direct answer prompts without reasoning requests
    - Fast response extraction
    - Confidence analysis for quick decisions
    - Streamlined evaluation process
    """
    
    def __init__(self, config_path: str = "configs/noreasoning.yaml"):
        """Initialize no-reasoning evaluator with direct configuration."""
        super().__init__(config_path)
        self.quick_responses = 0
        self.confidence_scores = []
        
    def format_prompt(self, question_data: Dict[str, Any]) -> str:
        """
        Format prompt for direct answer without reasoning.
        Uses conversation structure with pre-filled assistant thinking to prime direct answers.
        
        Args:
            question_data: Question data from dataset
            
        Returns:
            Conversation-formatted prompt with pre-filled assistant thinking
        """
        system_prompt = self.config.prompting['system_prompt']
        user_template = self.config.prompting['user_template']
        
        # Format user question
        user_prompt = user_template.format(
            question=question_data.question,
            choice_a=question_data.choices[0],
            choice_b=question_data.choices[1],
            choice_c=question_data.choices[2],
            choice_d=question_data.choices[3]
        )
        
        conversation = (
            f"{system_prompt}\n\n"
            f"User: {user_prompt}\n\n"
            f"Assistant: <think> Okay I have finished thinking.</think>\n"
            f"The answer is "
        )
        
        return conversation
    
    def _extract_direct_answer(self, response_text: str) -> Tuple[str, float]:
        """
        Extract answer with confidence analysis for direct responses.
        Uses the robust AnswerExtractor for reliability.
        
        Args:
            response_text: Generated response text
            
        Returns:
            Tuple of (answer_choice, confidence_score)
        """
        if not response_text:
            return "Invalid", 0.0
            
        try:
            # Use the robust answer extractor from base class
            predicted_choice, confidence_level = self.answer_extractor.get_extraction_confidence(response_text)
            
            # Convert confidence level to numeric score
            confidence_map = {
                "high": 0.9,
                "medium": 0.7, 
                "low": 0.4,
                "invalid": 0.0
            }
            base_confidence = confidence_map.get(confidence_level, 0.0)
            
            # Add directness bonus for no-reasoning evaluation
            response_length = len(response_text.split())
            directness_bonus = max(0, (30 - response_length) / 30) * 0.2
            
            final_confidence = min(base_confidence + directness_bonus, 1.0)
            
            return predicted_choice, final_confidence
            
        except Exception as e:
            print(f"Warning: Answer extraction failed: {e}")
            return "Invalid", 0.0
    
    def _is_quick_response(self, response_text: str, time_ms: float) -> bool:
        """
        Determine if response was generated quickly and directly.
        
        Args:
            response_text: Generated text
            time_ms: Generation time in milliseconds
            
        Returns:
            True if response was quick and direct
        """
        # Quick if under 2 seconds and under 20 words
        word_count = len(response_text.split())
        return time_ms < 2000 and word_count < 20
    
    def evaluate_subject(
        self,
        model_path: str,
        subject: str,
        output_dir: str = "./results"
    ) -> EvaluationResult:
        """
        Evaluate subject with no-reasoning optimizations.
        
        Args:
            model_path: Path to the model
            subject: MMLU subject to evaluate
            output_dir: Directory for output files
            
        Returns:
            EvaluationResult with no-reasoning specific metrics
        """
        # Reset no-reasoning counters
        self.quick_responses = 0
        self.confidence_scores = []
        
        if not self.model:
            self.setup_model(model_path)
            
        # Load dataset
        print(f"Loading subject: {subject}")
        questions = self.dataset_loader.load_subject(subject)
        
        if not questions:
            raise ValueError(f"No questions loaded for subject: {subject}")
            
        # Limit questions if specified
        num_questions = self.config.evaluation.get('num_questions')
        if num_questions and num_questions < len(questions):
            questions = questions[:num_questions]
            
        print(f"Evaluating {len(questions)} questions (no-reasoning mode)")
        
        # Setup telemetry monitoring
        run_name = f"{self.config.name}_{subject}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_name = model_path.split('/')[-1]
        
        with monitor_evaluation(
            output_dir=output_dir,
            run_name=run_name,
            model_name=model_name,
            config_name=self.config.name,
            evaluation_type=f"mmlu_{subject}_noreasoning"
        ) as monitor:
            question_results = []
            correct_count = 0
            
            # Use streaming CSV writer for detailed results
            with evaluation_csv_writer(output_dir, run_name, subject) as write_csv_row:
                for i, question_data in enumerate(questions):
                    print(f"Processing question {i+1}/{len(questions)} (direct)")
                    
                    # Format prompt
                    prompt = self.format_prompt(question_data)
                    
                    # Get prediction
                    prediction = self.model.predict(
                        prompt=prompt,
                        max_tokens=self.config.model['max_tokens'],
                        temperature=self.config.model['temperature'],
                        top_p=self.config.model['top_p']
                    )
                    
                    # Extract answer with confidence
                    predicted_choice, confidence = self._extract_direct_answer(prediction.generated_text)
                    prediction.predicted_choice = predicted_choice
                    
                    # Check if quick response
                    is_quick = self._is_quick_response(prediction.generated_text, prediction.total_time_ms)
                    if is_quick:
                        self.quick_responses += 1
                    self.confidence_scores.append(confidence)
                    
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
                        'confidence_score': confidence,
                        'is_quick_response': is_quick,
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
        
        # Calculate metrics
        accuracy = correct_count / len(questions) if questions else 0.0
        avg_time = sum(r['time_ms'] for r in question_results) / len(question_results)
        avg_tokens_per_sec = sum(r['tokens_per_second'] for r in question_results) / len(question_results)
        avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0.0
        
        # Create result with no-reasoning metrics
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
        
        # Store no-reasoning specific metrics in result object (not in question_results list)
        result.noreasoning_metrics = {
            'quick_responses': self.quick_responses,
            'quick_response_rate': self.quick_responses / len(questions),
            'avg_confidence': avg_confidence,
            'directness_score': self._calculate_directness_score(result)
        }
        
        # Save results if configured
        if self.config.output.get('save_detailed_responses', True):
            self._save_detailed_results(result, output_dir, run_name)
            
        return result
    
    def _calculate_directness_score(self, result: EvaluationResult) -> float:
        """
        Calculate directness score based on response speed and brevity.
        
        Args:
            result: Evaluation result
            
        Returns:
            Directness score (0.0 to 1.0)
        """
        # Combine speed and brevity metrics
        speed_component = min(100.0 / result.avg_time_per_question, 1.0) * 1000  # Normalize to 0-1
        
        # Calculate average response length
        total_words = sum(len(r.get('generated_text', '').split()) for r in result.question_results[:-1])
        avg_words = total_words / (len(result.question_results) - 1) if len(result.question_results) > 1 else 0
        brevity_component = max(0, (30 - avg_words) / 30)  # Shorter is better
        
        directness_score = (speed_component + brevity_component) / 2
        return min(directness_score, 1.0)
    
    def print_summary(self, result: EvaluationResult) -> None:
        """Print no-reasoning evaluation summary with directness metrics."""
        super().print_summary(result)
        
        # Get no-reasoning metrics from result object
        if hasattr(result, 'noreasoning_metrics') and result.noreasoning_metrics:
            noreasoning_metrics = result.noreasoning_metrics
            print(f"\nNO-REASONING METRICS:")
            print(f"Quick Responses: {noreasoning_metrics['quick_responses']}/{result.total_questions}")
            print(f"Quick Response Rate: {noreasoning_metrics['quick_response_rate']:.4f}")
            print(f"Avg Confidence: {noreasoning_metrics['avg_confidence']:.4f}")
            print(f"Directness Score: {noreasoning_metrics['directness_score']:.4f}")
            print(f"{'='*60}")
