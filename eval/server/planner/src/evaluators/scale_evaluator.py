"""
Scale Evaluator for test-time scaling with multiple samples and majority voting.

This evaluator extends the base evaluator with scaling-specific features:
- Multiple sample generation per question
- Majority voting across samples
- Detailed sample tracking and analysis
- Enhanced performance monitoring for scaled inference
"""

import os
from typing import Dict, Any, List, Tuple
from datetime import datetime
from collections import Counter

from .base_evaluator import BaseEvaluator, EvaluationResult
from ..models import PredictionResult
from ..telemetry import monitor_evaluation
from ..utils.csv_writer import evaluation_csv_writer
from ..utils import AnswerExtractor


class ScaleEvaluator(BaseEvaluator):
    """
    Scale evaluator for test-time scaling with multiple samples.
    
    Features:
    - Multiple sample generation per question
    - Majority voting across samples
    - Detailed sample tracking and voting analysis
    - Enhanced performance metrics for scaling
    """
    
    def __init__(self, config_path: str = "configs/scale.yaml"):
        """Initialize scale evaluator with scaling configuration."""
        super().__init__(config_path)
        self.total_samples_generated = 0
        self.voting_confidence_scores = []
        self.answer_extractor = AnswerExtractor()
        
    def format_prompt(self, question_data: Dict[str, Any]) -> str:
        """
        Format prompt for scaling evaluation.
        
        Args:
            question_data: Question data from dataset
            
        Returns:
            Formatted prompt string for scaling
        """
        system_prompt = self.config.prompting['system_prompt']
        user_template = self.config.prompting['user_template']
        
        # Format with scaling template
        user_prompt = user_template.format(
            question=question_data.question,
            choice_a=question_data.choices[0],
            choice_b=question_data.choices[1],
            choice_c=question_data.choices[2],
            choice_d=question_data.choices[3]
        )
        
        return f"{system_prompt}\n\n{user_prompt}"
    
    def _majority_vote(self, answers: List[str]) -> Tuple[str, float, Dict[str, int]]:
        """
        Perform majority voting on extracted answers.
        
        Args:
            answers: List of extracted answer choices
            
        Returns:
            Tuple of (majority_choice, confidence, vote_counts)
        """
        if not answers:
            return "Invalid", 0.0, {}
        
        # Filter out Invalid responses for voting
        valid_answers = [ans for ans in answers if ans != "Invalid"]
        
        if not valid_answers:
            return "Invalid", 0.0, {"Invalid": len(answers)}
        
        vote_counts = Counter(valid_answers)
        total_valid = len(valid_answers)
        
        most_common = vote_counts.most_common(1)[0]
        majority_choice = most_common[0]
        majority_count = most_common[1]
        
        # Calculate confidence as percentage of valid votes
        confidence = majority_count / total_valid if total_valid > 0 else 0.0
        
        # Include invalid count in vote_counts for tracking
        all_vote_counts = dict(vote_counts)
        invalid_count = len(answers) - len(valid_answers)
        if invalid_count > 0:
            all_vote_counts["Invalid"] = invalid_count
        
        return majority_choice, confidence, all_vote_counts
    
    def _generate_multiple_samples(
        self, 
        prompt: str, 
        num_samples: int
    ) -> Tuple[str, List[str], List[str], float, int, Dict[str, int], int]:
        """
        Generate multiple samples and perform majority voting using vLLM's efficient 'n' parameter.
        
        Args:
            prompt: Input prompt
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (final_choice, all_samples, sample_predictions, total_time_ms, total_tokens, vote_counts, input_tokens)
        """
        token_budget = self.config.scaling.get('max_new_tokens', 128)
        
        # Single call with vLLM's 'n' parameter
        sampling_kwargs = {
            'max_tokens': token_budget,
            'min_tokens': token_budget,  
            'temperature': self.config.model['temperature'],
            'top_p': self.config.model['top_p'],
            'top_k': self.config.model.get('top_k', 50),
            'repetition_penalty': self.config.model.get('repetition_penalty', 1.1),
            'n': num_samples 
        }
        
        if 'seed' in self.config.model and self.config.model['seed'] is not None:
            sampling_kwargs['seed'] = self.config.model['seed']
        
        # Single call to generate all samples efficiently
        prediction = self.model.predict(
            prompt=prompt,
            **sampling_kwargs
        )
        
        # Log efficiency improvement
        avg_time_per_sample = prediction.total_time_ms / num_samples
        print(f"Efficient scaling: {num_samples} samples in {prediction.total_time_ms:.1f}ms (avg {avg_time_per_sample:.1f}ms/sample)")
        if prediction.generated_texts is None or len(prediction.generated_texts) != num_samples:
            raise RuntimeError(f"Expected {num_samples} generated texts, got {len(prediction.generated_texts) if prediction.generated_texts else 0}")
        
        all_samples = prediction.generated_texts
        sample_predictions = []
        
        for i, generated_text in enumerate(all_samples):
            predicted_choice = self.answer_extractor.extract_choice(generated_text)
            sample_predictions.append(predicted_choice)
        
        total_time_ms = prediction.total_time_ms
        total_tokens = prediction.output_tokens
        input_tokens = prediction.input_tokens
        
        # Perform majority voting
        final_choice, confidence, vote_counts = self._majority_vote(sample_predictions)
        self.voting_confidence_scores.append(confidence)
        
        return final_choice, all_samples, sample_predictions, total_time_ms, total_tokens, vote_counts, input_tokens
    
    def evaluate_subject(
        self,
        model_path: str,
        subject: str,
        output_dir: str = "./results"
    ) -> EvaluationResult:
        """
        Evaluate subject with test-time scaling.
        
        Args:
            model_path: Path to the model
            subject: MMLU subject to evaluate
            output_dir: Directory for output files
            
        Returns:
            EvaluationResult with scaling-specific metrics
        """
        # Reset scaling counters
        self.total_samples_generated = 0
        self.voting_confidence_scores = []
        
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
            
        num_samples = self.config.scaling['num_samples']
        print(f"Evaluating {len(questions)} questions with {num_samples} samples each (efficient vLLM 'n' parameter scaling)")
        
        # Setup telemetry monitoring
        run_name = f"{self.config.name}_{subject}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_name = os.path.basename(model_path)
        
        # Run evaluation with telemetry
        with monitor_evaluation(
            output_dir=output_dir,
            run_name=run_name,
            model_name=model_name,
            config_name=self.config.name,
            evaluation_type=f"mmlu_{subject}_scaling"
        ) as monitor:
            question_results = []
            correct_count = 0
            
            # Use streaming CSV writer for detailed results
            with evaluation_csv_writer(output_dir, run_name, subject) as write_csv_row:
                for i, question_data in enumerate(questions):
                    print(f"Processing question {i+1}/{len(questions)} (scaling with {num_samples} samples)")
                    
                    # Format prompt
                    prompt = self.format_prompt(question_data)
                    
                    # Generate multiple samples and get majority vote
                    final_choice, all_samples, sample_predictions, total_time_ms, total_tokens, vote_counts, input_tokens = \
                        self._generate_multiple_samples(prompt, num_samples)
                    
                    self.total_samples_generated += num_samples
                    
                    # Create aggregated prediction result for compatibility
                    avg_time_per_sample = total_time_ms / num_samples
                    tokens_per_second = total_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0
                    
                    estimated_ttft = total_time_ms * 0.1 
                    estimated_decode_time = total_time_ms * 0.9 
                    
                    # Create a synthetic prediction result for telemetry
                    prediction = PredictionResult(
                        predicted_choice=final_choice,
                        generated_text=f"Majority vote from {num_samples} samples: {final_choice}",
                        input_tokens=input_tokens,
                        output_tokens=total_tokens,
                        ttft=estimated_ttft,
                        decode_time=estimated_decode_time,
                        total_time_ms=total_time_ms,
                        tokens_per_second=tokens_per_second
                    )
                    
                    correct_answer = question_data.correct_answer
                    is_correct = final_choice == correct_answer
                    if is_correct:
                        correct_count += 1
                        
                    question_result = {
                        'question_id': i,
                        'question': question_data.question,
                        'choices': question_data.choices,
                        'correct_answer': correct_answer,
                        'predicted_choice': final_choice,
                        'is_correct': is_correct,
                        'generated_text': prediction.generated_text,
                        'input_tokens': prediction.input_tokens,
                        'output_tokens': prediction.output_tokens,
                        'time_ms': prediction.total_time_ms,
                        'tokens_per_second': prediction.tokens_per_second,
                        'ttft': prediction.ttft,
                        'decode_time': prediction.decode_time,
                        # Scaling-specific metrics
                        'num_samples': num_samples,
                        'all_samples': all_samples,
                        'sample_predictions': sample_predictions,
                        'vote_counts': vote_counts,
                        'voting_confidence': self.voting_confidence_scores[-1]
                    }
                    question_results.append(question_result)
                    write_csv_row(i, question_data, prediction, correct_answer, final_choice, is_correct)
                    monitor.record_question_result(i, prediction)
                    
        # Calculate final metrics
        accuracy = correct_count / len(questions) if questions else 0.0
        avg_time = sum(r['time_ms'] for r in question_results) / len(question_results)
        avg_tokens_per_sec = sum(r['tokens_per_second'] for r in question_results) / len(question_results)
        avg_voting_confidence = sum(self.voting_confidence_scores) / len(self.voting_confidence_scores) if self.voting_confidence_scores else 0.0
        
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
        
        result.scaling_metrics = {
            'total_samples_generated': self.total_samples_generated,
            'samples_per_question': num_samples,
            'avg_voting_confidence': avg_voting_confidence,
            'scaling_efficiency': self._calculate_scaling_efficiency(result)
        }
        
        if self.config.output.get('save_detailed_responses', True):
            self._save_detailed_results(result, output_dir, run_name)
            
        return result
    
    def _calculate_scaling_efficiency(self, result: EvaluationResult) -> float:
        """
        Calculate scaling efficiency based on accuracy improvement vs computational cost.
        
        Args:
            result: Evaluation result
            
        Returns:
            Scaling efficiency score (0.0 to 1.0)
        """
        avg_confidence = sum(self.voting_confidence_scores) / len(self.voting_confidence_scores) if self.voting_confidence_scores else 0.0
        efficiency = (result.accuracy * 0.7) + (avg_confidence * 0.3)
        
        return min(efficiency, 1.0)
    
    def print_summary(self, result: EvaluationResult) -> None:
        """Print scaling evaluation summary with scaling-specific metrics."""
        super().print_summary(result)
        
        if hasattr(result, 'scaling_metrics') and result.scaling_metrics:
            scaling_metrics = result.scaling_metrics
            print(f"\nTEST-TIME SCALING METRICS:")
            print(f"Total Samples Generated: {scaling_metrics['total_samples_generated']}")
            print(f"Samples per Question: {scaling_metrics['samples_per_question']}")
            print(f"Avg Voting Confidence: {scaling_metrics['avg_voting_confidence']:.4f}")
            print(f"Scaling Efficiency: {scaling_metrics['scaling_efficiency']:.4f}")
            print(f"{'='*60}")
