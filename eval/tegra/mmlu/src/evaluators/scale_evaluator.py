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
        
        # Ensure scaling config exists
        if not hasattr(self.config, 'scaling') or self.config.scaling is None:
            raise ValueError("Scale evaluator requires 'scaling' section in config")
        
        # CRITICAL: Ensure num_samples is always an integer from the start
        if 'num_samples' in self.config.scaling:
            try:
                self.config.scaling['num_samples'] = int(self.config.scaling['num_samples'])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid num_samples in config: {self.config.scaling['num_samples']} - {e}")
        
        # CRITICAL: Also ensure other numeric config values are integers (fix sed string conversion issue)
        numeric_scaling_keys = ['max_new_tokens', 'min_new_tokens']
        for key in numeric_scaling_keys:
            if key in self.config.scaling:
                try:
                    self.config.scaling[key] = int(self.config.scaling[key])
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert {key} to int: {self.config.scaling[key]} - {e}")
        
        # CRITICAL: Also ensure model config numeric values are integers
        numeric_model_keys = ['max_tokens', 'max_model_len', 'tensor_parallel_size', 'top_k']
        for key in numeric_model_keys:
            if key in self.config.model:
                try:
                    self.config.model[key] = int(self.config.model[key])
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert model.{key} to int: {self.config.model[key]} - {e}")
        
        # CRITICAL: Ensure float values are properly typed too
        numeric_model_float_keys = ['temperature', 'top_p', 'gpu_memory_utilization', 'repetition_penalty']
        for key in numeric_model_float_keys:
            if key in self.config.model:
                try:
                    self.config.model[key] = float(self.config.model[key])
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert model.{key} to float: {self.config.model[key]} - {e}")
        
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
            return "Invalid", 60, {}
        
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
        num_samples: Any  
    ) -> Tuple[str, List[str], List[str], float, int, Dict[str, int], List[Dict[str, Any]], int]:
        """
        Generate multiple samples and perform majority voting.
        
        Args:
            prompt: Input prompt
            num_samples: Number of samples to generate (will be converted to int)
            
        Returns:
            Tuple of (final_choice, all_samples, sample_predictions, total_time_ms, total_tokens, vote_counts, sample_details, input_tokens)
        """
        try:
            num_samples = int(num_samples)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert num_samples to int: {num_samples} - {e}")
        
        # Get token budget for consistent output lengths
        token_budget = self.config.scaling.get('max_new_tokens', 128)
        
        # CORRECTED: Use vLLM's native 'n' parameter for multiple sequences
        sampling_kwargs = {
            'max_tokens': token_budget,
            'min_tokens': token_budget,  # ENFORCE: min_tokens = max_tokens for consistent length
            'temperature': self.config.model['temperature'],
            'top_p': self.config.model['top_p'],
            'n': num_samples
        }
        
        # Add seed if specified in config
        if hasattr(self.config, 'scaling') and 'seed' in self.config.scaling and self.config.scaling['seed'] is not None:
            sampling_kwargs['seed'] = self.config.scaling['seed']
        
        # Single call to generate all samples
        prediction = self.model.predict(
            prompt=prompt,
            **sampling_kwargs
        )
        
        # Extract all samples and process them
        all_samples = []
        sample_predictions = []
        sample_details = []
        total_time_ms = prediction.total_time_ms
        total_tokens = prediction.output_tokens
        input_tokens = prediction.input_tokens
        
        # Process each generated sequence
        for i, generated_text in enumerate(prediction.generated_texts):
            # Extract choice from this sample using the utility
            predicted_choice = self.answer_extractor.extract_choice(generated_text)
            
            all_samples.append(generated_text)
            sample_predictions.append(predicted_choice)
            
            # Store detailed sample information
            sample_detail = {
                'sample_id': i + 1,
                'generated_text': generated_text,
                'predicted_choice': predicted_choice,
                'input_tokens': prediction.input_tokens,
                'output_tokens': prediction.output_tokens // num_samples,  # Approximate per sample
                'time_ms': prediction.total_time_ms,  # Total time for all samples
                'tokens_per_second': prediction.tokens_per_second,
                'ttft': prediction.ttft,
                'decode_time': prediction.decode_time,
                'last_token_time': prediction.last_token_time  
            }
            sample_details.append(sample_detail)
        
        # Perform majority voting
        final_choice, confidence, vote_counts = self._majority_vote(sample_predictions)
        self.voting_confidence_scores.append(confidence)
        
        return final_choice, all_samples, sample_predictions, total_time_ms, total_tokens, vote_counts, sample_details, input_tokens
    
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
            
        # Get question limit from config
        num_questions = self.config.evaluation.get('num_questions')
        max_questions = None
        if num_questions is not None:
            try:
                max_questions = int(num_questions)
                print(f"🔍 DEBUG: Will limit to {max_questions} questions per subject")
            except (ValueError, TypeError):
                print(f"Warning: Invalid num_questions value: {num_questions}, ignoring limit")
        
        # Load dataset with question limit applied at loader level
        print(f"🔍 DEBUG: Loading subject: {subject} with max_questions={max_questions}")
        questions = self.dataset_loader.load_subject(subject, max_questions=max_questions)
        
        if not questions:
            raise ValueError(f"No questions loaded for subject: {subject}")
            
        print(f"🔍 DEBUG: Final loaded questions: {len(questions)} for subject {subject}")
            
        num_samples = self.config.scaling['num_samples']
        try:
            num_samples = int(num_samples)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert num_samples to int: {num_samples} - {e}")
        
        print(f"Evaluating {len(questions)} questions with {num_samples} samples each (scaling mode)")
        
        # Get token budget for consistent output lengths
        token_budget = self.config.scaling.get('max_new_tokens', 128)
        print(f"🔍 DEBUG: Token budget enforced: min_tokens = max_tokens = {token_budget}")
        
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
                    final_choice, all_samples, sample_predictions, total_time_ms, total_tokens, vote_counts, sample_details, input_tokens = \
                        self._generate_multiple_samples(prompt, num_samples)
                    
                    self.total_samples_generated += num_samples
                    
                    # Create aggregated prediction result for compatibility
                    avg_time_per_sample = total_time_ms / num_samples
                    avg_tokens_per_sample = total_tokens / num_samples
                    tokens_per_second = total_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0
                    
                    # Estimate timing components (since vLLM doesn't provide per-sequence timing for n > 1)
                    estimated_ttft = total_time_ms * 0.05  
                    estimated_decode_time = total_time_ms * 0.95 
                    
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
                    
                    # Check correctness
                    correct_answer = question_data.correct_answer
                    is_correct = final_choice == correct_answer
                    if is_correct:
                        correct_count += 1
                        
                    # Record detailed results
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
                        'last_token_time': prediction.last_token_time,
                        # Scaling-specific metrics
                        'num_samples': num_samples,
                        'all_samples': all_samples,
                        'sample_predictions': sample_predictions,
                        'vote_counts': vote_counts,
                        'voting_confidence': self.voting_confidence_scores[-1],
                        # NEW: Include detailed sample information
                        'sample_details': sample_details,
                        'scaling_summary': {
                            'total_samples_generated': num_samples,
                            'majority_vote': final_choice,
                            'voting_confidence': self.voting_confidence_scores[-1],
                            'vote_distribution': vote_counts,
                            'token_budget_enforced': token_budget,
                            'avg_time_per_sample': total_time_ms / num_samples,
                            'avg_tokens_per_sample': total_tokens / num_samples
                        }
                    }
                    question_results.append(question_result)
                    
                    # Write to CSV immediately using reusable module
                    write_csv_row(i, question_data, prediction, correct_answer, final_choice, is_correct)
                    
                    # Record in telemetry
                    monitor.record_question_result(i, prediction)
                    
        # Calculate final metrics
        accuracy = correct_count / len(questions) if questions else 0.0
        avg_time = sum(r['time_ms'] for r in question_results) / len(question_results)
        avg_tokens_per_sec = sum(r['tokens_per_second'] for r in question_results) / len(question_results)
        avg_voting_confidence = sum(self.voting_confidence_scores) / len(self.voting_confidence_scores) if self.voting_confidence_scores else 0.0
        
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
        
        # Add scaling-specific metrics
        result.question_results.append({
            'scaling_metrics': {
                'total_samples_generated': self.total_samples_generated,
                'samples_per_question': num_samples,
                'avg_voting_confidence': avg_voting_confidence,
                'scaling_efficiency': self._calculate_scaling_efficiency(result)
            }
        })
        
        # Save detailed results if configured
        if self.config.output.get('save_detailed_responses', True):
            self._save_detailed_results(result, output_dir, run_name)
            
        return result
    
    def evaluate_subject_on_questions(
        self,
        model_path: str,
        questions: List[Any],
        output_dir: str = "./results",
        run_name: str = None,
        min_tokens: int = None
    ) -> EvaluationResult:
        """
        Evaluate model on a specific list of questions with test-time scaling.
        
        Args:
            model_path: Path to the model
            questions: List of questions to evaluate
            output_dir: Directory for output files
            run_name: Optional run name override
            min_tokens: Optional min tokens override (for compatibility)
            
        Returns:
            EvaluationResult with scaling-specific metrics
        """
        # Reset scaling counters
        self.total_samples_generated = 0
        self.voting_confidence_scores = []
        
        if not self.model:
            self.setup_model(model_path)
            
        if not questions:
            raise ValueError("No questions provided for evaluation")
            
        # CRITICAL: Ensure num_samples is always an integer
        num_samples = self.config.scaling['num_samples']
        try:
            num_samples = int(num_samples)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert num_samples to int: {num_samples} - {e}")
        
        # Get token budget for consistent output lengths
        token_budget = self.config.scaling.get('max_new_tokens', 128)
        print(f"🔍 DEBUG: Token budget enforced: min_tokens = max_tokens = {token_budget}")
        
        print(f"Evaluating {len(questions)} questions with {num_samples} samples each (scaling mode)")
        
        # Setup telemetry monitoring
        if not run_name:
            run_name = f"{self.config.name}_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_name = os.path.basename(model_path)
        
        # Determine subject from first question if available
        subject = getattr(questions[0], 'subject', 'custom') if questions else 'custom'
        
        # Run evaluation with telemetry
        with monitor_evaluation(
            output_dir=output_dir,
            run_name=run_name,
            model_name=model_name,
            config_name=self.config.name,
            evaluation_type=f"scale_{subject}"
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
                    final_choice, all_samples, sample_predictions, total_time_ms, total_tokens, vote_counts, sample_details, input_tokens = \
                        self._generate_multiple_samples(prompt, num_samples)
                    
                    self.total_samples_generated += num_samples
                    
                    # Create aggregated prediction result for compatibility
                    avg_time_per_sample = total_time_ms / num_samples
                    avg_tokens_per_sample = total_tokens / num_samples
                    tokens_per_second = total_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0
                    
                    # Estimate timing components (since vLLM doesn't provide per-sequence timing for n > 1)
                    estimated_ttft = total_time_ms * 0.05 
                    estimated_decode_time = total_time_ms * 0.95  
                    
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
                    
                    # Check correctness
                    correct_answer = question_data.correct_answer
                    is_correct = final_choice == correct_answer
                    if is_correct:
                        correct_count += 1
                        
                    # Record detailed results with ALL scaling information
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
                        'last_token_time': prediction.last_token_time,
                        # Scaling-specific metrics
                        'num_samples': num_samples,
                        'all_samples': all_samples,
                        'sample_predictions': sample_predictions,
                        'vote_counts': vote_counts,
                        'voting_confidence': self.voting_confidence_scores[-1],
                        # NEW: Include detailed sample information
                        'sample_details': sample_details,
                        'scaling_summary': {
                            'total_samples_generated': num_samples,
                            'majority_vote': final_choice,
                            'voting_confidence': self.voting_confidence_scores[-1],
                            'vote_distribution': vote_counts,
                            'token_budget_enforced': token_budget,
                            'avg_time_per_sample': total_time_ms / num_samples,
                            'avg_tokens_per_sample': total_tokens / num_samples
                        }
                    }
                    question_results.append(question_result)
                    
                    # Write to CSV immediately using reusable module
                    write_csv_row(i, question_data, prediction, correct_answer, final_choice, is_correct)
                    
                    # Record in telemetry
                    monitor.record_question_result(i, prediction)
                    
        # Calculate final metrics
        accuracy = correct_count / len(questions) if questions else 0.0
        avg_time = sum(r['time_ms'] for r in question_results) / len(question_results)
        avg_tokens_per_sec = sum(r['tokens_per_second'] for r in question_results) / len(question_results)
        avg_voting_confidence = sum(self.voting_confidence_scores) / len(self.voting_confidence_scores) if self.voting_confidence_scores else 0.0
        
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
        
        # Add scaling-specific metrics
        result.question_results.append({
            'scaling_metrics': {
                'total_samples_generated': self.total_samples_generated,
                'samples_per_question': num_samples,
                'avg_voting_confidence': avg_voting_confidence,
                'scaling_efficiency': self._calculate_scaling_efficiency(result)
            }
        })
        
        # Save detailed results if configured
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
        # Simple efficiency metric: accuracy weighted by confidence
        avg_confidence = sum(self.voting_confidence_scores) / len(self.voting_confidence_scores) if self.voting_confidence_scores else 0.0
        
        # Efficiency combines accuracy and confidence
        efficiency = (result.accuracy * 0.7) + (avg_confidence * 0.3)
        
        return min(efficiency, 1.0)
    
    def print_summary(self, result: EvaluationResult) -> None:
        """Print scaling evaluation summary with scaling-specific metrics."""
        super().print_summary(result)
        
        # Find scaling metrics
        scaling_metrics = None
        for item in result.question_results:
            if 'scaling_metrics' in item:
                scaling_metrics = item['scaling_metrics']
                break
        
        if scaling_metrics:
            print(f"\nTEST-TIME SCALING METRICS:")
            print(f"Total Samples Generated: {scaling_metrics['total_samples_generated']}")
            print(f"Samples per Question: {scaling_metrics['samples_per_question']}")
            print(f"Avg Voting Confidence: {scaling_metrics['avg_voting_confidence']:.4f}")
            print(f"Scaling Efficiency: {scaling_metrics['scaling_efficiency']:.4f}")
            print(f"{'='*60}")