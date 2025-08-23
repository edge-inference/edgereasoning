"""
Natural Plan Scaling Evaluator for test-time scaling with multiple samples.

This evaluator extends the Natural Plan evaluator with scaling-specific features:
- Multiple sample generation per question using vLLM's 'n' parameter
- Majority voting across samples for Natural Plan tasks
- Detailed sample tracking and analysis
- Enhanced performance monitoring for scaled inference
"""

import os
from typing import Dict, Any, List, Tuple
from datetime import datetime
from collections import Counter

from .plan_evaluator_base import PlanEvaluatorBase
from .base_evaluator import EvaluationResult


class PlanEvaluatorScaling(PlanEvaluatorBase):
    """
    Natural Plan scaling evaluator for test-time scaling with multiple samples.
    
    Features:
    - Multiple sample generation per question
    - Majority voting across full-text responses  
    - Detailed sample tracking and voting analysis
    - Enhanced performance metrics for scaling
    - Consistent token enforcement for fair sample comparison
    """
    
    def __init__(self, config_path: str, task: str):
        """Initialize scaling evaluator with scaling configuration."""
        super().__init__(config_path, task)
        self.total_samples_generated = 0
        self.voting_confidence_scores = []
        
    def evaluate_task(
        self,
        model_path: str,
        output_dir: str = "./results",
        compute_metrics: bool = None,
    ):
        """
        Run the full task evaluation with scaling.
        
        This method overrides the base to ensure scaling parameters are properly handled.
        """
        # Ensure scaling config exists
        if not hasattr(self.config, 'scaling'):
            raise ValueError("Scaling configuration required but not found in config")
            
        scaling_config = getattr(self.config, "scaling", {})
        num_samples = scaling_config.get("num_samples", 1)
        
        if num_samples <= 1:
            raise ValueError(f"Scaling evaluator requires num_samples > 1, got {num_samples}")
            
        print(f"🔄 Natural Plan Scaling Evaluation: {num_samples} samples per question")
        
        # Reset scaling counters
        self.total_samples_generated = 0
        self.voting_confidence_scores = []
        
        # Override the evaluation loop to use scaling
        result = self._evaluate_with_scaling(
            model_path=model_path,
            output_dir=output_dir,
            compute_metrics=compute_metrics
        )
        
        return result
    
    def _generate_with_scaling(self, prompt: str, scaling_config: Dict[str, Any]):
        """
        Generate multiple samples and perform majority voting for Natural-Plan tasks.
        
        Uses vLLM's efficient 'n' parameter for parallel generation with consistent
        token lengths for fair comparison.
        """
        num_samples = scaling_config.get("num_samples", 8)
        
        token_budget = self.config.model["max_tokens"]
        
        sampling_kwargs = {
            'max_tokens': token_budget,
            'min_tokens': token_budget,  
            'temperature': self.config.model['temperature'],
            'top_p': self.config.model['top_p'],
            'top_k': self.config.model.get('top_k', 50),
            'repetition_penalty': self.config.model.get('repetition_penalty', 1.1),
            'stop': self.config.model.get("stop_sequences", ["<|im_end|>", "<|endoftext|>", "</s>"]),
            'n': num_samples 
        }
        
        if 'seed' in self.config.model and self.config.model['seed'] is not None:
            sampling_kwargs['seed'] = self.config.model['seed']
        prediction = self.model.predict(
            prompt=prompt,
            **sampling_kwargs
        )
        
        avg_time_per_sample = prediction.total_time_ms / num_samples
        print(f"Efficient scaling: {num_samples} samples in {prediction.total_time_ms:.1f}ms (avg {avg_time_per_sample:.1f}ms/sample)")
        
        if prediction.generated_texts is None or len(prediction.generated_texts) != num_samples:
            raise RuntimeError(f"Expected {num_samples} generated texts, got {len(prediction.generated_texts) if prediction.generated_texts else 0}")
        
        all_samples = prediction.generated_texts        
        self.total_samples_generated += num_samples
        
        # For Natural-Plan, we use the most common response as consensus
        response_counts = Counter(all_samples)
        majority_response, majority_count = response_counts.most_common(1)[0]
        
        # Calculate voting confidence
        voting_confidence = majority_count / len(all_samples) if all_samples else 0.0
        self.voting_confidence_scores.append(voting_confidence)
        
        prediction.generated_text = majority_response
        prediction.scaling_details = {
            'all_samples': all_samples,
            'vote_counts': dict(response_counts),
            'voting_confidence': voting_confidence,
            'num_samples': num_samples,
            'avg_time_per_sample': avg_time_per_sample
        }
        
        print(f"🔄 Scaling: {num_samples} samples → majority vote (confidence: {voting_confidence:.2f})")
        
        return prediction
    
    def _evaluate_with_scaling(self, model_path: str, output_dir: str = "./results", compute_metrics: bool = None):
        """Run evaluation with scaling-specific logic."""
        from ..utils.csv_writer import evaluation_csv_writer
        from ..telemetry import monitor_evaluation
        
        # Get scaling config
        scaling_config = getattr(self.config, "scaling", {})
        
        # Load model
        self.model = self.load_model(model_path)
        model_name = os.path.basename(model_path)
        
        # Load examples with filtering
        start_idx = self.config.evaluation.get("start_question", None)
        end_idx = self.config.evaluation.get("end_question", None)
        max_count = self.config.evaluation.get("num_questions", None)
        
        examples = self.loader.load(self.task, start_idx, end_idx, max_count)
        question_offset = start_idx if start_idx is not None else 0
        
        if compute_metrics is None:
            compute_metrics = self.config.evaluation.get("compute_metrics", True)
        
        print(f"[NaturalPlanScalingEvaluator] Processing {len(examples)} examples for task '{self.task}' with {scaling_config.get('num_samples', 8)} samples each")
        
        # Create results file
        run_name = f"{self.task}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        csv_file = os.path.join(output_dir, f"{run_name}.csv")
        
        question_results = []
        correct_count = 0
        
        with evaluation_csv_writer(csv_file, self.task) as write_csv_row:
            with monitor_evaluation(self.task, len(examples)) as monitor:
                for i, ex in enumerate(examples):
                    actual_question_idx = question_offset + i
                        
                    print(f"[{i+1}/{len(examples)}] Processing question {ex.example_id} (index {actual_question_idx}) with scaling")
                    prompt = self.format_prompt(ex)
                    
                    prediction = self._generate_with_scaling(prompt, scaling_config)

                    generated = prediction.generated_text.strip()
                    is_correct = False

                    if compute_metrics:
                        try:
                            if self.is_correct(ex, prediction):
                                is_correct = True
                                correct_count += 1
                        except Exception as e:
                            print(f"[ERROR] Validation failed for question {ex.example_id}: {e}")
                            is_correct = False 

                    question_result = {
                        "question_id": ex.example_id,
                        "prompt_tokens": prediction.input_tokens,
                        "output_tokens": prediction.output_tokens,
                        "generated_text": generated,
                        "is_correct": is_correct,
                        "time_ms": prediction.total_time_ms,
                        "tokens_per_second": prediction.tokens_per_second,
                        "ttft": prediction.ttft,
                        "decode_time": prediction.decode_time,
                        # Scaling-specific metrics
                        "num_samples": scaling_config.get("num_samples", 8),
                        "all_samples": prediction.scaling_details['all_samples'],
                        "vote_counts": prediction.scaling_details['vote_counts'],
                        "voting_confidence": prediction.scaling_details['voting_confidence'],
                        "avg_time_per_sample": prediction.scaling_details['avg_time_per_sample']
                    }
                    question_results.append(question_result)
                    write_csv_row(actual_question_idx, ex, prediction, None, None, is_correct, formatted_prompt=prompt)
                    monitor.record_question_result(actual_question_idx, prediction)
                    
                    # Save partial results every 10 questions
                    if (i + 1) % 10 == 0:
                        print(f"[PROGRESS] Completed {i+1}/{len(examples)} questions, accuracy so far: {correct_count/(i+1):.4f}")
        
        # Calculate final metrics
        accuracy = correct_count / len(examples) if compute_metrics else -1
        avg_time = sum(r["time_ms"] for r in question_results) / len(question_results)
        avg_tps = sum(r["tokens_per_second"] for r in question_results) / len(question_results)
        avg_ttft = sum(r["ttft"] for r in question_results) / len(question_results)
        avg_decode_time = sum(r["decode_time"] for r in question_results) / len(question_results)

        result = EvaluationResult(
            config_name=self.config.name,
            model_name=model_name,
            subject=self.task,
            total_questions=len(examples),
            correct_answers=correct_count,
            accuracy=accuracy,
            avg_time_per_question=avg_time,
            avg_tokens_per_second=avg_tps,
            question_results=question_results,
        )
        
        # Add scaling metrics
        if self.voting_confidence_scores:
            avg_voting_confidence = sum(self.voting_confidence_scores) / len(self.voting_confidence_scores)
            result.scaling_metrics = {
                'total_samples_generated': self.total_samples_generated,
                'samples_per_question': self.total_samples_generated / len(examples) if examples else 0,
                'avg_voting_confidence': avg_voting_confidence,
                'scaling_efficiency': self._calculate_scaling_efficiency(result, avg_voting_confidence),
                'avg_ttft': avg_ttft,
                'avg_decode_time': avg_decode_time,
                'avg_time_per_sample': avg_time / scaling_config.get("num_samples", 8)
            }

        if self.config.output.get("save_detailed_responses", True):
            self._save_detailed_results(result, output_dir, run_name)

        return result
    
    def _calculate_scaling_efficiency(self, result, avg_confidence: float) -> float:
        """
        Calculate scaling efficiency based on accuracy improvement vs computational cost.
        
        For Natural Plan tasks, efficiency combines:
        - Task accuracy (primary metric)
        - Voting confidence (consensus strength)  
        - Sample efficiency (accuracy per sample)
        """
        # Base efficiency: accuracy weighted by confidence
        base_efficiency = (result.accuracy * 0.7) + (avg_confidence * 0.3)
        
        # Sample efficiency: penalize excessive sampling if accuracy is low
        samples_per_question = self.total_samples_generated / result.total_questions if result.total_questions > 0 else 1
        sample_penalty = max(0.8, 1.0 - (samples_per_question - 1) * 0.02)  
        efficiency = base_efficiency * sample_penalty
        return min(efficiency, 1.0)
    
    def print_summary(self, result) -> None:
        """Print scaling evaluation summary with detailed scaling metrics."""
        print(f"\n{'='*60}")
        print(f"Config: {result.config_name}")
        print(f"Model: {result.model_name}")
        print(f"Task: {result.subject}")
        print(f"Questions: {result.total_questions}")
        print(f"Correct: {result.correct_answers}")
        print(f"Accuracy: {result.accuracy:.4f} ({result.accuracy*100:.2f}%)")
        print(f"Avg Time/Question: {result.avg_time_per_question:.2f}ms")
        print(f"Avg Tokens/Second: {result.avg_tokens_per_second:.2f}")
        print(f"{'='*60}")
        
        # Print detailed scaling metrics
        if hasattr(result, 'scaling_metrics') and result.scaling_metrics:
            scaling_metrics = result.scaling_metrics
            print(f"\n🔄 TEST-TIME SCALING METRICS:")
            print(f"Total Samples Generated: {scaling_metrics['total_samples_generated']}")
            print(f"Samples per Question: {scaling_metrics['samples_per_question']:.1f}")
            print(f"Avg Voting Confidence: {scaling_metrics['avg_voting_confidence']:.4f}")
            print(f"Scaling Efficiency: {scaling_metrics['scaling_efficiency']:.4f}")
            
            # Performance metrics
            print(f"\n⚡ PERFORMANCE METRICS:")
            print(f"Avg Time to First Token: {scaling_metrics['avg_ttft']:.2f}ms")
            print(f"Avg Decode Time: {scaling_metrics['avg_decode_time']:.2f}ms")
            print(f"Avg Time per Sample: {scaling_metrics['avg_time_per_sample']:.2f}ms")
            
            if self.voting_confidence_scores:
                high_confidence_count = sum(1 for conf in self.voting_confidence_scores if conf >= 0.8)
                low_confidence_count = sum(1 for conf in self.voting_confidence_scores if conf < 0.5)
                
                print(f"\n📊 CONFIDENCE DISTRIBUTION:")
                print(f"High Confidence (≥0.8): {high_confidence_count}/{len(self.voting_confidence_scores)} ({high_confidence_count/len(self.voting_confidence_scores)*100:.1f}%)")
                print(f"Low Confidence (<0.5): {low_confidence_count}/{len(self.voting_confidence_scores)} ({low_confidence_count/len(self.voting_confidence_scores)*100:.1f}%)")
            
            print(f"{'='*60}")