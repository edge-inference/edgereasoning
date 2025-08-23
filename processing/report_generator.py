"""
Report Generator

Generates comprehensive evaluation reports in multiple formats.
Creates executive summaries, detailed analysis, and presentation-ready outputs.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from jinja2 import Template
import logging

from .data_models import ConsolidatedResult, ModelResult


class ReportGenerator:
    """
    Generates comprehensive evaluation reports.
    
    Features:
    - Executive summary reports
    - Detailed technical analysis
    - HTML presentation reports
    - JSON data exports
    - Markdown documentation
    """
    
    def __init__(self, consolidated_results: ConsolidatedResult):
        """
        Initialize report generator.
        
        Args:
            consolidated_results: Consolidated evaluation results
        """
        self.results = consolidated_results
        self.logger = logging.getLogger(__name__)
    
    def generate_executive_summary(self) -> Dict:
        """
        Generate executive summary of evaluation results.
        
        Returns:
            Dictionary containing executive summary data
        """
        if not self.results.models:
            return {"error": "No models available for summary"}
        
        # Calculate summary statistics
        all_accuracies = [model.overall_accuracy for model in self.results.models.values()]
        all_questions = sum(model.total_questions for model in self.results.models.values())
        all_subjects = set()
        for model in self.results.models.values():
            all_subjects.update(model.subjects.keys())
        
        # Find best and worst performing models
        best_model = max(self.results.models.values(), key=lambda m: m.overall_accuracy)
        worst_model = min(self.results.models.values(), key=lambda m: m.overall_accuracy)
        fastest_model = max(self.results.models.values(), key=lambda m: m.avg_tokens_per_second)
        
        summary = {
            "evaluation_overview": {
                "models_evaluated": len(self.results.models),
                "total_subjects": len(all_subjects),
                "total_questions": all_questions,
                "evaluation_date": self.results.processing_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            },
            "performance_summary": {
                "best_accuracy": {
                    "model": best_model.model_name,
                    "accuracy": best_model.overall_accuracy,
                    "correct_answers": best_model.total_correct,
                    "total_questions": best_model.total_questions
                },
                "worst_accuracy": {
                    "model": worst_model.model_name,
                    "accuracy": worst_model.overall_accuracy,
                    "correct_answers": worst_model.total_correct,
                    "total_questions": worst_model.total_questions
                },
                "fastest_model": {
                    "model": fastest_model.model_name,
                    "tokens_per_second": fastest_model.avg_tokens_per_second,
                    "avg_total_time_ms": fastest_model.avg_total_time
                },
                "accuracy_range": {
                    "min": min(all_accuracies),
                    "max": max(all_accuracies),
                    "mean": sum(all_accuracies) / len(all_accuracies),
                    "std": pd.Series(all_accuracies).std()
                }
            },
            "subject_analysis": {
                "subjects_evaluated": sorted(list(all_subjects)),
                "hardest_subjects": self._get_hardest_subjects(),
                "easiest_subjects": self._get_easiest_subjects()
            },
            "recommendations": self._generate_recommendations()
        }
        
        return summary
    
    def _get_hardest_subjects(self, top_n: int = 5) -> List[Dict]:
        """Get the hardest subjects based on average accuracy."""
        subject_accuracies = {}
        
        for model in self.results.models.values():
            for subject_name, subject_result in model.subjects.items():
                if subject_name not in subject_accuracies:
                    subject_accuracies[subject_name] = []
                subject_accuracies[subject_name].append(subject_result.accuracy)
        
        # Calculate average accuracy per subject
        avg_accuracies = {
            subject: sum(accuracies) / len(accuracies)
            for subject, accuracies in subject_accuracies.items()
        }
        
        # Sort by difficulty (lowest accuracy = hardest)
        hardest = sorted(avg_accuracies.items(), key=lambda x: x[1])[:top_n]
        
        return [{"subject": subject, "avg_accuracy": accuracy} for subject, accuracy in hardest]
    
    def _get_easiest_subjects(self, top_n: int = 5) -> List[Dict]:
        """Get the easiest subjects based on average accuracy."""
        subject_accuracies = {}
        
        for model in self.results.models.values():
            for subject_name, subject_result in model.subjects.items():
                if subject_name not in subject_accuracies:
                    subject_accuracies[subject_name] = []
                subject_accuracies[subject_name].append(subject_result.accuracy)
        
        # Calculate average accuracy per subject
        avg_accuracies = {
            subject: sum(accuracies) / len(accuracies)
            for subject, accuracies in subject_accuracies.items()
        }
        
        # Sort by ease (highest accuracy = easiest)
        easiest = sorted(avg_accuracies.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return [{"subject": subject, "avg_accuracy": accuracy} for subject, accuracy in easiest]
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        if not self.results.models:
            return recommendations
        
        # Accuracy-based recommendations
        accuracies = [model.overall_accuracy for model in self.results.models.values()]
        avg_accuracy = sum(accuracies) / len(accuracies)
        
        if avg_accuracy < 0.5:
            recommendations.append("Overall model performance is below 50%. Consider fine-tuning or using larger models.")
        elif avg_accuracy > 0.8:
            recommendations.append("Excellent overall performance! Consider evaluating on more challenging datasets.")
        
        # Performance-based recommendations
        speeds = [model.avg_tokens_per_second for model in self.results.models.values()]
        speed_variance = pd.Series(speeds).std()
        
        if speed_variance > 50:
            recommendations.append("High variance in generation speed. Investigate model size and optimization differences.")
        
        # Subject-specific recommendations
        hardest_subjects = self._get_hardest_subjects(3)
        if hardest_subjects and hardest_subjects[0]["avg_accuracy"] < 0.3:
            recommendations.append(f"Very low performance on {hardest_subjects[0]['subject']}. Consider subject-specific training.")
        
        return recommendations
    
    def generate_detailed_report(self) -> Dict:
        """
        Generate detailed technical report.
        
        Returns:
            Dictionary containing detailed analysis
        """
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "models_count": len(self.results.models),
                "processor_version": "1.0.0"
            },
            "executive_summary": self.generate_executive_summary(),
            "model_details": {},
            "comparative_analysis": self._generate_comparative_analysis(),
            "technical_metrics": self._generate_technical_metrics()
        }
        
        # Add detailed model information
        for model_name, model_result in self.results.models.items():
            report["model_details"][model_name] = {
                "overall_accuracy": model_result.overall_accuracy,
                "total_questions": model_result.total_questions,
                "total_correct": model_result.total_correct,
                "subjects_count": len(model_result.subjects),
                "avg_ttft_ms": model_result.avg_ttft,
                "avg_decode_time_ms": model_result.avg_decode_time,
                "avg_total_time_ms": model_result.avg_total_time,
                "avg_tokens_per_second": model_result.avg_tokens_per_second,
                "total_input_tokens": model_result.total_input_tokens,
                "total_output_tokens": model_result.total_output_tokens,
                "timestamp": model_result.timestamp.isoformat(),
                "config_name": model_result.config_name,
                "subject_performance": {
                    subject_name: {
                        "accuracy": subject_result.accuracy,
                        "questions": subject_result.total_questions,
                        "correct": subject_result.correct_answers
                    }
                    for subject_name, subject_result in model_result.subjects.items()
                }
            }
        
        return report
    
    def _generate_comparative_analysis(self) -> Dict:
        """Generate comparative analysis across models."""
        if len(self.results.models) < 2:
            return {"note": "Comparative analysis requires at least 2 models"}
        
        models = list(self.results.models.values())
        
        # Accuracy comparison
        accuracies = [model.overall_accuracy for model in models]
        accuracy_diff = max(accuracies) - min(accuracies)
        
        # Speed comparison
        speeds = [model.avg_tokens_per_second for model in models]
        speed_diff = max(speeds) - min(speeds)
        
        return {
            "accuracy_spread": {
                "range": accuracy_diff,
                "coefficient_of_variation": pd.Series(accuracies).std() / pd.Series(accuracies).mean()
            },
            "speed_spread": {
                "range": speed_diff,
                "coefficient_of_variation": pd.Series(speeds).std() / pd.Series(speeds).mean()
            },
            "efficiency_analysis": self._analyze_efficiency()
        }
    
    def _analyze_efficiency(self) -> Dict:
        """Analyze accuracy vs speed efficiency."""
        efficiency_data = []
        
        for model_name, model_result in self.results.models.items():
            efficiency_score = model_result.overall_accuracy * model_result.avg_tokens_per_second
            efficiency_data.append({
                "model": model_name,
                "accuracy": model_result.overall_accuracy,
                "speed": model_result.avg_tokens_per_second,
                "efficiency_score": efficiency_score
            })
        
        # Find most efficient model
        best_efficiency = max(efficiency_data, key=lambda x: x["efficiency_score"])
        
        return {
            "most_efficient_model": best_efficiency,
            "efficiency_ranking": sorted(efficiency_data, key=lambda x: x["efficiency_score"], reverse=True)
        }
    
    def _generate_technical_metrics(self) -> Dict:
        """Generate technical performance metrics."""
        all_ttft = []
        all_decode_times = []
        all_speeds = []
        
        for model in self.results.models.values():
            all_ttft.append(model.avg_ttft)
            all_decode_times.append(model.avg_decode_time)
            all_speeds.append(model.avg_tokens_per_second)
        
        return {
            "ttft_statistics": {
                "min": min(all_ttft),
                "max": max(all_ttft),
                "mean": sum(all_ttft) / len(all_ttft),
                "std": pd.Series(all_ttft).std()
            },
            "decode_time_statistics": {
                "min": min(all_decode_times),
                "max": max(all_decode_times),
                "mean": sum(all_decode_times) / len(all_decode_times),
                "std": pd.Series(all_decode_times).std()
            },
            "speed_statistics": {
                "min": min(all_speeds),
                "max": max(all_speeds),
                "mean": sum(all_speeds) / len(all_speeds),
                "std": pd.Series(all_speeds).std()
            }
        }
    
    def save_json_report(self, output_path: str) -> None:
        """
        Save detailed report as JSON.
        
        Args:
            output_path: Path to save the JSON report
        """
        report = self.generate_detailed_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"JSON report saved: {output_path}")
    
    def save_markdown_report(self, output_path: str) -> None:
        """
        Save executive summary as Markdown.
        
        Args:
            output_path: Path to save the Markdown report
        """
        summary = self.generate_executive_summary()
        
        markdown_template = """# MMLU Evaluation Report
        
## Executive Summary

**Evaluation Date:** {{ summary.evaluation_overview.evaluation_date }}  
**Models Evaluated:** {{ summary.evaluation_overview.models_evaluated }}  
**Total Subjects:** {{ summary.evaluation_overview.total_subjects }}  
**Total Questions:** {{ summary.evaluation_overview.total_questions }}  

## Performance Highlights

### Best Performing Model
- **Model:** {{ summary.performance_summary.best_accuracy.model }}
- **Accuracy:** {{ "%.2f%%" | format(summary.performance_summary.best_accuracy.accuracy * 100) }}
- **Correct Answers:** {{ summary.performance_summary.best_accuracy.correct_answers }}/{{ summary.performance_summary.best_accuracy.total_questions }}

### Fastest Model
- **Model:** {{ summary.performance_summary.fastest_model.model }}
- **Speed:** {{ "%.1f" | format(summary.performance_summary.fastest_model.tokens_per_second) }} tokens/second
- **Avg Time:** {{ "%.1f" | format(summary.performance_summary.fastest_model.avg_total_time_ms) }}ms

## Subject Analysis

### Hardest Subjects
{% for subject in summary.subject_analysis.hardest_subjects %}
- **{{ subject.subject }}:** {{ "%.2f%%" | format(subject.avg_accuracy * 100) }}
{% endfor %}

### Easiest Subjects  
{% for subject in summary.subject_analysis.easiest_subjects %}
- **{{ subject.subject }}:** {{ "%.2f%%" | format(subject.avg_accuracy * 100) }}
{% endfor %}

## Recommendations
{% for rec in summary.recommendations %}
- {{ rec }}
{% endfor %}

---
*Report generated by MMLU Evaluation Processor*
"""
        
        template = Template(markdown_template)
        markdown_content = template.render(summary=summary)
        
        with open(output_path, 'w') as f:
            f.write(markdown_content)
        
        self.logger.info(f"Markdown report saved: {output_path}")
    
    def generate_all_reports(self, output_dir: str) -> None:
        """
        Generate all report formats.
        
        Args:
            output_dir: Directory to save all reports
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate all report formats
        self.save_json_report(output_path / f"detailed_report_{timestamp}.json")
        self.save_markdown_report(output_path / f"executive_summary_{timestamp}.md")
        
        # Save executive summary as JSON for easy parsing
        summary = self.generate_executive_summary()
        with open(output_path / f"executive_summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"All reports generated in: {output_dir}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("EVALUATION REPORT SUMMARY")
        print("="*80)
        print(f"Models evaluated: {summary['evaluation_overview']['models_evaluated']}")
        print(f"Total questions: {summary['evaluation_overview']['total_questions']:,}")
        print(f"Best model: {summary['performance_summary']['best_accuracy']['model']} "
              f"({summary['performance_summary']['best_accuracy']['accuracy']:.2%})")
        print(f"Fastest model: {summary['performance_summary']['fastest_model']['model']} "
              f"({summary['performance_summary']['fastest_model']['tokens_per_second']:.1f} tok/s)")
        print("="*80)
