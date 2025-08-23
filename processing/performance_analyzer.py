"""
Performance Analyzer

Advanced analysis of model performance across subjects and metrics.
Provides statistical analysis, ranking, and visualization capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

from .data_models import ConsolidatedResult, ModelResult, SubjectResult


class PerformanceAnalyzer:
    """
    Advanced performance analysis for MMLU evaluation results.
    
    Features:
    - Statistical analysis across models and subjects
    - Performance ranking and comparison
    - Correlation analysis between metrics
    - Visualization generation
    - Outlier detection and analysis
    """
    
    def __init__(self, consolidated_results: ConsolidatedResult):
        """
        Initialize analyzer with consolidated results.
        
        Args:
            consolidated_results: Consolidated evaluation results
        """
        self.results = consolidated_results
        self.logger = logging.getLogger(__name__)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def get_accuracy_rankings(self) -> pd.DataFrame:
        """
        Get model accuracy rankings across all subjects.
        
        Returns:
            DataFrame with accuracy rankings and statistics
        """
        rankings = []
        
        for model_name, model_result in self.results.models.items():
            rankings.append({
                'model_name': model_name,
                'overall_accuracy': model_result.overall_accuracy,
                'total_questions': model_result.total_questions,
                'correct_answers': model_result.total_correct,
                'subjects_count': len(model_result.subjects),
                'successful_subjects': model_result.successful_subjects
            })
        
        df = pd.DataFrame(rankings)
        df = df.sort_values('overall_accuracy', ascending=False)
        df['accuracy_rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_performance_rankings(self) -> pd.DataFrame:
        """
        Get model performance rankings across timing metrics.
        
        Returns:
            DataFrame with performance rankings
        """
        rankings = []
        
        for model_name, model_result in self.results.models.items():
            rankings.append({
                'model_name': model_name,
                'avg_ttft_ms': model_result.avg_ttft,
                'avg_decode_time_ms': model_result.avg_decode_time,
                'avg_total_time_ms': model_result.avg_total_time,
                'avg_tokens_per_second': model_result.avg_tokens_per_second,
                'total_tokens': model_result.total_output_tokens,
                'overall_accuracy': model_result.overall_accuracy
            })
        
        df = pd.DataFrame(rankings)
        
        # Add rankings (lower is better for time metrics, higher for tokens/sec)
        df['ttft_rank'] = df['avg_ttft_ms'].rank(method='min')
        df['decode_time_rank'] = df['avg_decode_time_ms'].rank(method='min')
        df['total_time_rank'] = df['avg_total_time_ms'].rank(method='min')
        df['tokens_per_sec_rank'] = df['avg_tokens_per_second'].rank(method='min', ascending=False)
        df['accuracy_rank'] = df['overall_accuracy'].rank(method='min', ascending=False)
        
        # Calculate composite performance score (lower is better)
        df['composite_score'] = (df['ttft_rank'] + df['decode_time_rank'] + 
                                df['total_time_rank'] + df['tokens_per_sec_rank']) / 4
        df['composite_rank'] = df['composite_score'].rank(method='min')
        
        return df.sort_values('composite_rank')
    
    def get_subject_analysis(self) -> pd.DataFrame:
        """
        Analyze performance across different subjects.
        
        Returns:
            DataFrame with subject-level statistics
        """
        subject_data = []
        
        # Collect all subject results
        for model_name, model_result in self.results.models.items():
            for subject_name, subject_result in model_result.subjects.items():
                subject_data.append({
                    'model_name': model_name,
                    'subject': subject_name,
                    'accuracy': subject_result.accuracy,
                    'total_questions': subject_result.total_questions,
                    'avg_ttft_ms': subject_result.avg_ttft,
                    'avg_tokens_per_second': subject_result.avg_tokens_per_second,
                    'avg_total_time_ms': subject_result.avg_total_time
                })
        
        df = pd.DataFrame(subject_data)
        
        # Calculate subject-level statistics
        subject_stats = df.groupby('subject').agg({
            'accuracy': ['mean', 'std', 'min', 'max'],
            'total_questions': 'first',
            'avg_ttft_ms': ['mean', 'std'],
            'avg_tokens_per_second': ['mean', 'std'],
            'avg_total_time_ms': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        subject_stats.columns = ['_'.join(col).strip() for col in subject_stats.columns]
        subject_stats = subject_stats.reset_index()
        
        # Add difficulty ranking (lower accuracy = harder)
        subject_stats['difficulty_rank'] = subject_stats['accuracy_mean'].rank(method='min')
        
        return subject_stats.sort_values('difficulty_rank')
    
    def analyze_correlations(self) -> pd.DataFrame:
        """
        Analyze correlations between different performance metrics.
        
        Returns:
            Correlation matrix DataFrame
        """
        metrics_data = []
        
        for model_name, model_result in self.results.models.items():
            metrics_data.append({
                'model_name': model_name,
                'accuracy': model_result.overall_accuracy,
                'ttft_ms': model_result.avg_ttft,
                'decode_time_ms': model_result.avg_decode_time,
                'total_time_ms': model_result.avg_total_time,
                'tokens_per_second': model_result.avg_tokens_per_second,
                'total_input_tokens': model_result.total_input_tokens,
                'total_output_tokens': model_result.total_output_tokens
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Calculate correlation matrix
        numeric_cols = ['accuracy', 'ttft_ms', 'decode_time_ms', 'total_time_ms', 
                       'tokens_per_second', 'total_input_tokens', 'total_output_tokens']
        
        correlation_matrix = df[numeric_cols].corr()
        
        return correlation_matrix
    
    def detect_outliers(self, metric: str = 'accuracy') -> pd.DataFrame:
        """
        Detect outliers in performance metrics using IQR method.
        
        Args:
            metric: Metric to analyze for outliers
            
        Returns:
            DataFrame with outlier analysis
        """
        data = []
        
        if metric == 'accuracy':
            for model_name, model_result in self.results.models.items():
                for subject_name, subject_result in model_result.subjects.items():
                    data.append({
                        'model_name': model_name,
                        'subject': subject_name,
                        'value': subject_result.accuracy
                    })
        elif metric == 'tokens_per_second':
            for model_name, model_result in self.results.models.items():
                for subject_name, subject_result in model_result.subjects.items():
                    data.append({
                        'model_name': model_name,
                        'subject': subject_name,
                        'value': subject_result.avg_tokens_per_second
                    })
        else:
            self.logger.warning(f"Unsupported metric for outlier detection: {metric}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
        
        # Calculate IQR
        Q1 = df['value'].quantile(0.25)
        Q3 = df['value'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        df['is_outlier'] = (df['value'] < lower_bound) | (df['value'] > upper_bound)
        df['outlier_type'] = df.apply(
            lambda row: 'low' if row['value'] < lower_bound else 
                       ('high' if row['value'] > upper_bound else 'normal'), 
            axis=1
        )
        
        return df[df['is_outlier']].sort_values('value')
    
    def create_accuracy_comparison_plot(self, output_path: str) -> None:
        """
        Create a bar plot comparing model accuracies.
        
        Args:
            output_path: Path to save the plot
        """
        rankings = self.get_accuracy_rankings()
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(rankings['model_name'], rankings['overall_accuracy'])
        
        # Color bars by rank
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(rankings)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Overall Accuracy', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, rankings['overall_accuracy']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{accuracy:.2%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Accuracy comparison plot saved: {output_path}")
    
    def create_performance_heatmap(self, output_path: str) -> None:
        """
        Create a heatmap showing model performance across subjects.
        
        Args:
            output_path: Path to save the plot
        """
        # Prepare data for heatmap
        subject_data = []
        for model_name, model_result in self.results.models.items():
            for subject_name, subject_result in model_result.subjects.items():
                subject_data.append({
                    'model_name': model_name,
                    'subject': subject_name,
                    'accuracy': subject_result.accuracy
                })
        
        df = pd.DataFrame(subject_data)
        
        if df.empty:
            self.logger.warning("No data available for heatmap")
            return
        
        # Pivot for heatmap
        heatmap_data = df.pivot(index='model_name', columns='subject', values='accuracy')
        
        plt.figure(figsize=(20, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0.5, vmin=0, vmax=1)
        
        plt.title('Model Performance Heatmap by Subject', fontsize=16, fontweight='bold')
        plt.xlabel('Subject', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance heatmap saved: {output_path}")
    
    def create_correlation_plot(self, output_path: str) -> None:
        """
        Create a correlation matrix plot for performance metrics.
        
        Args:
            output_path: Path to save the plot
        """
        correlation_matrix = self.analyze_correlations()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, square=True, linewidths=0.5)
        
        plt.title('Performance Metrics Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Correlation plot saved: {output_path}")
    
    def create_timing_comparison_plot(self, output_path: str) -> None:
        """
        Create a comparison plot for timing metrics.
        
        Args:
            output_path: Path to save the plot
        """
        performance_data = []
        
        for model_name, model_result in self.results.models.items():
            performance_data.append({
                'model_name': model_name,
                'TTFT (ms)': model_result.avg_ttft,
                'Decode Time (ms)': model_result.avg_decode_time,
                'Total Time (ms)': model_result.avg_total_time,
                'Tokens/Second': model_result.avg_tokens_per_second
            })
        
        df = pd.DataFrame(performance_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # TTFT comparison
        axes[0, 0].bar(df['model_name'], df['TTFT (ms)'])
        axes[0, 0].set_title('Time to First Token (TTFT)')
        axes[0, 0].set_ylabel('Milliseconds')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Decode time comparison
        axes[0, 1].bar(df['model_name'], df['Decode Time (ms)'])
        axes[0, 1].set_title('Decode Time')
        axes[0, 1].set_ylabel('Milliseconds')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Total time comparison
        axes[1, 0].bar(df['model_name'], df['Total Time (ms)'])
        axes[1, 0].set_title('Total Time')
        axes[1, 0].set_ylabel('Milliseconds')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Tokens per second comparison
        axes[1, 1].bar(df['model_name'], df['Tokens/Second'])
        axes[1, 1].set_title('Generation Speed')
        axes[1, 1].set_ylabel('Tokens per Second')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Timing comparison plot saved: {output_path}")
    
    def generate_analysis_report(self, output_dir: str) -> None:
        """
        Generate comprehensive analysis report with all plots and statistics.
        
        Args:
            output_dir: Directory to save analysis outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate statistical analyses
        accuracy_rankings = self.get_accuracy_rankings()
        performance_rankings = self.get_performance_rankings()
        subject_analysis = self.get_subject_analysis()
        correlation_matrix = self.analyze_correlations()
        outliers = self.detect_outliers('accuracy')
        
        # Save analysis tables
        accuracy_rankings.to_csv(output_path / f"accuracy_rankings_{timestamp}.csv", index=False)
        performance_rankings.to_csv(output_path / f"performance_rankings_{timestamp}.csv", index=False)
        subject_analysis.to_csv(output_path / f"subject_analysis_{timestamp}.csv", index=False)
        correlation_matrix.to_csv(output_path / f"correlation_matrix_{timestamp}.csv")
        
        if not outliers.empty:
            outliers.to_csv(output_path / f"accuracy_outliers_{timestamp}.csv", index=False)
        
        # Generate plots
        self.create_accuracy_comparison_plot(output_path / f"accuracy_comparison_{timestamp}.png")
        self.create_performance_heatmap(output_path / f"performance_heatmap_{timestamp}.png")
        self.create_correlation_plot(output_path / f"correlation_matrix_{timestamp}.png")
        self.create_timing_comparison_plot(output_path / f"timing_comparison_{timestamp}.png")
        
        self.logger.info(f"Analysis report generated in: {output_dir}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)
        print(f"Models analyzed: {len(self.results.models)}")
        print(f"Analysis timestamp: {timestamp}")
        print("\nTop 3 Models by Accuracy:")
        for i, row in accuracy_rankings.head(3).iterrows():
            print(f"  {row['accuracy_rank']}. {row['model_name']}: {row['overall_accuracy']:.2%}")
        print("\nTop 3 Models by Performance:")
        for i, row in performance_rankings.head(3).iterrows():
            print(f"  {int(row['composite_rank'])}. {row['model_name']}: {row['avg_tokens_per_second']:.1f} tok/s")
        print("="*80)
