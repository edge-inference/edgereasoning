"""
Result Consolidator

Consolidates detailed CSV results 
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import asdict

from .data_models import QuestionResult, SubjectResult, ModelResult, ConsolidatedResult


class ResultConsolidator:
    
    def __init__(self, results_base_dir: str = "./results"):
        self.results_base_dir = Path(results_base_dir)
        self.logger = logging.getLogger(__name__)
        
    def discover_model_directories(self) -> Dict[str, Path]:
        """Discover model directories."""
        model_dirs = {}
        
        if not self.results_base_dir.exists():
            self.logger.warning(f"Results directory not found: {self.results_base_dir}")
            return model_dirs
            
        for item in self.results_base_dir.iterdir():
            if item.is_dir() and not item.name.endswith('.log') and item.name != 'processed_results':
                # Use directory name as key - JSON files will provide clean model names
                model_dirs[item.name] = item
                self.logger.debug(f"Found directory: {item.name}")
                    
        self.logger.info(f"Discovered {len(model_dirs)} model directories")
        return model_dirs
    
    def load_subject_csv(self, csv_path: Path) -> List[QuestionResult]:
        questions = []
        
        try:
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                try:
                    question = QuestionResult.from_csv_row(row.to_dict())
                    questions.append(question)
                except Exception as e:
                    self.logger.warning(f"Failed to parse question row in {csv_path}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Failed to load CSV {csv_path}: {e}")
            
        return questions
    
    def load_model_results(self, directory_name: str, model_dir: Path) -> Optional[ModelResult]:
        subjects = {}
        clean_model_name = directory_name  # fallback
        
        summary_path = model_dir / "summary.json"
        timestamp = datetime.now()
        config_name = "unknown"
        
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                    timestamp_str = summary.get('timestamp', '')
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        except:
                            pass
                    config_name = summary.get('config', 'unknown')
            except Exception as e:
                self.logger.warning(f"Failed to load summary for {directory_name}: {e}")
        
        # Look for JSON result files instead of CSV files
        json_files = list(model_dir.glob("results_*.json"))
        
        if json_files:
            self.logger.info(f"Found JSON result files in {directory_name} with {len(json_files)} files")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                    
                    # Extract clean model name and other info from JSON
                    clean_model_name = json_data.get('model_name', directory_name)
                    subject_name = json_data.get('subject', 'unknown')
                    question_results = json_data.get('question_results', [])
                    
                    # Convert JSON question results to QuestionResult objects
                    questions = []
                    for q_data in question_results:
                        try:
                            question = QuestionResult(
                                question_id=q_data.get('question_id', 0),
                                subject=subject_name,
                                question=q_data.get('question', ''),
                                choices=q_data.get('choices', []),
                                correct_answer=q_data.get('correct_answer', ''),
                                predicted_choice=q_data.get('predicted_choice', ''),
                                is_correct=q_data.get('is_correct', False),
                                generated_text=q_data.get('generated_text', ''),
                                ttft=float(q_data.get('ttft', 0.0)),
                                decode_time=float(q_data.get('decode_time', 0.0)),
                                total_time_ms=float(q_data.get('time_ms', 0.0)),
                                tokens_per_second=float(q_data.get('tokens_per_second', 0.0)),
                                input_tokens=int(q_data.get('input_tokens', 0)),
                                output_tokens=int(q_data.get('output_tokens', 0)),
                                generated_text_length=len(q_data.get('generated_text', ''))
                            )
                            questions.append(question)
                        except Exception as e:
                            self.logger.warning(f"Failed to parse question in {json_file.name}: {e}")
                    
                    if questions:
                        if subject_name not in subjects:
                            subjects[subject_name] = []
                        subjects[subject_name].extend(questions)
                        self.logger.debug(f"Loaded {len(questions)} questions for {clean_model_name}/{subject_name} from {json_file.name}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to load JSON file {json_file.name}: {e}")
            
            # Convert to SubjectResult objects using the clean model name
            final_subjects = {}
            for subject_name, questions in subjects.items():
                if questions:
                    subject_result = SubjectResult.from_questions(clean_model_name, subject_name, questions)
                    final_subjects[subject_name] = subject_result
            
            subjects = final_subjects
            
        else:
            for subject_dir in model_dir.iterdir():
                if not subject_dir.is_dir():
                    continue
                    
                subject_name = subject_dir.name
                
                csv_files = list(subject_dir.glob("detailed_results_*.csv"))
                if not csv_files:
                    self.logger.warning(f"No detailed results CSV found for {clean_model_name}/{subject_name}")
                    continue
                    
                csv_path = csv_files[0]
                questions = self.load_subject_csv(csv_path)
                
                if questions:
                    subject_result = SubjectResult.from_questions(clean_model_name, subject_name, questions)
                    subjects[subject_name] = subject_result
                    self.logger.debug(f"Loaded {len(questions)} questions for {clean_model_name}/{subject_name}")
                else:
                    self.logger.warning(f"No valid questions loaded for {clean_model_name}/{subject_name}")
        
        if not subjects:
            self.logger.error(f"No subjects loaded for directory {directory_name}")
            return None
            
        model_result = ModelResult.from_subjects(clean_model_name, subjects, timestamp, config_name)
        self.logger.info(f"Loaded model {clean_model_name}: {len(subjects)} subjects, "
                        f"{model_result.total_questions} questions, "
                        f"{model_result.overall_accuracy:.2%} accuracy")
        
        return model_result
    
    def consolidate_all_results(self) -> ConsolidatedResult:
        consolidated = ConsolidatedResult()
        model_dirs = self.discover_model_directories()
        
        for directory_name, model_dir in model_dirs.items():
            self.logger.info(f"Processing directory: {directory_name}")
            # Load model results - JSON files will provide clean model names
            model_result = self.load_model_results(directory_name, model_dir)
            
            if model_result:
                # Use directory_name as unique key to avoid overwrites between different configs
                consolidated.models[directory_name] = model_result
            else:
                self.logger.error(f"Failed to load results for directory: {directory_name}")
        
        self.logger.info(f"Consolidation complete: {len(consolidated.models)} models processed")
        return consolidated
    
    def create_consolidated_csv(self, consolidated: ConsolidatedResult, output_path: str) -> None:
        all_rows = []
        
        for model_name, model_result in consolidated.models.items():
            for subject_name, subject_result in model_result.subjects.items():
                for question in subject_result.questions:
                    row = {
                        'model_name': model_result.model_name,
                        'model_timestamp': model_result.timestamp.isoformat(),
                        'config_name': model_result.config_name,
                        **asdict(question)
                    }
                    all_rows.append(row)
        
        df = pd.DataFrame(all_rows)
        # Remove verbose columns not needed for aggregation
        columns_to_drop = ['question', 'choices', 'generated_text']
        existing_drop = [c for c in columns_to_drop if c in df.columns]
        if existing_drop:
            df = df.drop(columns=existing_drop)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Consolidated CSV saved: {output_path} ({len(all_rows)} questions)")
    
    def create_multi_sheet_excel(self, consolidated: ConsolidatedResult, output_path: str) -> None:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            overview_df = consolidated.get_comparison_dataframe()
            overview_df.to_excel(writer, sheet_name='Overview', index=False)
            
            subject_comparison_df = consolidated.get_subject_comparison_dataframe()
            subject_comparison_df.to_excel(writer, sheet_name='Subject_Comparison', index=False)
            
            for _key, model_result in consolidated.models.items():
                model_rows = []
                
                for subject_name, subject_result in model_result.subjects.items():
                    for question in subject_result.questions:
                        row = {
                            'subject': subject_name,
                            **asdict(question)
                        }
                        model_rows.append(row)
                
                if model_rows:
                    model_df = pd.DataFrame(model_rows)
                    safe_sheet_name = re.sub(r'[^\w\s-]', '_', model_result.model_name)[:31]
                    model_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
            
        self.logger.info(f"Multi-sheet Excel saved: {output_path}")
    
    def create_performance_summary(self, consolidated: ConsolidatedResult, output_path: str) -> None:
        summary_rows = []
        
        for _key, model_result in consolidated.models.items():
            summary_rows.append({
                'model_name': model_result.model_name,
                'metric_type': 'overall',
                'subject': 'ALL',
                'accuracy': model_result.overall_accuracy,
                'total_questions': model_result.total_questions,
                'correct_answers': model_result.total_correct,
                'avg_ttft_ms': model_result.avg_ttft,
                'avg_decode_time_ms': model_result.avg_decode_time,
                'avg_total_time_ms': model_result.avg_total_time,
                'avg_tokens_per_second': model_result.avg_tokens_per_second,
                'total_input_tokens': model_result.total_input_tokens,
                'total_output_tokens': model_result.total_output_tokens,
                'timestamp': model_result.timestamp.isoformat()
            })
            
            for subject_name, subject_result in model_result.subjects.items():
                summary_rows.append({
                    'model_name': model_result.model_name,
                    'metric_type': 'subject',
                    'subject': subject_name,
                    'accuracy': subject_result.accuracy,
                    'total_questions': subject_result.total_questions,
                    'correct_answers': subject_result.correct_answers,
                    'avg_ttft_ms': subject_result.avg_ttft,
                    'avg_decode_time_ms': subject_result.avg_decode_time,
                    'avg_total_time_ms': subject_result.avg_total_time,
                    'avg_tokens_per_second': subject_result.avg_tokens_per_second,
                    'total_input_tokens': subject_result.total_input_tokens,
                    'total_output_tokens': subject_result.total_output_tokens,
                    'timestamp': model_result.timestamp.isoformat()
                })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_path, index=False)
        self.logger.info(f"Performance summary saved: {output_path}")
    
    def process_all(self, output_dir: str = "./processed_results") -> ConsolidatedResult:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        consolidated = self.consolidate_all_results()
        
        if not consolidated.models:
            self.logger.warning("No models found to process")
            return consolidated
        
        self.create_consolidated_csv(
            consolidated, 
            output_path / f"all_results_consolidated_{timestamp}.csv"
        )
        
        self.create_multi_sheet_excel(
            consolidated,
            output_path / f"all_results_by_model_{timestamp}.xlsx"
        )
        
        self.create_performance_summary(
            consolidated,
            output_path / f"performance_summary_{timestamp}.csv"
        )
        
        self.logger.info(f"Processing complete. Results saved to: {output_dir}")
        return consolidated
    

