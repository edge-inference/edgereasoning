"""
Custom Loader for Controlled MMLU Experiments.

Allows selection of specific questions by subject and question_id for controlled input length studies.
"""

import csv
import os
from typing import List, Dict, Optional
from .mmlu_loader import MMLUQuestion

class CustomLoader:
    """
    Custom loader for controlled MMLU experiments.
    - Selects questions by subject and question_id.
    - Builds custom datasets for input length sweeps.
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.questions = self._load_questions()

    def _load_questions(self) -> List[Dict]:
        questions = []
        with open(self.csv_path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or not row.get('input_tokens'):
                    print("DEBUG: Row keys:", row.keys())
                    print("DEBUG: Row raw:", row)
                    print("[WARN] Skipping empty or malformed row")
                    continue
                try:
                    choices = eval(row['choices']) if isinstance(row['choices'], str) else row['choices']
                    
                    # Validate choices
                    if not isinstance(choices, list):
                        print(f"[WARN] Choices is not a list for question {row.get('question_id', 'unknown')}: {type(choices)}")
                        continue
                    
                    if len(choices) != 4:
                        print(f"[WARN] Question {row.get('question_id', 'unknown')} has {len(choices)} choices instead of 4")
                        continue
                    
                    # Ensure all choices are non-empty strings
                    choices = [str(choice).strip() for choice in choices]
                    if any(not choice for choice in choices):
                        print(f"[WARN] Question {row.get('question_id', 'unknown')} has empty choices")
                        continue
                    
                    questions.append({
                        'input_tokens': int(row['input_tokens']),
                        'output_tokens': int(row['output_tokens']),
                        'question_id': str(row['question_id']),
                        'subject': row['subject'],
                        'question': row['question'],
                        'choices': choices,
                        'correct_answer': row['correct_answer']
                    })
                except KeyError as e:
                    print(f"[WARN] Missing field {e} in row: {row.keys()}")
                except Exception as e:
                    print(f"[WARN] Error parsing row: {e}")
        return questions

    def get_question(self, subject: str, input_tokens: int) -> Optional[MMLUQuestion]:
        """
        Get a question by subject and input token count.
        Returns the first match found.
        """
        for q in self.questions:
            if q['subject'] == subject and q['input_tokens'] == input_tokens:
                return MMLUQuestion(
                    question_id=q['question_id'],
                    subject=q['subject'],
                    question=q['question'],
                    choices=q['choices'],
                    correct_answer=q['correct_answer'],
                    raw_data=q
                )
        return None

    def get_question_by_input_tokens(self, input_tokens: int) -> Optional[MMLUQuestion]:
        """
        Get a question by its input token count.
        
        Args:
            input_tokens: Number of input tokens to match
            
        Returns:
            MMLUQuestion if found, None otherwise
        """
        for question in self.questions:
            if question['input_tokens'] == input_tokens:
                return MMLUQuestion(
                    question_id=question['question_id'],
                    subject=question['subject'],
                    question=question['question'],
                    choices=question['choices'],
                    correct_answer=question['correct_answer'],
                    raw_data=question
                )
        
        return None

    def build_custom_dataset(self, input_token_list: List[int]) -> List[MMLUQuestion]:
        """
        Build a dataset with one question per input token length (any subject).
        For input_tokens==1, always generate a synthetic question.
        """
        dataset = []
        for input_tokens in input_token_list:
            if input_tokens == 1:
                # Synthetic 1-token question
                q = MMLUQuestion(
                    question_id="synthetic_1tok",
                    subject="synthetic",
                    question="?",
                    choices=["A", "B", "C", "D"],
                    correct_answer="A",
                    raw_data={
                        'input_tokens': 1,
                        'output_tokens': 0,
                        'question_id': "synthetic_1tok",
                        'subject': "synthetic",
                        'question': "?",
                        'choices': ["A", "B", "C", "D"],
                        'correct_answer': "A"
                    }
                )
                dataset.append(q)
            else:
                q = self.get_question_by_input_tokens(input_tokens)
                if q:
                    dataset.append(q)
        return dataset

    def get_available_subjects(self) -> List[str]:
        return sorted(set(q['subject'] for q in self.questions))
