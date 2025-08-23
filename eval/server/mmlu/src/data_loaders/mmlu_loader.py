"""
Professional MMLU-Redux dataset loader for performance evaluation.

This module provides clean, efficient loading and management of MMLU-Redux datasets
with proper error handling and standardized question formatting.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Iterator
from datasets import load_dataset, get_dataset_config_names
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MMLUQuestion:
    """Standardized MMLU question container."""
    question_id: str
    subject: str
    question: str
    choices: List[str]  # [A, B, C, D] options
    correct_answer: str  # 'A', 'B', 'C', or 'D'
    raw_data: Dict = None  # Original dataset entry for reference


class MMLULoader:
    """
    Professional MMLU-Redux dataset loader.
    
    Features:
    - Efficient dataset loading with caching
    - Subject filtering and management
    - Standardized question format
    - Error handling and validation
    - Subset support for testing
    """
    
    DATASET_NAME = "edinburgh-dawg/mmlu-redux"
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize MMLU loader.
        
        Args:
            cache_dir: Optional cache directory for datasets
        """
        self.cache_dir = cache_dir
        self._available_subjects = None
        self._loaded_datasets = {}  # Cache for loaded datasets
        
    def get_available_subjects(self) -> List[str]:
        """
        Get list of available MMLU subjects.
        
        Returns:
            List of subject names
        """
        if self._available_subjects is None:
            try:
                self._available_subjects = get_dataset_config_names(self.DATASET_NAME)
                logger.info(f"Found {len(self._available_subjects)} MMLU subjects")
            except Exception as e:
                logger.error(f"Failed to get MMLU subjects: {e}")
                self._available_subjects = []
                
        return self._available_subjects
    
    def load_subject(
        self, 
        subject: str, 
        split: str = "test",
        max_questions: Optional[int] = None
    ) -> List[MMLUQuestion]:
        """
        Load questions for a specific subject.
        
        Args:
            subject: Subject name (e.g., 'electrical_engineering')
            split: Dataset split ('test', 'train', 'dev')
            max_questions: Maximum number of questions to load (for testing)
            
        Returns:
            List of MMLUQuestion objects
        """
        cache_key = f"{subject}_{split}"
        
        # Check cache first
        if cache_key in self._loaded_datasets:
            logger.info(f"Using cached dataset: {subject} ({split})")
            questions = self._loaded_datasets[cache_key]
        else:
            # Load from HuggingFace
            try:
                logger.info(f"Loading dataset: {subject} ({split})")
                dataset = load_dataset(
                    self.DATASET_NAME, 
                    subject, 
                    split=split,
                    cache_dir=self.cache_dir
                )
                
                # Convert to our format
                questions = []
                for idx, item in enumerate(dataset):
                    question = self._convert_to_mmlu_question(item, subject, idx)
                    if question:
                        questions.append(question)
                
                # Cache the result
                self._loaded_datasets[cache_key] = questions
                logger.info(f"Loaded {len(questions)} questions for {subject}")
                
            except Exception as e:
                logger.error(f"Failed to load {subject}: {e}")
                return []
        
        # Apply max_questions limit if specified
        if max_questions is not None and max_questions > 0:
            questions = questions[:max_questions]
            logger.info(f"Limited to {len(questions)} questions")
            
        return questions
    
    def load_multiple_subjects(
        self,
        subjects: List[str],
        split: str = "test",
        max_questions_per_subject: Optional[int] = None
    ) -> Dict[str, List[MMLUQuestion]]:
        """
        Load questions for multiple subjects.
        
        Args:
            subjects: List of subject names
            split: Dataset split
            max_questions_per_subject: Max questions per subject
            
        Returns:
            Dictionary mapping subject names to question lists
        """
        results = {}
        
        for subject in subjects:
            questions = self.load_subject(subject, split, max_questions_per_subject)
            if questions:
                results[subject] = questions
            else:
                logger.warning(f"No questions loaded for subject: {subject}")
                
        logger.info(f"Loaded {len(results)} subjects successfully")
        return results
    
    def load_all_subjects(
        self,
        split: str = "test",
        max_questions_per_subject: Optional[int] = None
    ) -> Dict[str, List[MMLUQuestion]]:
        """
        Load all available subjects.
        
        Args:
            split: Dataset split
            max_questions_per_subject: Max questions per subject
            
        Returns:
            Dictionary mapping subject names to question lists
        """
        subjects = self.get_available_subjects()
        return self.load_multiple_subjects(subjects, split, max_questions_per_subject)
    
    def get_subject_iterator(
        self,
        subjects: List[str],
        split: str = "test",
        max_questions_per_subject: Optional[int] = None
    ) -> Iterator[tuple[str, MMLUQuestion]]:
        """
        Get iterator over questions from multiple subjects.
        
        Args:
            subjects: List of subject names
            split: Dataset split
            max_questions_per_subject: Max questions per subject
            
        Yields:
            Tuples of (subject_name, question)
        """
        for subject in subjects:
            questions = self.load_subject(subject, split, max_questions_per_subject)
            for question in questions:
                yield subject, question
    
    def _convert_to_mmlu_question(
        self, 
        item: Dict, 
        subject: str, 
        idx: int
    ) -> Optional[MMLUQuestion]:
        """
        Convert raw dataset item to MMLUQuestion.
        
        Args:
            item: Raw dataset item
            subject: Subject name
            idx: Question index
            
        Returns:
            MMLUQuestion object or None if conversion fails
        """
        try:
            # Extract question text
            question_text = item.get('question', '').strip()
            if not question_text:
                logger.warning(f"Empty question in {subject} at index {idx}")
                return None
            
            # Extract choices - MMLU-Redux format uses 'choices' list, not individual A/B/C/D fields
            raw_choices = item.get('choices', [])
            if not raw_choices or len(raw_choices) != 4:
                logger.warning(f"Invalid choices format in {subject} at index {idx}: {raw_choices}")
                return None
            
            # Clean up choices
            choices = [choice.strip() for choice in raw_choices]
            if any(not choice for choice in choices):
                logger.warning(f"Empty choice found in {subject} at index {idx}")
                return None
            
            # Extract correct answer - should be index (0-3), convert to letter
            correct_answer_idx = item.get('answer')
            if correct_answer_idx is None or correct_answer_idx not in [0, 1, 2, 3]:
                logger.warning(f"Invalid answer index '{correct_answer_idx}' in {subject} at index {idx}")
                return None
            
            # Convert index to letter
            choice_keys = ['A', 'B', 'C', 'D']
            correct_answer = choice_keys[correct_answer_idx]
            
            # Create question ID
            question_id = f"{subject}_{idx:04d}"
            
            return MMLUQuestion(
                question_id=question_id,
                subject=subject,
                question=question_text,
                choices=choices,
                correct_answer=correct_answer,
                raw_data=item
            )
            
        except Exception as e:
            logger.error(f"Failed to convert question {idx} in {subject}: {e}")
            return None
    
    def format_question_for_prompt(
        self, 
        question: MMLUQuestion,
        include_choices: bool = True
    ) -> str:
        """
        Format question for model prompting.
        
        Args:
            question: MMLUQuestion object
            include_choices: Whether to include multiple choice options
            
        Returns:
            Formatted question string
        """
        formatted = question.question
        
        if include_choices and question.choices:
            formatted += "\n\nOptions:"
            for i, choice in enumerate(question.choices):
                letter = chr(ord('A') + i)
                formatted += f"\n{letter}. {choice}"
                
        return formatted
    
    def get_dataset_stats(self) -> Dict[str, int]:
        """
        Get statistics about available datasets.
        
        Returns:
            Dictionary with subject names and question counts
        """
        subjects = self.get_available_subjects()
        stats = {}
        
        for subject in subjects[:5]:  # Sample first 5 for speed
            try:
                questions = self.load_subject(subject, max_questions=None)
                stats[subject] = len(questions)
            except Exception as e:
                logger.warning(f"Failed to get stats for {subject}: {e}")
                stats[subject] = 0
                
        return stats
    
    def validate_subject(self, subject: str) -> bool:
        """
        Validate if subject exists in dataset.
        
        Args:
            subject: Subject name to validate
            
        Returns:
            True if subject exists, False otherwise
        """
        available = self.get_available_subjects()
        return subject in available
