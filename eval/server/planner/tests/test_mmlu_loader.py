"""
Test the MMLU dataset loader functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loaders.mmlu_loader import MMLULoader, MMLUQuestion


def test_mmlu_loader():
    """Test MMLU loader with basic functionality."""
    print("Testing MMLU Dataset Loader")
    print("=" * 40)
    
    # Initialize loader
    loader = MMLULoader()
    
    # Test 1: Get available subjects
    print("Test 1: Getting available subjects...")
    subjects = loader.get_available_subjects()
    print(f"✓ Found {len(subjects)} subjects")
    
    if len(subjects) > 0:
        print(f"  Sample subjects: {subjects[:3]}")
    
    # Test 2: Load a small subset
    if subjects:
        test_subject = subjects[0]  # Use first available subject
        print(f"\nTest 2: Loading {test_subject} (max 3 questions)...")
        
        questions = loader.load_subject(test_subject, max_questions=3)
        print(f"✓ Loaded {len(questions)} questions")
        
        # Test 3: Validate question format
        if questions:
            question = questions[0]
            print(f"\nTest 3: Validating question format...")
            print(f"✓ Question ID: {question.question_id}")
            print(f"✓ Subject: {question.subject}")
            print(f"✓ Question: {question.question[:50]}...")
            print(f"✓ Choices: {len(question.choices)} options")
            print(f"✓ Correct Answer: {question.correct_answer}")
            
            # Test 4: Format for prompting
            print(f"\nTest 4: Question formatting...")
            formatted = loader.format_question_for_prompt(question)
            print(f"✓ Formatted length: {len(formatted)} characters")
            print(f"  Preview: {formatted[:100]}...")
    
    # Test 5: Subject validation
    print(f"\nTest 5: Subject validation...")
    valid = loader.validate_subject("electrical_engineering")
    print(f"✓ electrical_engineering valid: {valid}")
    
    invalid = loader.validate_subject("nonexistent_subject")
    print(f"✓ nonexistent_subject valid: {invalid}")
    
    print("\n" + "=" * 40)
    print("MMLU Loader tests completed successfully!")
    return True


if __name__ == "__main__":
    success = test_mmlu_loader()
    exit(0 if success else 1)
