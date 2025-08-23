"""
Test the answer extraction utility with various response formats.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.answer_extraction import AnswerExtractor


def test_answer_extraction():
    """Test answer extraction with various formats."""
    extractor = AnswerExtractor()
    
    test_cases = [
        # Boxed format
        ("The answer is \\boxed{D}", "D", "high"),
        
        # Explicit answer
        ("The correct answer is C.", "C", "high"),
        ("My choice is B", "B", "high"),
        
        # Markdown format
        ("**Answer: A**", "A", "medium"),
        ("**Option D**", "D", "medium"),
        
        # Option format
        ("Option: B", "B", "medium"),
        ("Choice C", "C", "medium"),
        
        # Simple responses
        ("A", "A", "low"),
        ("(B)", "B", "low"),
        
        # End pattern
        ("Therefore the answer would be D.", "D", "low"),
        
        # Invalid cases
        ("I don't know", "Invalid", "invalid"),
        ("", "Invalid", "invalid"),
    ]
    
    print("Testing Answer Extraction Utility")
    print("=" * 50)
    
    passed = 0
    total = len(test_cases)
    
    for test_input, expected_choice, expected_confidence in test_cases:
        choice, confidence = extractor.get_extraction_confidence(test_input)
        
        status = "✓" if choice == expected_choice and confidence == expected_confidence else "✗"
        print(f"{status} Input: '{test_input[:30]}{'...' if len(test_input) > 30 else ''}'")
        print(f"   Expected: {expected_choice} ({expected_confidence})")
        print(f"   Got: {choice} ({confidence})")
        
        if choice == expected_choice and confidence == expected_confidence:
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = test_answer_extraction()
    exit(0 if success else 1)
