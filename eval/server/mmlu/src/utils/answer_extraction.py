"""
Professional answer extraction utility for multiple-choice evaluations.

This module provides robust extraction of predicted choices (A, B, C, D) from
model-generated text with comprehensive pattern matching and false positive avoidance.
"""

import re
from typing import Optional


class AnswerExtractor:
    """
    Professional answer extractor for multiple-choice questions.
    
    Features:
    - Multiple extraction patterns (boxed, explicit, markdown, etc.)
    - False positive avoidance
    - Robust handling of various response formats
    - Clear extraction priority order
    """
    
    def __init__(self):
        """Initialize answer extractor with compiled patterns."""
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient reuse."""
        # Boxed answer format: \boxed{D}
        self.boxed_pattern = re.compile(
            r'\\boxed\{([A-D])\}',
            re.IGNORECASE
        )
        
        # Explicit answer statements
        self.answer_pattern = re.compile(
            r'\b(?i:Answer|The\s+(?:final\s+|correct\s+)?(?:answer|choice)\s+is|'
            r'My\s+choice\s+is|Select(?:ed)?\s+(?:choice|option)\s*is)\s*[:\-=\s]*'
            r'(?:(?i:Option|Choice)\s+)?\s*[\[\(]?([A-D])[\]\)\.,:]?(?=\s|$|[^\w])',
            re.VERBOSE | re.IGNORECASE
        )
        
        # Markdown answer format: **Answer: D**
        self.markdown_answer_pattern = re.compile(
            r'\*\*(?:Answer|Selected)\s*[:\-=\s]*(?:(?i:Option|Choice)\s+)?([A-D])\*\*',
            re.IGNORECASE
        )
        
        # Markdown option format: **Option D**
        self.markdown_option_pattern = re.compile(
            r'\*\*(?:Option|Choice)\s*(?:[:\-=\s]+\s*)?(?P<choice_letter>[A-D])\*\*'
            r'(?!\s+(?:is|are|was|were|details|describes|concerns|relates\s+to|'
            r'refers\s+to|means|entails|states|suggests|provides|offers|'
            r'talks\sabout|covers|deals\swith|seems|appears|might\sbe|'
            r'could\sbe|would\sbe|has|contains|involves|represents)\b)',
            re.VERBOSE | re.IGNORECASE
        )
        
        # Explicit option format: Option: D
        self.explicit_option_pattern = re.compile(
            r'\b(?i:Option|Choice)\s*(?:[:\-=\s])+\s*\s*[\[\(]?([A-D])[\]\)\.,:]?'
            r'(?=\s|$|[^\w])(?!\s+(?:is|are|was|were|details|describes|concerns|'
            r'relates\s+to|refers\s+to|means|entails|states|suggests|provides|'
            r'offers|talks\sabout|covers|deals\swith|seems|appears|might\sbe|'
            r'could\sbe|would\sbe|has|contains|involves|represents)\b)',
            re.VERBOSE | re.IGNORECASE
        )
        
        # End-of-text choice pattern
        self.end_choice_pattern = re.compile(
            r'(?:(?i:is|was|be|denotes|represents|therefore|hence|thus|so|'
            r'the\sresult\sis|the\sanswer\swould\sbe)\s*[:\-=\s]*)?'
            r'\s*[\[\(\{]?([A-D])[\]\)\}\.,:]?\W*$',
            re.VERBOSE | re.IGNORECASE
        )
        
        # Isolated choice pattern for very short responses
        self.isolated_choice_pattern = re.compile(
            r'^\s*[\[\(\{]?([A-D])[\]\)\}\.,:]?\s*$'
        )
    
    def extract_choice(self, generated_text: str) -> str:
        """
        Extract predicted choice from generated text.
        
        Args:
            generated_text: Model-generated response text
            
        Returns:
            Extracted choice ('A', 'B', 'C', 'D') or 'Invalid' if no valid choice found
        """
        if not generated_text:
            return "Invalid"
            
        text = generated_text.strip()
        
        # Priority 1: Boxed format (highest confidence)
        choice = self._extract_boxed_answer(text)
        if choice:
            return choice
            
        # Priority 2: Explicit answer statements
        choice = self._extract_explicit_answer(text)
        if choice:
            return choice
            
        # Priority 3: Markdown answer format
        choice = self._extract_markdown_answer(text)
        if choice:
            return choice
            
        # Priority 4: Markdown option format
        choice = self._extract_markdown_option(text)
        if choice:
            return choice
            
        # Priority 5: Explicit option format
        choice = self._extract_explicit_option(text)
        if choice:
            return choice
            
        # Priority 6: Isolated single letter (for very short responses)
        choice = self._extract_isolated_choice(text)
        if choice:
            return choice
            
        # Priority 7: Direct letter match (single letter responses)
        if text in ['A', 'B', 'C', 'D']:
            return text
            
        # Priority 8: End-of-text pattern (lowest confidence)
        choice = self._extract_end_choice(text)
        if choice:
            return choice
            
        return "Invalid"
    
    def _extract_boxed_answer(self, text: str) -> Optional[str]:
        """Extract from boxed format: \\boxed{D}"""
        matches = self.boxed_pattern.findall(text)
        return matches[-1].upper() if matches else None
    
    def _extract_explicit_answer(self, text: str) -> Optional[str]:
        """Extract from explicit answer statements."""
        matches = self.answer_pattern.findall(text)
        return matches[-1].upper() if matches else None
    
    def _extract_markdown_answer(self, text: str) -> Optional[str]:
        """Extract from markdown answer format."""
        matches = self.markdown_answer_pattern.findall(text)
        return matches[-1].upper() if matches else None
    
    def _extract_markdown_option(self, text: str) -> Optional[str]:
        """Extract from markdown option format."""
        matches = self.markdown_option_pattern.finditer(text)
        found_options = [match.group('choice_letter') for match in matches]
        return found_options[-1].upper() if found_options else None
    
    def _extract_explicit_option(self, text: str) -> Optional[str]:
        """Extract from explicit option format."""
        matches = self.explicit_option_pattern.findall(text)
        return matches[-1].upper() if matches else None
    
    def _extract_isolated_choice(self, text: str) -> Optional[str]:
        """Extract from very short, isolated responses."""
        if len(text) <= 5:
            match = self.isolated_choice_pattern.match(text)
            return match.group(1).upper() if match else None
        return None
    
    def _extract_end_choice(self, text: str) -> Optional[str]:
        """Extract from end-of-text patterns."""
        match = self.end_choice_pattern.search(text)
        if match:
            # Additional validation to avoid false positives
            full_match_start = match.start(0)
            if full_match_start == 0 or not text[full_match_start - 1].isalpha():
                return match.group(1).upper()
        return None
    
    def get_extraction_confidence(self, generated_text: str) -> tuple[str, str]:
        """
        Extract choice with confidence level.
        
        Args:
            generated_text: Model-generated response text
            
        Returns:
            Tuple of (choice, confidence_level)
            confidence_level: 'high', 'medium', 'low', 'invalid'
        """
        if not generated_text:
            return "Invalid", "invalid"
            
        text = generated_text.strip()
        
        # High confidence patterns
        for pattern_func, confidence in [
            (self._extract_boxed_answer, "high"),
            (self._extract_explicit_answer, "high"),
            (self._extract_markdown_answer, "medium"),
            (self._extract_markdown_option, "medium"),
            (self._extract_explicit_option, "medium"),
        ]:
            choice = pattern_func(text)
            if choice:
                return choice, confidence
        
        # Low confidence patterns
        choice = self._extract_isolated_choice(text)
        if choice:
            return choice, "low"
            
        if text in ['A', 'B', 'C', 'D']:
            return text, "low"
            
        choice = self._extract_end_choice(text)
        if choice:
            return choice, "low"
            
        return "Invalid", "invalid"
    
    def validate_response_quality(self, generated_text: str) -> dict:
        """
        Analyze response quality and extraction reliability.
        
        Args:
            generated_text: Model-generated response text
            
        Returns:
            Dictionary with quality metrics
        """
        choice, confidence = self.get_extraction_confidence(generated_text)
        
        return {
            "extracted_choice": choice,
            "confidence_level": confidence,
            "response_length": len(generated_text),
            "is_valid": choice != "Invalid",
            "has_reasoning": len(generated_text) > 20,  # Simple heuristic
            "multiple_choices_mentioned": len(re.findall(r'\b[A-D]\b', generated_text)) > 1
        }
