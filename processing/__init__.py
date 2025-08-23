"""
Results Processor Module

This module provides comprehensive processing and analysis tools for MMLU evaluation results.
Handles CSV consolidation, performance analysis, and reporting for multiple models and subjects.
"""

from .result_consolidator import ResultConsolidator
from .performance_analyzer import PerformanceAnalyzer
from .report_generator import ReportGenerator
from .data_models import *

__all__ = [
    'ResultConsolidator',
    'PerformanceAnalyzer', 
    'ReportGenerator',
    'ModelResult',
    'SubjectResult',
    'ConsolidatedResult'
]
