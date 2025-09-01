#!/usr/bin/env python3
"""
utilities for mmlu evaluations
"""

import sys
import os
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[3]  
sys.path.insert(0, str(project_root))

from loaders.results import get_results_config

def main():
    results_config = get_results_config()
    results_dir = results_config.get_result_base_dir('mmlu', 'tegra')
    print(str(results_dir))

if __name__ == "__main__":
    main()
