#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Resume Natural-Plan evaluation from partial results.

Usage:
    python bench/tools/resume_eval.py --partial results/partial_np_meeting_14b_20250730_154624.json
    python bench/tools/resume_eval.py --run-name np_meeting_14b_20250730_154624
"""
import argparse
import json
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Resume Natural-Plan evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--partial", help="Path to partial_*.json file")
    group.add_argument("--run-name", help="Run name to find partial results for")
    parser.add_argument("--results-dir", default="./results", help="Results directory")
    
    args = parser.parse_args()
    
    if args.partial:
        partial_file = Path(args.partial)
    else:
        partial_file = Path(args.results_dir) / f"partial_{args.run_name}.json"
    
    if not partial_file.exists():
        print(f"ERROR: Partial results file not found: {partial_file}")
        sys.exit(1)
    
    with open(partial_file) as f:
        data = json.load(f)
    
    question_results = data.get("question_results", [])
    completed_count = data.get("completed_count", 0)
    timestamp = data.get("timestamp", "unknown")
    
    print(f"=== Partial Results Summary ===")
    print(f"File: {partial_file}")
    print(f"Timestamp: {timestamp}")
    print(f"Completed: {completed_count} questions")
    print(f"Total results: {len(question_results)}")
    
    if question_results:
        correct_count = sum(1 for r in question_results if r.get("is_correct", False))
        print(f"Accuracy: {correct_count}/{completed_count} ({correct_count/completed_count*100:.1f}%)")
        
        # Show some sample results
        print(f"\nSample results:")
        for i, result in enumerate(question_results[:3]):
            status = "✓" if result.get("is_correct") else "✗"
            print(f"  {status} {result['question_id']}: {result['generated_text'][:100]}...")
        
        if len(question_results) > 3:
            print(f"  ... and {len(question_results) - 3} more")
    
    # Check if we can convert to final format
    output_file = partial_file.parent / f"results_{partial_file.stem.replace('partial_', '')}.json"
    
    if not output_file.exists():
        print(f"\nTo use these results with planner_predictions.py:")
        print(f"  python bench/tools/planner_predictions.py \\")
        print(f"    --results {output_file} \\")
        print(f"    --dataset eval/data/meeting_planning.json \\")
        print(f"    --out meeting_planning_with_preds.json")
        print(f"\nFirst, create the results file manually or restart the evaluation to complete it.")
    else:
        print(f"\nResults file exists: {output_file}")
        print(f"You can use planner_predictions.py to merge these results.")

if __name__ == "__main__":
    main()