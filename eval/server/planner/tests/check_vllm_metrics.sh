#!/bin/bash

# Simple runner for VLLM timing tests
# Tests various methods to extract precise timing metrics

set -e

echo "=== VLLM Timing Metrics Test ==="
echo "Testing methods to extract TTFT and generation metrics"
echo ""

# Check if we're in the right directory
if [ ! -f "test_timing.py" ]; then
    echo "Error: test_timing.py not found. Run from tests/ directory."
    exit 1
fi

# Set up environment
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# Check for required packages
echo "Checking dependencies..."
python -c "import vllm; print(f'VLLM version: {vllm.__version__}')" || {
    echo "Error: VLLM not installed. Install with: pip install vllm"
    exit 1
}

# Run the main test
echo "Running timing tests..."
python test_timing.py

echo ""
echo "=== Manual Async Test ==="
echo "To test async streaming timing, run:"
echo "python -c \"import asyncio; from test_timing import test_async_vllm_timing; asyncio.run(test_async_vllm_timing())\""