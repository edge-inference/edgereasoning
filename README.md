
# Edge Reasoning LLM Evaluation Framework

## Quick Setup

```bash
make venv                    # Create virtual environment
source .venv/bin/activate    # Activate environment  
make setup                   # Auto-detect platform and install dependencies
```

## Usage

**Platform Detection:**
```bash
make platform               # Show detected platform
make info                   # Show device information
```

**Server Evaluations:**
```bash
make server-mmlu            # MMLU benchmark
make planner                # Planner evaluation
```

**Tegra Evaluations:**
```bash
make tegra-base             # Base MMLU evaluation
make tegra-budget           # Budget evaluation
make tegra-synthetic        # Synthetic benchmarks
```

**Individual Benchmarks:**
```bash
make prefill                # Prefill experiments (Tegra)
make decode                 # Decode experiments (Tegra)
```

**Results:** All outputs saved to `data/` directory with timestamp subdirectories.

**Help:** `make help` for all available targets.

## Analytical Models

```bash
python extract_io_length.py    # Extract I/O length data
python plot_sweep.py           # Generate sweep plots
python interactive.py          # Interactive analysis
python latency_model.py        # Latency modeling
```

## Missing packages after setup? Try manual install with your .venv activated and in project root.

```bash
pip install -r requirements.txt
```
