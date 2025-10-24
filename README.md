
# Edge Reasoning LLM Evaluation Framework

## Quick Setup

```bash
make venv                    # Create virtual environment
source .venv/bin/activate    # Activate environment  
make setup                   # Auto-detect platform and install dependencies
```

`make setup` will require `vllm==v0.10.0` and sometimes it shows no matching version can be found. 
We can build vllm 0.10.0 from source by the following commands
```bash
git clone vllm
git checkout v0.10.0
export VLLM_COMMIT=6d8d0a24c02bfd84d46b3016b865a44f048ae84b 
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/a2/98/a9c7fd61e27fd766f3cb7022613b214f053836ba7749d12a4cd5ab4644fc/vllm-0.10.0-cp38-abi3-manylinux1_x86_64.whl
uv pip install -e .
uv pip install -e ".[server]"

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
