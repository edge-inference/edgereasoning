# VLLM MMLU Benchmark

A comprehensive MMLU evaluation system edge systems with perf and energy instrumentation.

## 🚀 Features

- **VLLM Integration**: High-performance inference with comprehensive metrics
- **Parameter Sweeping**: Automated model and token limit sweeps
- **Three Evaluation Modes**: Base (reasoning), Budget (efficient), NoReasoning (direct)
- **Telemetry Monitoring**: Real-time performance and energy tracking
- **Answer Extraction**: Multi-pattern choice extraction with confidence scoring
- **Configuration-Driven**: YAML-based evaluation configurations

## 📁 Project Structure

```
vllm-mmlu-benchmark/
├── src/
│   ├── models/              
│   │   └── vllm_model.py    
│   ├── evaluators/          
│   │   ├── base_evaluator.py        
│   │   ├── budget_evaluator.py      
│   │   └── noreasoning_evaluator.py 
│   ├── telemetry/           
│   │   └── monitor.py       
│   ├── data_loaders/            
│   │   └── mmlu_loader.py   
│   └── utils/               
│       └── answer_extraction.py 
├── configs/                 
│   ├── base.yaml           
│   ├── budget.yaml         
│   └── noreasoning.yaml                   
├── budget.py           
├── budget.sh         
└── README.md              
```

## 🛠️ Installation

```bash
pip install vllm transformers datasets pyyaml tqdm
```

## 📊 Usage

### Parameter Sweeping

```bash
# Run automated sweep across models and token limits
./budget.sh

# Single evaluation
python3 scripts/budget.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --max-tokens 256
```

### Configuration

Edit `budget.sh` to customize:

```bash
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

MAX_TOKENS_VALUES=(128 256 512)
```

## ⚙️ Evaluation Modes

**Base**: Full reasoning with 4096 tokens
**Budget**: Efficient evaluation with configurable token limits (128-512)
**NoReasoning**: Direct answer selection with 4096 tokens

## 📈 Output Files

### Summary Results (`all_subjects_summary.json`)
```json
{
  "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
  "overall_accuracy": 0.319,
  "total_questions": 3000,
  "total_correct": 959,
  "config_details": {
    "model_settings": {
      "max_tokens": 128,
      "temperature": 0.6
    }
  }
}
```

### Performance Metrics (`detailed_results_*.csv`)
Per-question results with timing, token counts, and accuracy data.

### Telemetry Data (`tegrastats_*.log`, `energy_*.csv`)
System performance and energy consumption metrics.

## 🔬 Supported Models

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `l3lab/L1-Qwen-1.5B-Max`
- Custom models via command line

## � Architecture

- **VLLM Integration**: Native API with performance monitoring
- **Modular Design**: Pluggable evaluators and configurations  
- **Telemetry System**: Real-time energy and performance tracking
- **Answer Extraction**: Multi-pattern matching with confidence scoring

---

**Test-Time Scaling Test**

# Scale Benchmark for Jetson Orin

This directory contains a test-time scaling mmlu benchmark for Jetson Orin single GPU setup. The benchmark evaluates multiple models, seeds, and sample sizes.

## Overview

- **Platform**: Single GPU Jetson Orin  
- **Models**: 3 models (configurable in `tt_scaling.sh`)
- **Seeds**: 3 seeds for reproducibility (42, 1337, 2023)
- **Questions**: 5 questions per MMLU subject (150 total questions max)
- **Sample Sizes**: 1, 2, 4, 8, 16, 32 samples per question
- **Token Budget**: 256 tokens

## Key Features

- **Test-time Scaling**: Multiple samples per question with majority voting
- **Memory Optimization**: Aggressive cleanup between runs for Jetson constraints
- **Resume Support**: Can resume from specific model/seed/sample size
- **Telemetry**: Comprehensive performance monitoring and energy tracking
- **Validation**: Built-in error handling and memory management

## Files

- `tt_scaling.py` - Main scale evaluation script (Jetson optimized)
- `tt_scaling.sh` - test time scaling sweep script for all combinations
- `configs/scale.yaml` - Scale evaluation configuration
- `src/evaluators/scale_evaluator.py` - Scale evaluator implementation
- `test_scale_setup.py` - Setup validation script

## Quick Start

1. **Test Setup**:
   ```bash
   python test_scale_setup.py
   ```

2. **Run Single Evaluation**:
   ```bash
   python scripts/tt_scaling.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --num-samples 4 --token-budget 256 --max-subjects 5
   ```

3. **Run Full Sweep**:
   ```bash
   ./tt_scaling.sh
   ```

## Configuration

### Models (in `tt_scaling.sh`)
```bash
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
)
```

### Jetson Optimizations
- GPU memory utilization: 65% (reduced from 75%)
- Max samples: 32 (reduced from 64)
- Questions per subject: 5 (for fast evaluation)
- Max subjects: 30 (configurable)
- Single token budget: 256 (simplified from 3 budgets)
- Aggressive memory cleanup between runs

```

## Output

Results are saved in `./results/scale_jetson_<timestamp>_<model>/`:
- `summary.json` - Overall metrics and scaling analysis
- `<subject>/` - Per-subject detailed results
- CSV files with question-level data
- Telemetry logs with performance metrics

## Scaling Metrics

The benchmark tracks:
- **Samples Generated**: Total samples across all questions
- **Voting Confidence**: Average confidence in majority vote decisions  
- **Scaling Efficiency**: Accuracy improvement vs computational cost
- **Performance**: Tokens/second, memory usage, energy consumption

## Memory Management

Jetson-specific optimizations:
- Force garbage collection after each subject
- Clear CUDA cache between runs
- Reset peak memory stats
- Cleanup model on failures

## Expected Runtime

For full sweep (3 models × 3 seeds × 1 token budget × 6 sample sizes):
- **Total runs**: 54 evaluations
- **Questions per run**: ~150 (5 per subject × 30 subjects)
- **Estimated time**: 3-5 hours (depending on model sizes)

## Troubleshooting

1. **Out of Memory**: Reduce `max_subjects` or `gpu_memory_utilization`
2. **vLLM Errors**: Check model compatibility and memory requirements
3. **Dataset Issues**: Verify internet connection for MMLU-Redux download
4. **Resume Issues**: Check checkpoint file and logs for last successful run

## Integration

This benchmark integrates with the existing evaluation framework:
- Uses same `VLLMModel` interface as budget/base evaluators
- Compatible with telemetry monitoring system
- Follows same CSV output format
- Uses same answer extraction utilities 
