# Results Processor

Consolidates MMLU evaluation CSV files and generates performance analysis using `files/results.yaml` configuration.

## Usage

```bash
# Process MMLU server results using files/results.yaml
python postprocess.py --sub-config server

# Process synthetic data (prefill/decode/scaling)
python postprocess.py --sub-config prefill
python postprocess.py --sub-config decode

# Custom directories (overrides config)
python postprocess.py --results-dir ./my_results --output-dir ./analysis

# Consolidate only (no analysis)
python postprocess.py --sub-config server --consolidate-only

# With analysis and reports
python postprocess.py --sub-config server --analysis
```

## Input Structure

typical mmlu evaluation results looks like one below

```
results/
├── TIMESTAMP_base_MODEL/
│   ├── summary.json
│   ├── anatomy/
│   │   └── detailed_results_base_anatomy_TIMESTAMP.csv
│   └── physics/
│       └── detailed_results_base_physics_TIMESTAMP.csv
├── TIMESTAMP_scale_MODEL_64samples_512tokens/
└── TIMESTAMP_budget_MODEL_256tok/
```

## Outputs

- `all_results_consolidated_TIMESTAMP.csv` - All results in one file
- `all_results_by_model_TIMESTAMP.xlsx` - Excel with sheets per model
- `performance_summary_TIMESTAMP.csv` - Performance metrics

## Requirements

```bash
pandas numpy matplotlib seaborn openpyxl
```