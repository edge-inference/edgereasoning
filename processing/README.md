# Results Processor

A comprehensive results processing suite for MMLU evaluation data. Consolidates CSV files, generates performance analysis, and creates detailed reports.

## Features

- **🔄 Data Consolidation**: Automatically discovers and consolidates detailed CSV results from multiple models and subjects
- **📊 Multi-format Output**: Generates consolidated CSVs, multi-sheet Excel files, and performance summaries
- **📈 Performance Analysis**: Statistical analysis, rankings, correlations, and outlier detection
- **📋 Comprehensive Reports**: Executive summaries, detailed technical reports, and markdown documentation
- **🎯 Visualization**: Automated generation of comparison plots, heatmaps, and correlation matrices

## Quick Start

### Basic Usage

```bash
# Process all results in the default ./results directory
python process_results.py

# Process with custom directories
python process_results.py --results-dir ./my_results --output-dir ./my_analysis
```

### Advanced Usage

```bash
# Consolidate only (no analysis)
python process_results.py --consolidate-only

# Verbose output with detailed logging
python process_results.py --verbose

# Skip plot generation for faster processing
python process_results.py --no-plots
```

## Expected Input Structure

The processor expects results organized as follows:

```
results/
├── base_all_subjects_TIMESTAMP_MODEL1/
│   ├── summary.json
│   ├── anatomy/
│   │   └── detailed_results_base_anatomy_TIMESTAMP.csv
│   ├── physics/
│   │   └── detailed_results_base_physics_TIMESTAMP.csv
│   └── ...
├── base_all_subjects_TIMESTAMP_MODEL2/
│   └── ...
└── ...
```

## Generated Outputs

### Consolidated Data
- `all_results_consolidated_TIMESTAMP.csv` - All questions from all models in one file
- `all_results_by_model_TIMESTAMP.xlsx` - Multi-sheet Excel with one sheet per model
- `performance_summary_TIMESTAMP.csv` - Performance metrics summary

### Analysis
- `accuracy_rankings_TIMESTAMP.csv` - Model accuracy rankings
- `performance_rankings_TIMESTAMP.csv` - Speed and efficiency rankings  
- `subject_analysis_TIMESTAMP.csv` - Subject difficulty analysis
- `correlation_matrix_TIMESTAMP.csv` - Metric correlations

### Visualizations
- `accuracy_comparison_TIMESTAMP.png` - Model accuracy bar chart
- `performance_heatmap_TIMESTAMP.png` - Model vs subject performance heatmap
- `correlation_matrix_TIMESTAMP.png` - Performance metrics correlation plot
- `timing_comparison_TIMESTAMP.png` - Timing metrics comparison

### Reports
- `executive_summary_TIMESTAMP.md` - Executive summary in Markdown
- `executive_summary_TIMESTAMP.json` - Executive summary data
- `detailed_report_TIMESTAMP.json` - Comprehensive technical report

## CSV Structure

The processor expects detailed results CSVs with the following columns:

```csv
question_id,subject,question,choices,correct_answer,predicted_choice,is_correct,generated_text,ttft,decode_time,total_time_ms,tokens_per_second,input_tokens,output_tokens,generated_text_length
```

## Performance Metrics

The processor analyzes these key performance metrics:

- **Accuracy**: Overall and subject-specific accuracy
- **TTFT**: Time to First Token (ms)
- **Decode Time**: Time for generation after first token (ms)
- **Total Time**: End-to-end inference time (ms)
- **Tokens/Second**: Generation speed
- **Token Counts**: Input and output token statistics

## Installation Requirements

```bash
pip install pandas numpy matplotlib seaborn scipy jinja2 openpyxl
```

## Usage Examples

### Example 1: Standard Processing
```bash
python process_results.py --results-dir ./results --output-dir ./analysis --verbose
```

### Example 2: Quick Consolidation
```bash
python process_results.py --consolidate-only --output-dir ./consolidated
```

### Example 3: Custom Analysis
```python
from processor import ResultConsolidator, PerformanceAnalyzer

# Load and consolidate results
consolidator = ResultConsolidator("./results")
consolidated = consolidator.consolidate_all_results()

# Generate analysis
analyzer = PerformanceAnalyzer(consolidated)
rankings = analyzer.get_accuracy_rankings()
print(rankings)
```

## Output File Examples

### Consolidated CSV Structure
```csv
model_name,model_timestamp,config_name,question_id,subject,question,choices,correct_answer,predicted_choice,is_correct,generated_text,ttft,decode_time,total_time_ms,tokens_per_second,input_tokens,output_tokens,generated_text_length
DeepSeek-R1-Distill-Qwen-14B,2025-06-22T02:32:02,base,0,anatomy,Which of the following is a disorder...,['Dyslexia',...],D,D,True,...,327.155,1325.473,1652.830,139.760,127,231,1155
```

### Performance Summary Structure  
```csv
model_name,metric_type,subject,accuracy,total_questions,correct_answers,avg_ttft_ms,avg_decode_time_ms,avg_total_time_ms,avg_tokens_per_second,total_input_tokens,total_output_tokens,timestamp
DeepSeek-R1-Distill-Qwen-14B,overall,ALL,0.742,3000,2226,89.45,1456.23,1545.68,172.34,127450,231560,2025-06-22T02:32:02
```

## API Reference

### ResultConsolidator
- `discover_model_directories()` - Find all model result directories
- `consolidate_all_results()` - Load and consolidate all results
- `create_consolidated_csv()` - Generate single consolidated CSV
- `create_multi_sheet_excel()` - Generate Excel with sheets per model

### PerformanceAnalyzer  
- `get_accuracy_rankings()` - Model accuracy rankings
- `get_performance_rankings()` - Speed/efficiency rankings
- `get_subject_analysis()` - Subject difficulty analysis
- `analyze_correlations()` - Metric correlation analysis
- `detect_outliers()` - Statistical outlier detection

### ReportGenerator
- `generate_executive_summary()` - High-level summary
- `generate_detailed_report()` - Comprehensive technical report
- `save_markdown_report()` - Markdown format report
- `save_json_report()` - JSON format report

## Troubleshooting

### Common Issues

**No models found**: Check that your results directory follows the expected structure with `base_all_subjects_*` directories.

**Missing CSV files**: Ensure each subject directory contains `detailed_results_*.csv` files.

**Import errors**: Install required dependencies: `pip install pandas numpy matplotlib seaborn scipy jinja2 openpyxl`

**Memory issues**: For large datasets, use `--consolidate-only` first, then process in smaller batches.

### Logging

Enable verbose logging to debug issues:
```bash
python process_results.py --verbose
```

Log files are automatically generated as `processor_TIMESTAMP.log`.
