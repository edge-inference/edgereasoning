# Server MMLU Evaluation

Server-side MMLU benchmark evaluation module.

## Project Structure

```
eval/server/mmlu/
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
├── scripts/
│   ├── base.py
│   ├── budget.py
│   └── noreasoning.py
└── run.sh
```

## Installation

Dependencies are managed via the root `setup.py`. From repository root:

```bash
make venv
source .venv/bin/activate  
make setup
```

## Usage

### Basic Usage

```bash
# Run default base evaluation
./run.sh

# Run specific modes  
./run.sh base
./run.sh budget
./run.sh noreasoning
./run.sh scale

# With custom model
./run.sh base --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# With token limit
./run.sh budget --max-tokens 256
```

### Via Makefile (Recommended)

From repository root:
```bash
make server-mmlu    
```

## Evaluation Modes

* **Base**: Full reasoning with 4096 tokens
* **Budget**: Configurable token limits (128-512)  
* **NoReasoning**: Direct answer selection with 4096 tokens
* **Scale**: Parameter scaling experiments

## Output Files

Results are saved to `data/mmlu/server/` with timestamp directories.

### Summary Results
JSON files with overall accuracy and model configuration.

### Performance Metrics
CSV files with per-question results, timing, and token counts.

### Telemetry Data  
System performance and energy consumption logs.

## Supported Models

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- `Custom models via command line`
