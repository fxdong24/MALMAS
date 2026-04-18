# MALMAS: Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation

This repository contains the implementation of **MALMAS** (Memory-Augmented LLM-based Multi-Agent System), a novel approach for automated feature engineering on tabular data using multiple LLM-powered agents with a memory mechanism.

## Overview

MALMAS addresses the challenge of automated feature generation by:

- Employing **six specialized agents**, each focusing on distinct feature transformation strategies
- Introducing a **Router Agent** that dynamically selects active agent subsets per iteration
- Implementing a **multi-level memory system** (procedural, feedback, conceptual) to guide feature generation across rounds
- Leveraging **LLM-generated conceptual summaries** to capture effective feature patterns

## Project Structure

```
MALMAS-63DB/
├── main_demo/                    # Core MALMAS implementation
│   ├── pipeline.py              # Main pipeline with Router integration
│   ├── router.py                # Router Agent for dynamic agent selection
│   ├── memory.py                # Agent memory mechanism
│   ├── main_func.py             # Core functions (evaluation, LLM calls)
│   ├── model_factory.py         # ML model factory (XGBoost, LightGBM, etc.)
│   ├── ablations.py             # Ablation study implementations
│   └── path_helper.py           # Path utilities
│
├── baselines/                   # Baseline method implementations
│   ├── baseline_func.py         # DFS, AutoFeat, OpenFE, CAAFE, OCTree, LLMFE
│   ├── utils_xg.py              # OCTree utilities
│   └── LLMFE_demo/              # LLM-FE baseline implementation
│
├── prompt_files/                # Agent prompt templates
│   ├── unaryfeature.txt         # UnaryFeatureAgent
│   ├── crosscompositional.txt   # CrossCompositionalAgent
│   ├── aggregationconstruct.txt # AggregationConstructAgent
│   ├── temporalfeature.txt      # TemporalFeatureAgent
│   ├── localtransform.txt       # LocalTransformAgent
│   ├── localpattern.txt         # LocalPatternAgent
│   └── codegeneration.txt       # Code generation prompt
│
├── data_file/                   # Dataset loaders and metadata
│   ├── Adult_Census_Income/
│   ├── Titanic/
│   ├── Bank_Marketing/
│   └── ... (21 datasets total)
│
├── global_config.py             # Global configuration (LLM settings, model params)
├── README.md                    # This file
├── web_app/                     # Web interface
│   ├── server.py               # FastAPI backend server
│   └── static/index.html       # Web frontend
│
└── *.ipynb                      # Experiment notebooks (21 datasets)
```

## Installation

### Requirements

- Python 3.8+
- OpenAI API key or compatible LLM service (DeepSeek, Qwen, etc.)

### LLM Configuration

Edit `global_config.py` to configure your LLM API:

```python
LLM = {
    "code_temp": 0.2,
    "llm_model": "deepseek-chat",  # or "gpt-4o-mini", "qwen-max"
    "api_key": "your-api-key-here",
    "base_url": "https://api.deepseek.com",  # or your provider's URL
}
```

**Security Note**: For production use, consider using environment variables instead of hardcoding credentials:

```python
import os
LLM = {
    "llm_model": "deepseek-chat",
    "api_key": os.getenv("LLM_API_KEY"),
    "base_url": os.getenv("LLM_BASE_URL"),
}
```

## Quick Start

### Run a Single Experiment

```python
import asyncio
from data_file.Titanic import Titanic as read_data
from main_demo.pipeline import MALMAS_random_experiments_async
import global_config

# Configure task
prompt_path_list = global_config.prompt_path_list
task_name = "Titanic"
task = "classification"  # or "regression"

# Run MALMAS
result = await MALMAS_random_experiments_async(
    task_name=task_name,
    task=task,
    read_data_class=read_data,
    model_name="xgboost",
    metric="auc",  # or "acc" for classification, "nrmse" for regression
    prompt_path_list=prompt_path_list,
    Nround=4,                      # Number of rounds
    min_effective=2,               # Min effective features for conceptual summary
    long_memory_feature_num=4,     # Top features to persist
    temp=1.0,                      # LLM temperature
    verbose=True
)

print(result)
```

### Web Interface

A simple web interface is provided for easy experiment running:

```bash
cd web_app
python server.py
```

Then open `http://localhost:8000` in your browser.

**Features:**

- Upload CSV file directly
- Auto-detect columns and target column
- Configure experiment parameters (task type, model, metric, rounds)
- Configure LLM settings (model, API key, base URL)
- Real-time experiment logs
- Results display after completion

### Router Configuration

The Router dynamically selects active agent subsets per round. By default, it decides based on dataset characteristics and agent performance without constraints.

```python
# Limit maximum agents per round
await MALMAS_random_experiments_async(
    ...,
    router_max_agents=3
)

# Set both min and max
await MALMAS_random_experiments_async(
    ...,
    router_min_agents=2,
    router_max_agents=4
)
```

## Datasets

The repository includes 21 tabular datasets:

**Classification (15)**:

- Adult Census Income, Bank Marketing, Banknote Authentication
- Breast Cancer Wisconsin, Car Evaluation, Credit Approval
- Heart Disease, Jungle Chess, Pima Indians Diabetes
- Student Score, Telco Customer Churn, Titanic
- Wine Quality, Balance Scale

**Regression (6)**:

- Airfoil Self Noise, Bike Sharing, Crab Age Prediction
- Energy Efficiency (Y1), House Price, Insurance
- Medical Cost Personal

Each dataset includes:

- Data loader class (`{Dataset}.py`)
- Metadata description (`parsed_description.json`, `enriched_description.json`)
- Task description (`taskdescription.txt`)

## Running Baselines

```python
from baselines.baseline_func import run_baseline_experiments

# Run all baselines (DFS, AutoFeat, OpenFE, CAAFE, OCTree, LLMFE)
result = run_baseline_experiments(
    task_name="Titanic",
    task="classification",
    read_data_class=read_data,
    model_name="xgboost",
    metric="auc"
)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{malmas2024,
  title={Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data},
  author={[Authors]},
  booktitle={ACL},
  year={2024}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact [fengxiandong@mail.ustc.edu.cn].
