# Cascading Disaster Prediction

How can ML modeling capture cascade dynamics in disaster sequences?

## Project Overview

This project analyzes the NOAA Storm Events Database to predict cascading disaster patterns — understanding how one weather event can trigger secondary events (e.g., hurricanes → flash floods → debris flows). We frame this as a **multilabel classification** problem: given a primary weather event, predict which (if any) secondary event types it will trigger.

The project also includes an **equity analysis** examining whether cascading disasters disproportionately impact socially vulnerable communities using the CDC Social Vulnerability Index.

## Quick Start

```bash
# Option 1: Conda (recommended)
conda env create -f environment.yml
conda activate cascading-disaster

# Option 2: pip
pip install -e .
```

See [docs/ENV_SETUP.md](docs/ENV_SETUP.md) for detailed environment setup.

## Project Structure

```
cascading-disaster-prediction/
├── data/
│   └── processed/                    # Labeled datasets (gitignored)
│       └── events_labeled_full.csv   # ~967K events with cascade labels
├── docs/
│   ├── assignments/                  # Course assignment write-ups
│   ├── cascade_definition.md         # Cascade pattern definitions & references
│   ├── cascade_eda_notebook.md       # Guide to the EDA notebook
│   ├── events_labeled_full.md        # Dataset column documentation
│   ├── features.md                   # Feature engineering documentation
│   ├── feature_eng.md                # Feature pipeline usage guide
│   ├── model_training.md             # Model training instructions
│   └── ENV_SETUP.md                  # Environment setup guide
├── notebooks/
│   ├── 01_cascade_eda.ipynb          # Exploratory data analysis
│   ├── 02_xgboost_multilabel_training.ipynb  # XGBoost multilabel models
│   ├── 03_sklearn_multilabel_baselines.ipynb  # Sklearn baselines (OvR, chains)
│   ├── 04_neural_networks.ipynb      # Neural network architectures
│   └── 05_equity_analysis.ipynb      # Fairness & vulnerability analysis
├── results/
│   └── cascade_eda/                  # EDA visualizations and summary tables
├── src/
│   ├── cascade_definition.py         # 130+ documented causal cascade patterns
│   ├── cascade_identification.py     # Cascade detection from event data
│   ├── labeling.py                   # Multilabel target creation
│   ├── data_loader.py                # NOAA data loading & parsing
│   ├── features.py                   # Base feature engineering
│   ├── aggregate_features.py         # Learned aggregate statistics
│   └── prepare_data.py               # Train/test splitting & serialization
├── environment.yml                   # Conda environment spec
├── pyproject.toml                    # pip/setuptools config
└── GDRIVE_SETUP.md                   # Google Drive data access setup
```

## Pipeline

### 1. Data Loading & Cascade Identification

Load NOAA Storm Events (details, fatalities, locations tables), identify cascade pairs using temporal, spatial, and domain-knowledge constraints, and create multilabel targets.

```python
from src.data_loader import load_all_storm_data, join_storm_data
from src.cascade_identification import identify_cascades
from src.labeling import create_cascade_labels
```

### 2. Feature Engineering

Transform raw event data into ~170 features across temporal, impact, spatial, event-detail, historical, and aggregate categories.

```bash
python src/prepare_data.py --split_type chronological --filter_cascades False
```

See [docs/features.md](docs/features.md) for the full feature catalog.

### 3. Model Training

| Notebook | Model | Task |
|----------|-------|------|
| `02_xgboost_multilabel_training` | XGBoost (Binary Relevance) | Multilabel with per-label threshold tuning |
| `03_sklearn_multilabel_baselines` | Logistic Regression, Classifier Chains | Multilabel baselines (OvR, chain) |
| `04_neural_networks` | MLP, Shared-bottom, Weather-embedding NN | Neural multilabel with Optuna tuning |

See [docs/model_training.md](docs/model_training.md) for training details.

### 4. Equity Analysis

Notebook `05_equity_analysis` examines differential cascade exposure and impact across CDC Social Vulnerability Index quartiles.

## Documentation

| Document | Description |
|----------|-------------|
| [Cascade Definition](docs/cascade_definition.md) | 130+ scientifically documented causal cascade pairs with references |
| [Dataset Schema](docs/events_labeled_full.md) | Column-by-column documentation of the labeled dataset |
| [Feature Engineering](docs/features.md) | Full catalog of ~170 engineered features |
| [Feature Pipeline](docs/feature_eng.md) | How to run the data preparation pipeline |
| [Model Training](docs/model_training.md) | Model architectures and training instructions |
| [EDA Notebook Guide](docs/cascade_eda_notebook.md) | Walkthrough of the exploratory analysis |
| [Environment Setup](docs/ENV_SETUP.md) | Conda and pip installation instructions |
| [Google Drive Setup](GDRIVE_SETUP.md) | Team data access via Google Drive |

## Data Source

[NOAA Storm Events Database](https://www.ncei.noaa.gov/stormevents/ftp.jsp) — event-level storm and weather incident data from the National Weather Service (2011-2025).

## Collaboration

[Project doc](https://docs.google.com/document/d/1Es5w9ciBOnjKLxs32QlmwyXypAC8rs1m9SwG4Km8HAE/edit?usp=sharing)
