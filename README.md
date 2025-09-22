# Demand Forecasting Pipeline

A production-level demand forecasting system for time series prediction with comprehensive model evaluation and forecasting capabilities.

## Project Structure

```
forecasting/
├── main.py                      # Modular pipeline using src/ components
├── requirements.txt             # Python dependencies
├── DATABASE_UPLOAD_README.md    # Database upload documentation
├── config/
│   ├── config.py               # Configuration parameters
│   └── database_config.py      # Database configuration
├── src/                        # Modular source code components
│   ├── data/                   # Data loading and processing
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Data loading utilities
│   │   └── data_processor.py   # Data preparation
│   ├── features/               # Feature engineering
│   │   ├── __init__.py
│   │   ├── time_features.py    # Time-based features
│   │   └── lag_features.py     # Lag and rolling features
│   ├── models/                 # Model definitions and training
│   │   ├── __init__.py
│   │   ├── model_configs.py    # Model configurations
│   │   └── ensemble.py         # Ensemble methods
│   ├── evaluation/             # Model evaluation utilities
│   │   ├── __init__.py
│   │   ├── evaluator.py        # Model evaluator class
│   │   └── metrics.py          # Evaluation metrics
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── holidays.py         # Holiday utilities
│       └── metrics.py          # Custom metrics
├── data/                       # Data files
│   └── raw/                    # Raw data files
│       ├── static/             # CSV data files
│       │   ├── training_data.csv
│       │   ├── demand_data.csv
│       │   └── ...
│       └── sql/                # SQL queries
│           ├── demand_data.sql
│           └── ...
├── notebooks/                  # Jupyter notebooks for data analysis
│   ├── data_fetch.ipynb
│   ├── data_preprocessing.ipynb
│   └── data_postprocessing.ipynb
├── scripts/                    # Utility scripts
│   ├── upload_to_database.py   # Database upload script
│   ├── setup.py               # Project setup script
│   └── main_original.py       # Backup of original script
├── updated_demand_forecasting_outputs/ # Modular pipeline outputs  
└── original_demand_forecasting_outputs/ # Original script outputs (if run)
```

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python main.py
```

This runs the modular forecasting pipeline using organized `src/` components.  
**Output Directory**: `updated_demand_forecasting_outputs/`

**Note**: If you run the original script (`python scripts/main_original.py`), it will create a separate directory called `original_demand_forecasting_outputs/`.

### 3. Upload Results to Database (Optional)

```bash
python scripts/upload_to_database.py
```

This uploads the forecasting results to SQL Server database with meaningful table names.

## Features

- **Multi-model forecasting**: Random Forest, XGBoost, LightGBM, Ridge, Elastic Net
- **Hyperparameter optimization**: Using Optuna for automated parameter tuning
- **Feature engineering**: Time-based features, lag features, rolling statistics
- **Comprehensive evaluation**: Multiple metrics (MAE, RMSE, MAPE, R²)
- **Future predictions**: Daily, weekly, and monthly forecasts
- **Model persistence**: Save/load trained models
- **Leakage prevention**: Strict data splitting and feature engineering

## Configuration

Edit `config/config.py` to modify:
- Model parameters and hyperparameter search spaces
- Training configuration (CV folds, trials, etc.)
- Feature engineering settings
- Forecasting periods

## Data Requirements

The pipeline expects a CSV file with the following columns:
- TrxDate: Transaction date
- RouteCode: Route identifier
- ItemCode: Item identifier
- TotalQuantity: Target variable for forecasting
- Additional columns for enhanced features

## Output

The pipeline generates:
- Trained models (saved as .joblib files)
- Evaluation reports and metrics
- Future forecasts (daily, weekly, monthly)
- Comprehensive Excel reports with multiple sheets

## Modular Design

The codebase has been organized into logical modules:
- **Data**: Loading and preprocessing utilities
- **Features**: Feature engineering components
- **Utils**: Shared utilities (metrics, holidays)
- **Config**: Centralized configuration management

This structure improves maintainability while preserving the original functionality.