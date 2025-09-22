# Demand Forecasting System

A modular machine learning pipeline for demand forecasting using ensemble methods.

## Overview

This project implements a comprehensive demand forecasting system that uses multiple machine learning models (Random Forest, XGBoost, LightGBM, Ridge, ElasticNet) combined in an ensemble to predict future demand patterns.

## Features

- **Modular Architecture**: Clean separation of concerns with organized modules for data processing, feature engineering, model training, and evaluation
- **Ensemble Learning**: Combines multiple ML models for improved accuracy
- **Automated Feature Engineering**: Time-based features, lag features, rolling statistics
- **Hyperparameter Optimization**: Uses Optuna for automated hyperparameter tuning
- **Multiple Forecast Horizons**: Generates daily, weekly, and monthly forecasts
- **Database Integration**: Supports SQL Server for data storage and retrieval

## Project Structure

```
demand_forecasting/
├── config/                  # Configuration files
│   ├── config.py           # Main configuration
│   └── database_config_template.py  # Database config template
├── src/                    # Source code modules
│   ├── data/              # Data loading and processing
│   ├── features/          # Feature engineering
│   ├── models/            # Model definitions and ensemble
│   ├── evaluation/        # Model evaluation metrics
│   └── utils/             # Utility functions
├── scripts/               # Standalone scripts
│   └── upload_to_database.py  # Database upload utility
├── notebooks/             # Jupyter notebooks for analysis
├── main.py               # Main pipeline execution
├── requirements.txt      # Python dependencies
└── .env.example         # Environment variables template
```

## Setup

### Prerequisites

- Python 3.8 or higher
- SQL Server (optional, for database storage)
- ODBC Driver 17 for SQL Server (if using database features)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd demand_forecasting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your database credentials
```

## Configuration

### Database Setup (Optional)

If you want to use database features:

1. Copy `.env.example` to `.env`
2. Fill in your database credentials:
```
DB_SERVER=your_server_address
DB_DATABASE=your_database_name
DB_USERNAME=your_username
DB_PASSWORD=your_password
```

### Data Setup

Place your training data in the appropriate directory structure:
- Training data: `data/raw/static/training_data.csv`

Required columns in training data:
- `TrxDate`: Transaction date
- `RouteCode`: Route identifier
- `ItemCode`: Item identifier
- `TotalQuantity`: Quantity to forecast
- Additional columns: `WarehouseCode`, `ItemName`, `CategoryName`, `AvgUnitPrice`, `SalesValue`

## Usage

### Running the Main Pipeline

```bash
python main.py
```

This will:
1. Load and validate data
2. Process each RouteCode-ItemCode combination
3. Train multiple models with hyperparameter optimization
4. Create ensemble predictions
5. Generate forecasts for multiple horizons
6. Save outputs to `demand_forecasting_outputs/`

### Uploading Results to Database

```bash
python scripts/upload_to_database.py
```

## Model Details

The system uses the following models in an ensemble:
- **Random Forest**: Robust to outliers, handles non-linear relationships
- **XGBoost**: Gradient boosting for high accuracy
- **LightGBM**: Fast gradient boosting
- **Ridge Regression**: Linear model with L2 regularization
- **ElasticNet**: Linear model with L1 and L2 regularization

## Output Files

The pipeline generates:
- `merged_daily_forecasts.csv`: 30-day daily forecasts
- `merged_weekly_forecasts.csv`: 12-week weekly forecasts
- `merged_monthly_forecasts.csv`: 6-month monthly forecasts
- `training_summary.csv`: Model performance metrics
- Individual combination outputs in subdirectories

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Security Notes

- Never commit `.env` files or `database_config.py` with real credentials
- Use environment variables for all sensitive information
- Review `.gitignore` before committing

## License

[Specify your license here]

## Contact

[Your contact information]