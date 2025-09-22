"""
Configuration settings for demand forecasting pipeline
Author: Divya Nayan (divyanayan88@gmail.com)
Copyright: © 2024 Divya Nayan. All rights reserved.
"""

import os
from pathlib import Path

# Project root directory - using absolute path resolution
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data paths - converted to strings for compatibility
DATA_PATH = str(PROJECT_ROOT / 'data' / 'raw' / 'static' / 'training_data.csv')
OUTPUT_DIR = str(PROJECT_ROOT / 'demand_forecasting_outputs')

# Model configuration
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
        'max_depth': {'type': 'int_none', 'low': 3, 'high': 15},
        'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
        'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
        'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]}
    },
    'xgboost': {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
        'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0}
    },
    'lightgbm': {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
        'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0}
    },
    'ridge': {
        'alpha': {'type': 'float', 'low': 0.1, 'high': 10.0}
    },
    'elastic_net': {
        'alpha': {'type': 'float', 'low': 0.1, 'high': 10.0},
        'l1_ratio': {'type': 'float', 'low': 0.1, 'high': 0.9}
    }
}

# Training configuration
TRAINING_CONFIG = {
    'n_trials': 50,
    'cv_folds': 5,
    'test_size': 0.2,
    'random_state': 42
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'lag_features': [1, 2, 3, 7, 14, 21, 30],
    'rolling_windows': [3, 7, 14, 30],
    'ewm_spans': [3, 7, 14, 30]
}

# Forecasting configuration
FORECAST_CONFIG = {
    'daily_periods': 30,
    'weekly_periods': 12,
    'monthly_periods': 6
}