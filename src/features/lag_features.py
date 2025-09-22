"""
Lag and rolling feature engineering for demand forecasting
Author: Divya Nayan (divyanayan88@gmail.com)
Copyright: © 2024 Divya Nayan. All rights reserved.
"""

import pandas as pd
import numpy as np


def create_lag_features(df: pd.DataFrame, target_col: str = 'TotalQuantity') -> pd.DataFrame:
    """Create lag features with strict data leakage prevention"""
    df = df.copy()
    
    # STRICT LAG FEATURES - All shifted to prevent leakage
    lags = [1, 2, 3, 7, 14, 21, 30]
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # ROLLING STATISTICS - All with proper shifting to prevent leakage
    windows = [3, 7, 14, 30]
    for window in windows:
        # Shift by 1 and then calculate rolling stats to prevent leakage
        shifted_series = df[target_col].shift(1)
        df[f'rolling_mean_{window}'] = shifted_series.rolling(
            window=window, min_periods=max(1, window//2)
        ).mean()
        df[f'rolling_std_{window}'] = shifted_series.rolling(
            window=window, min_periods=max(1, window//2)
        ).std()
        df[f'rolling_median_{window}'] = shifted_series.rolling(
            window=window, min_periods=max(1, window//2)
        ).median()
    
    # EXPONENTIAL WEIGHTED FEATURES - All with proper shifting
    for span in [3, 7, 14, 30]:
        df[f'ewm_mean_{span}'] = df[target_col].shift(1).ewm(
            span=span, min_periods=1
        ).mean()
    
    # TREND FEATURES - All with proper shifting
    def safe_trend(x):
        if len(x) >= 2:
            return np.polyfit(range(len(x)), x, 1)[0]
        return 0
    
    df['trend_3'] = df[target_col].shift(1).rolling(
        window=3, min_periods=2
    ).apply(safe_trend, raw=False)
    
    df['trend_7'] = df[target_col].shift(1).rolling(
        window=7, min_periods=3
    ).apply(safe_trend, raw=False)
    
    return df