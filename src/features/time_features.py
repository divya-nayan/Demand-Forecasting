"""
Time-based feature engineering for demand forecasting
Author: Divya Nayan (divyanayan88@gmail.com)
Copyright: © 2024 Divya Nayan. All rights reserved.
"""

import pandas as pd
import numpy as np
from typing import Set


def create_time_features(df: pd.DataFrame, holidays_set: Set) -> pd.DataFrame:
    """Create time-based features without data leakage"""
    df = df.copy()
    
    # Basic temporal features
    df['DayOfWeek'] = df['TrxDate'].dt.dayofweek
    df['Month'] = df['TrxDate'].dt.month
    df['Quarter'] = df['TrxDate'].dt.quarter
    df['DayOfMonth'] = df['TrxDate'].dt.day
    df['WeekOfYear'] = df['TrxDate'].dt.isocalendar().week
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['IsHoliday'] = df['TrxDate'].dt.date.isin(holidays_set).astype(int)
    
    # Cyclical encoding for temporal features
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfMonth_sin'] = np.sin(2 * np.pi * df['DayOfMonth'] / 31)
    df['DayOfMonth_cos'] = np.cos(2 * np.pi * df['DayOfMonth'] / 31)
    
    return df