"""
Evaluation metrics for demand forecasting models
Author: Divya Nayan (divyanayan88@gmail.com)
Copyright: © 2024 Divya Nayan. All rights reserved.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple
from ..utils.metrics import calculate_mape


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all evaluation metrics"""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': calculate_mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }


def create_evaluation_summary(model_results: Dict, route_code: str, item_code: str) -> pd.DataFrame:
    """Create evaluation summary DataFrame from model results"""
    eval_data = []
    
    for model_name, result in model_results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            eval_data.append({
                'RouteCode': route_code,
                'ItemCode': item_code,
                'Model': model_name,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE'],
                'R2': metrics['R2'],
                'Best_Params': str(result.get('best_params', {}))
            })
    
    return pd.DataFrame(eval_data)