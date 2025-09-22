"""
Model evaluation and performance analysis for demand forecasting
Author: Divya Nayan (divyanayan88@gmail.com)
Copyright: © 2024 Divya Nayan. All rights reserved.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .metrics import calculate_all_metrics, create_evaluation_summary


class ModelEvaluator:
    """Handle model evaluation and performance analysis"""
    
    def __init__(self):
        self.evaluation_results = []
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str, route_code: str, item_code: str) -> Dict[str, float]:
        """Evaluate a single model and store results"""
        metrics = calculate_all_metrics(y_true, y_pred)
        
        # Store evaluation result
        eval_result = {
            'RouteCode': route_code,
            'ItemCode': item_code,
            'Model': model_name,
            **metrics
        }
        self.evaluation_results.append(eval_result)
        
        return metrics
    
    def get_best_model(self, models_results: Dict, metric: str = 'MAE') -> str:
        """Find the best performing model based on specified metric"""
        best_model = None
        best_score = float('inf') if metric in ['MAE', 'RMSE', 'MAPE'] else float('-inf')
        
        for model_name, result in models_results.items():
            if 'metrics' in result:
                score = result['metrics'][metric]
                if metric in ['MAE', 'RMSE', 'MAPE']:
                    if score < best_score:
                        best_score = score
                        best_model = model_name
                else:  # R2
                    if score > best_score:
                        best_score = score
                        best_model = model_name
        
        return best_model
    
    def create_summary_report(self) -> pd.DataFrame:
        """Create summary report of all evaluations"""
        return pd.DataFrame(self.evaluation_results)
    
    def clear_results(self):
        """Clear stored evaluation results"""
        self.evaluation_results = []