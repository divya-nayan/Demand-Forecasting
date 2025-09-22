import numpy as np
from typing import Union, List


def calculate_mape(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> float:
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100