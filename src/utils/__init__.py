"""
Utility functions for demand forecasting
Author: Divya Nayan (divyanayan88@gmail.com)
Copyright: © 2024 Divya Nayan. All rights reserved.
"""

from .holidays import get_uae_holidays
from .metrics import calculate_mape

__all__ = ['get_uae_holidays', 'calculate_mape']