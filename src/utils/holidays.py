import holidays
from typing import Set
from datetime import date


def get_uae_holidays(start_year: int, end_year: int) -> Set[date]:
    """Get UAE holidays for the given year range"""
    uae_holidays = holidays.country_holidays('AE')
    holiday_dates = []
    for year in range(start_year, end_year + 3):
        for date_val in uae_holidays.get(year, {}):
            holiday_dates.append(date_val)
    return set(holiday_dates)