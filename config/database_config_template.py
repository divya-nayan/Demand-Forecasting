"""
Database configuration template for demand forecasting system
Copy this file to database_config.py and fill in your actual credentials
DO NOT commit database_config.py to version control!
Author: Divya Nayan (divyanayan88@gmail.com)
Copyright: © 2024 Divya Nayan. All rights reserved.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database connection settings from environment variables
DATABASE_CONFIG = {
    'server': os.getenv('DB_SERVER', 'your_server_address'),
    'database': os.getenv('DB_DATABASE', 'your_database_name'),
    'username': os.getenv('DB_USERNAME', 'your_username'),
    'password': os.getenv('DB_PASSWORD', 'your_password'),
    'driver': os.getenv('DB_DRIVER', 'ODBC+Driver+17+for+SQL+Server')
}

# Table naming patterns
TABLE_NAME_PATTERNS = {
    'historical_predictions': 'historical_predictions_{date_range}_{run_id}',
    'daily_forecasts': 'daily_forecast_route{routes}_{date_range}_{run_id}',
    'weekly_forecasts': 'weekly_forecast_route{routes}_{date_range}_{run_id}',
    'monthly_forecasts': 'monthly_forecast_route{routes}_{date_range}_{run_id}',
    'training_summary': 'training_summary_{run_id}'
}

def get_connection_string():
    """Get SQLAlchemy connection string"""
    config = DATABASE_CONFIG
    return f"mssql+pyodbc://{config['username']}:{config['password']}@{config['server']}/{config['database']}?driver={config['driver']}"