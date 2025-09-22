#!/usr/bin/env python3
"""
Template for data fetching from database
Use this as a reference for creating notebooks
"""

import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_database_connection():
    """Create database connection using environment variables"""
    server = os.getenv('DB_SERVER')
    database = os.getenv('DB_DATABASE')
    username = os.getenv('DB_USERNAME')
    password = os.getenv('DB_PASSWORD')
    driver = os.getenv('DB_DRIVER', 'ODBC+Driver+17+for+SQL+Server')

    if not all([server, database, username, password]):
        raise ValueError("Missing database credentials. Please set environment variables or create .env file")

    connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'
    engine = create_engine(connection_string)
    return engine

def fetch_data(query):
    """Fetch data from database"""
    engine = get_database_connection()
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df

# Example usage:
if __name__ == "__main__":
    # Example query (replace with your actual query)
    query = "SELECT TOP 10 * FROM your_table"

    try:
        data = fetch_data(query)
        print(f"Fetched {len(data)} records")
        print(data.head())
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure you have created a .env file with database credentials")