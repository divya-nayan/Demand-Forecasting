#!/usr/bin/env python3
"""
Upload specific forecasting files to database with new date-only naming
Author: Divya Nayan (divyanayan88@gmail.com)
Copyright: © 2024 Divya Nayan. All rights reserved.
"""

import pandas as pd
import os
from datetime import datetime
from sqlalchemy import create_engine
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseUploader:
    """Upload files to database with new naming format"""

    def __init__(self):
        # Database configuration from environment variables
        self.server = os.getenv('DB_SERVER')
        self.database = os.getenv('DB_DATABASE')
        self.username = os.getenv('DB_USERNAME')
        self.password = os.getenv('DB_PASSWORD')
        self.driver = os.getenv('DB_DRIVER', 'ODBC+Driver+17+for+SQL+Server')

        # Validate credentials
        if not all([self.server, self.database, self.username, self.password]):
            raise ValueError("Missing database credentials. Please set environment variables or create .env file")

        self.connection_string = f'mssql+pyodbc://{self.username}:{self.password}@{self.server}/{self.database}?driver={self.driver}'
        self.engine = None

        # Current date for table naming (date only, no time)
        self.run_date = datetime.now().strftime('%Y%m%d')

    def connect_to_database(self):
        """Create database connection"""
        try:
            self.engine = create_engine(self.connection_string)
            logger.info(f"Connected to database: {self.server}/{self.database}")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False

    def disconnect(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")

    def get_forecast_date_range(self, df):
        """Extract date range from forecast data for table naming"""
        try:
            if 'Date' in df.columns:
                min_date = pd.to_datetime(df['Date']).min().strftime('%Y%m%d')
                max_date = pd.to_datetime(df['Date']).max().strftime('%Y%m%d')
                return f"{min_date}_to_{max_date}"
            elif 'TrxDate' in df.columns:
                min_date = pd.to_datetime(df['TrxDate']).min().strftime('%Y%m%d')
                max_date = pd.to_datetime(df['TrxDate']).max().strftime('%Y%m%d')
                return f"{min_date}_to_{max_date}"
            else:
                return "unknown_dates"
        except:
            return "unknown_dates"

    def upload_daily_forecasts(self, file_path):
        """Upload daily forecasts file"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded daily forecasts: {len(df)} records")

            # Create meaningful table name with route and date range
            date_range = self.get_forecast_date_range(df)

            # Get unique routes for table name
            if 'RouteCode' in df.columns:
                routes = df['RouteCode'].unique()
                if len(routes) <= 3:
                    route_str = '_'.join([str(r) for r in sorted(routes)])
                else:
                    route_str = f"{len(routes)}routes"
            else:
                route_str = "allroutes"

            table_name = f"yaumi_demand_daily_forecast_route_{route_str}_{date_range}_run_{self.run_date}"

            # Add metadata columns
            df['upload_datetime'] = datetime.now()
            df['model_run_id'] = self.run_date
            df['forecast_type'] = 'daily'

            # Upload to database
            df.to_sql(table_name, self.engine, if_exists='append', index=False)
            logger.info(f"Daily forecasts uploaded to table: {table_name}")

            return True, table_name

        except Exception as e:
            logger.error(f"Error uploading daily forecasts: {str(e)}")
            return False, None

    def upload_historical_predictions(self, file_path):
        """Upload historical predictions file"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded historical predictions: {len(df)} records")

            # Create meaningful table name
            date_range = self.get_forecast_date_range(df)
            table_name = f"yaumi_demand_historical_model_performance_{date_range}_run_{self.run_date}"

            # Add metadata columns
            df['upload_datetime'] = datetime.now()
            df['model_run_id'] = self.run_date

            # Upload to database
            df.to_sql(table_name, self.engine, if_exists='append', index=False)
            logger.info(f"Historical predictions uploaded to table: {table_name}")

            return True, table_name

        except Exception as e:
            logger.error(f"Error uploading historical predictions: {str(e)}")
            return False, None

def main():
    """Main function to upload files with new naming"""
    # File paths - using dynamic paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    daily_forecasts_path = os.path.join(project_root, 'demand_forecasting_outputs', 'merged_daily_forecasts.csv')
    historical_predictions_path = os.path.join(project_root, 'demand_forecasting_outputs', 'merged_historical_predictions.csv')

    try:
        uploader = DatabaseUploader()
    except ValueError as e:
        logger.error(str(e))
        logger.error("Please create a .env file with database credentials (see .env.example)")
        return False

    if not uploader.connect_to_database():
        return False

    print("="*60)
    print("UPLOADING FILES WITH NEW DATE-ONLY NAMING")
    print("="*60)

    success_count = 0
    total_uploads = 2
    uploaded_tables = []

    # Upload daily forecasts
    print("\n--- Uploading Daily Forecasts ---")
    success, table_name = uploader.upload_daily_forecasts(daily_forecasts_path)
    if success:
        success_count += 1
        uploaded_tables.append(table_name)
        print(f"✓ SUCCESS: {table_name}")
    else:
        print("✗ FAILED: Daily forecasts upload")

    # Upload historical predictions
    print("\n--- Uploading Historical Predictions ---")
    success, table_name = uploader.upload_historical_predictions(historical_predictions_path)
    if success:
        success_count += 1
        uploaded_tables.append(table_name)
        print(f"✓ SUCCESS: {table_name}")
    else:
        print("✗ FAILED: Historical predictions upload")

    uploader.disconnect()

    print("="*60)
    print(f"UPLOAD COMPLETE: {success_count}/{total_uploads} successful")
    print("="*60)

    if success_count == total_uploads:
        print("\n✅ ALL UPLOADS COMPLETED SUCCESSFULLY!")
        print(f"Database: {uploader.server}/{uploader.database}")
        print(f"Run Date: {uploader.run_date}")
        print("\nTables Created:")
        for table in uploaded_tables:
            print(f"  • {table}")
        return True
    else:
        print(f"\n⚠️ {total_uploads - success_count} uploads failed.")
        return False

if __name__ == "__main__":
    main()