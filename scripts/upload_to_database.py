#!/usr/bin/env python3
"""
Database Upload Script for Forecasting Results
Uploads merged forecasting outputs to SQL Server database
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
    """Handle database uploads for forecasting results"""
    
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
        
        # Output directory - dynamically set based on project structure
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(project_root, 'demand_forecasting_outputs')
        
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
    
    def upload_historical_predictions(self):
        """Upload merged historical predictions to database"""
        file_path = os.path.join(self.output_dir, 'merged_historical_predictions.csv')
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
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
            logger.info(f"✓ Historical predictions uploaded to table: {table_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading historical predictions: {str(e)}")
            return False
    
    def upload_daily_forecasts(self):
        """Upload merged daily forecasts to database"""
        file_path = os.path.join(self.output_dir, 'merged_daily_forecasts.csv')
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
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
            logger.info(f"✓ Daily forecasts uploaded to table: {table_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading daily forecasts: {str(e)}")
            return False
    
    def upload_weekly_forecasts(self):
        """Upload merged weekly forecasts to database"""
        file_path = os.path.join(self.output_dir, 'merged_weekly_forecasts.csv')
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return False
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded weekly forecasts: {len(df)} records")
            
            # Create meaningful table name
            date_range = self.get_forecast_date_range(df)
            
            if 'RouteCode' in df.columns:
                routes = df['RouteCode'].unique()
                if len(routes) <= 3:
                    route_str = '_'.join([str(r) for r in sorted(routes)])
                else:
                    route_str = f"{len(routes)}routes"
            else:
                route_str = "allroutes"
            
            table_name = f"yaumi_demand_weekly_forecast_route_{route_str}_{date_range}_run_{self.run_date}"
            
            # Add metadata columns
            df['upload_datetime'] = datetime.now()
            df['model_run_id'] = self.run_date
            df['forecast_type'] = 'weekly'
            
            # Upload to database
            df.to_sql(table_name, self.engine, if_exists='append', index=False)
            logger.info(f"✓ Weekly forecasts uploaded to table: {table_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading weekly forecasts: {str(e)}")
            return False
    
    def upload_monthly_forecasts(self):
        """Upload merged monthly forecasts to database"""
        file_path = os.path.join(self.output_dir, 'merged_monthly_forecasts.csv')
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return False
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded monthly forecasts: {len(df)} records")
            
            # Create meaningful table name
            date_range = self.get_forecast_date_range(df)
            
            if 'RouteCode' in df.columns:
                routes = df['RouteCode'].unique()
                if len(routes) <= 3:
                    route_str = '_'.join([str(r) for r in sorted(routes)])
                else:
                    route_str = f"{len(routes)}routes"
            else:
                route_str = "allroutes"
            
            table_name = f"yaumi_demand_monthly_forecast_route_{route_str}_{date_range}_run_{self.run_date}"
            
            # Add metadata columns
            df['upload_datetime'] = datetime.now()
            df['model_run_id'] = self.run_date
            df['forecast_type'] = 'monthly'
            
            # Upload to database
            df.to_sql(table_name, self.engine, if_exists='append', index=False)
            logger.info(f"✓ Monthly forecasts uploaded to table: {table_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading monthly forecasts: {str(e)}")
            return False
    
    def upload_training_summary(self):
        """Upload training summary to database"""
        file_path = os.path.join(self.output_dir, 'training_summary.csv')
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return False
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded training summary: {len(df)} records")
            
            table_name = f"yaumi_demand_ml_models_training_summary_run_{self.run_date}"
            
            # Add metadata columns
            df['upload_datetime'] = datetime.now()
            df['model_run_id'] = self.run_date
            
            # Upload to database
            df.to_sql(table_name, self.engine, if_exists='append', index=False)
            logger.info(f"✓ Training summary uploaded to table: {table_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading training summary: {str(e)}")
            return False
    
    def upload_all(self):
        """Upload all available forecast results to database"""
        if not self.connect_to_database():
            return False
        
        logger.info("="*60)
        logger.info("STARTING DATABASE UPLOAD PROCESS")
        logger.info("="*60)
        
        success_count = 0
        total_uploads = 0
        
        # List of uploads to perform
        uploads = [
            ("Historical Predictions", self.upload_historical_predictions),
            ("Daily Forecasts", self.upload_daily_forecasts),
            ("Weekly Forecasts", self.upload_weekly_forecasts),
            ("Monthly Forecasts", self.upload_monthly_forecasts),
            ("Training Summary", self.upload_training_summary)
        ]
        
        for name, upload_func in uploads:
            total_uploads += 1
            logger.info(f"\n--- Uploading {name} ---")
            if upload_func():
                success_count += 1
            else:
                logger.error(f"Failed to upload {name}")
        
        self.disconnect()
        
        logger.info("="*60)
        logger.info(f"UPLOAD COMPLETE: {success_count}/{total_uploads} successful")
        logger.info("="*60)
        
        return success_count == total_uploads


def main():
    """Main function to run the database upload"""
    try:
        uploader = DatabaseUploader()
    except ValueError as e:
        logger.error(str(e))
        logger.error("Please create a .env file with database credentials (see .env.example)")
        return
    
    # Check if output directory exists
    if not os.path.exists(uploader.output_dir):
        logger.error(f"Output directory not found: {uploader.output_dir}")
        logger.error("Please run main.py first to generate the forecasting outputs.")
        return
    
    # Check if any output files exist
    required_files = [
        'merged_historical_predictions.csv',
        'merged_daily_forecasts.csv'
    ]
    
    existing_files = []
    for file in required_files:
        if os.path.exists(os.path.join(uploader.output_dir, file)):
            existing_files.append(file)
    
    if not existing_files:
        logger.error("No forecast output files found!")
        logger.error("Please run main.py first to generate the forecasting outputs.")
        return
    
    logger.info(f"Found {len(existing_files)} output files to upload")
    
    # Perform uploads
    success = uploader.upload_all()
    
    if success:
        print("\n✅ All uploads completed successfully!")
        print(f"Data uploaded to database: {uploader.server}/{uploader.database}")
        print(f"Run ID: {uploader.run_date}")
    else:
        print("\n⚠️ Some uploads failed. Check the logs above for details.")


if __name__ == "__main__":
    main()