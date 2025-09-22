# Database Upload Guide

Author: Divya Nayan (divyanayan88@gmail.com)
Copyright: © 2024 Divya Nayan. All rights reserved.

This guide explains how to upload forecasting results to the SQL Server database after running the main pipeline.

## Prerequisites

1. **Run the main forecasting pipeline first:**
   ```bash
   python standalone_demand_script.py
   ```
   This generates the output files in `demand_forecasting_outputs/`

2. **Ensure SQLAlchemy is installed:**
   ```bash
   uv pip install sqlalchemy
   # or
   pip install sqlalchemy
   ```

## Database Upload Process

### Step 1: Run the Upload Script
```bash
python scripts/upload_to_database.py
```

### What Gets Uploaded

The script uploads the following files to SQL Server (`YaumiAIML` database):

| File | Table Name Pattern | Description |
|------|-------------------|-------------|
| `merged_historical_predictions.csv` | `yaumi_demand_historical_model_performance_{date_range}_run_{run_id}` | Model predictions on historical test data |
| `merged_daily_forecasts.csv` | `yaumi_demand_daily_forecast_route_{routes}_{date_range}_run_{run_id}` | Daily forecasts for next 30 days |
| `merged_weekly_forecasts.csv` | `yaumi_demand_weekly_forecast_route_{routes}_{date_range}_run_{run_id}` | Weekly forecasts for next 12 weeks |
| `merged_monthly_forecasts.csv` | `yaumi_demand_monthly_forecast_route_{routes}_{date_range}_run_{run_id}` | Monthly forecasts for next 6 months |
| `training_summary.csv` | `yaumi_demand_ml_models_training_summary_run_{run_id}` | Model training performance summary |

### Example Table Names

For a run on 2025-01-15:

- `yaumi_demand_historical_model_performance_20210102_to_20241231_run_20250115`
- `yaumi_demand_daily_forecast_route_1004_20250116_to_20250215_run_20250115`  
- `yaumi_demand_weekly_forecast_route_1004_20250120_to_20250412_run_20250115`
- `yaumi_demand_monthly_forecast_route_1004_20250201_to_20250731_run_20250115`
- `yaumi_demand_ml_models_training_summary_run_20250115`

### Additional Metadata Columns

Each table includes these automatically added columns:
- `upload_datetime` - When the data was uploaded
- `model_run_id` - Unique identifier for the model run
- `forecast_type` - Type of forecast (daily/weekly/monthly)

## Configuration

Database settings are in `config/database_config.py`:

```python
DATABASE_CONFIG = {
    'server': '20.46.47.104',
    'database': 'YaumiAIML', 
    'username': 'sandeep',
    'password': 'Winit$1234',
    'driver': 'ODBC+Driver+17+for+SQL+Server'
}
```

## Usage Workflow

### Complete Process:
```bash
# 1. Run forecasting pipeline
python standalone_demand_script.py

# 2. Upload results to database
python scripts/upload_to_database.py
```

### Expected Output:
```
2025-01-15 14:30:25,123 - INFO - ============================================================
2025-01-15 14:30:25,123 - INFO - STARTING DATABASE UPLOAD PROCESS
2025-01-15 14:30:25,123 - INFO - ============================================================
2025-01-15 14:30:25,200 - INFO - Connected to database: 20.46.47.104/YaumiAIML

--- Uploading Historical Predictions ---
2025-01-15 14:30:26,100 - INFO - Loaded historical predictions: 15432 records
2025-01-15 14:30:27,300 - INFO - ✓ Historical predictions uploaded to table: historical_predictions_20210102_to_20241231_20250115_143025

--- Uploading Daily Forecasts ---
2025-01-15 14:30:28,100 - INFO - Loaded daily forecasts: 1170 records
2025-01-15 14:30:28,800 - INFO - ✓ Daily forecasts uploaded to table: daily_forecast_route1004_20250116_to_20250215_20250115_143025

... (similar for weekly, monthly, training summary)

2025-01-15 14:30:35,123 - INFO - ============================================================
2025-01-15 14:30:35,123 - INFO - UPLOAD COMPLETE: 5/5 successful
2025-01-15 14:30:35,123 - INFO - ============================================================

✅ All uploads completed successfully!
Data uploaded to database: 20.46.47.104/YaumiAIML
Run ID: 20250115_143025
```

## Troubleshooting

### Common Issues:

1. **"Output directory not found"**
   - Run `python standalone_demand_script.py` first to generate outputs

2. **"No forecast output files found"** 
   - Ensure `standalone_demand_script.py` completed successfully
   - Check that `demand_forecasting_outputs/` directory exists

3. **Database connection errors**
   - Verify network connectivity to server
   - Check credentials in `config/database_config.py`
   - Ensure SQL Server ODBC driver is installed

4. **Table already exists errors**
   - The script uses `if_exists='append'` so data is added to existing tables
   - Each run creates uniquely named tables with date stamps

## Database Schema

Tables will have columns based on the CSV structure plus metadata:
- All original CSV columns
- `upload_datetime` (datetime)
- `model_run_id` (varchar)
- `forecast_type` (varchar, for forecast tables)

## Security Note

Database credentials are stored in plain text in the config file. In production, consider using:
- Environment variables
- Azure Key Vault
- Encrypted configuration files