#!/usr/bin/env python3
"""
Modular Demand Forecasting Pipeline
This version uses modular components from src/ directory
Performs the same functionality as standalone_demand_script.py but with modular architecture
"""

import pandas as pd
import numpy as np
import warnings
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import joblib

warnings.filterwarnings('ignore')

# Import modular components
from src.data import load_and_validate_data, prepare_combination_data
from src.features import create_time_features, create_lag_features
from src.models import get_model_configs, create_ensemble
from src.evaluation import ModelEvaluator, calculate_all_metrics
from src.utils import get_uae_holidays, calculate_mape

# Import ML libraries
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModularDemandForecastingPipeline:
    """Modular version of the demand forecasting pipeline using src/ components"""
    
    def __init__(self, data_path: str, output_dir: str = None):
        if output_dir is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(project_root, 'demand_forecasting_outputs')
        self.data_path = data_path
        self.output_dir = output_dir
        self.holidays_set = None
        self.evaluator = ModelEvaluator()
        self.all_evaluations = []
        self.all_historical_predictions = []
        self.all_daily_forecasts = []
        self.all_weekly_forecasts = []
        self.all_monthly_forecasts = []
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories for saving models and outputs"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'evaluation'), exist_ok=True)
        logger.info("Created output directories")
        
    def get_combo_dir(self, route_code: str, item_code: str) -> str:
        """Get directory path for specific combination"""
        combo_dir = os.path.join(self.output_dir, f"{route_code}_{item_code}")
        os.makedirs(os.path.join(combo_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(combo_dir, 'outputs'), exist_ok=True)
        os.makedirs(os.path.join(combo_dir, 'evaluation'), exist_ok=True)
        return combo_dir

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and validate data using modular components"""
        df, holidays_set = load_and_validate_data(self.data_path)
        self.holidays_set = holidays_set
        return df

    def prepare_combination_data_modular(self, df: pd.DataFrame, route_code: str, item_code: str) -> Optional[Dict]:
        """Prepare data for specific combination using modular components"""
        return prepare_combination_data(df, route_code, item_code, self.holidays_set)

    def optimize_model(self, model_name: str, model_config: Dict, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Optimize model hyperparameters using Optuna"""
        def objective(trial):
            params = {}
            model_class = model_config['model_class']
            param_space = model_config['param_space']
            
            for param_name, param_range in param_space.items():
                if param_name in ['random_state', 'verbosity', 'max_iter']:
                    params[param_name] = param_range
                elif isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            
            try:
                model = model_class(**params)
                cv_scores = []
                
                tscv = TimeSeriesSplit(n_splits=3, test_size=max(30, len(X_train) // 6))
                
                for train_idx, val_idx in tscv.split(X_train):
                    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    model.fit(X_cv_train, y_cv_train)
                    y_pred = model.predict(X_cv_val)
                    cv_scores.append(mean_absolute_error(y_cv_val, y_pred))
                
                return np.mean(cv_scores)
                
            except Exception as e:
                logger.warning(f"Trial failed for {model_name}: {str(e)}")
                return float('inf')
        
        try:
            study = optuna.create_study(direction='minimize', 
                                      sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=50, timeout=300)
            
            best_params = study.best_params
            # Add fixed parameters
            for param_name, param_value in model_config['param_space'].items():
                if param_name in ['random_state', 'verbosity', 'max_iter']:
                    best_params[param_name] = param_value
            
            return {
                'best_params': best_params,
                'best_score': study.best_value,
                'n_trials': len(study.trials)
            }
            
        except Exception as e:
            logger.error(f"Optimization failed for {model_name}: {str(e)}")
            return {'best_params': model_config.get('param_space', {}), 'best_score': float('inf')}

    def train_models(self, data_dict: Dict) -> Tuple[Dict, object]:
        """Train all models using modular components"""
        X_train_scaled = data_dict['X_train_scaled']
        y_train = data_dict['y_train']
        X_test_scaled = data_dict['X_test_scaled'] 
        y_test = data_dict['y_test']
        
        model_configs = get_model_configs()
        model_results = {}
        
        for model_name, config in model_configs.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Optimize hyperparameters
                optimization_result = self.optimize_model(
                    model_name, config, 
                    pd.DataFrame(X_train_scaled), pd.Series(y_train)
                )
                
                # Train final model with best params
                best_params = optimization_result['best_params']
                model = config['model_class'](**best_params)
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                # Calculate metrics using modular evaluation
                train_metrics = calculate_all_metrics(y_train, train_pred)
                test_metrics = calculate_all_metrics(y_test, test_pred)
                
                model_results[model_name] = {
                    'model': model,
                    'best_params': best_params,
                    'train_metrics': train_metrics,
                    'metrics': test_metrics,
                    'test_predictions': test_pred,
                    'train_predictions': train_pred,
                    'optimization_trials': optimization_result.get('n_trials', 0)
                }
                
                logger.info(f"{model_name} - Test MAE: {test_metrics['MAE']:.3f}, RMSE: {test_metrics['RMSE']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Create ensemble using modular component
        ensemble_result = create_ensemble(
            model_results, 
            pd.DataFrame(X_train_scaled),
            pd.Series(y_train),
            data_dict['feature_columns']
        )
        
        return model_results, ensemble_result

    def make_future_predictions(self, model_package: Dict, route_code: str, item_code: str, 
                               latest_data: pd.DataFrame) -> Dict:
        """Generate future predictions using trained models"""
        try:
            if 'ensemble_model' not in model_package:
                logger.error("No ensemble model found")
                return {}
            
            ensemble_model = model_package['ensemble_model']
            feature_columns = model_package['feature_columns']
            
            # Get the last known date
            last_date = latest_data['TrxDate'].max()
            
            # Generate future dates
            daily_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
            weekly_dates = pd.date_range(start=last_date + timedelta(days=7), periods=12, freq='W')  
            monthly_dates = pd.date_range(start=last_date + timedelta(days=30), periods=6, freq='M')
            
            predictions = {}
            
            # Create predictions for each timeframe
            for period_name, dates in [('daily', daily_dates), ('weekly', weekly_dates), ('monthly', monthly_dates)]:
                future_df = pd.DataFrame({'TrxDate': dates})
                future_df['RouteCode'] = route_code
                future_df['ItemCode'] = item_code
                
                # Add time features using modular component
                future_df = create_time_features(future_df, self.holidays_set)
                
                # Add lag features (using last known values)
                last_quantity = latest_data['TotalQuantity'].iloc[-1] if len(latest_data) > 0 else 0
                for lag in [1, 2, 3, 7, 14, 21, 30]:
                    future_df[f'lag_{lag}'] = last_quantity
                
                # Add rolling and other features with default values
                for window in [3, 7, 14, 30]:
                    future_df[f'rolling_mean_{window}'] = last_quantity
                    future_df[f'rolling_std_{window}'] = 0
                    future_df[f'rolling_median_{window}'] = last_quantity
                
                for span in [3, 7, 14, 30]:
                    future_df[f'ewm_mean_{span}'] = last_quantity
                
                future_df['trend_3'] = 0
                future_df['trend_7'] = 0
                
                # Select and order features to match training
                available_features = [col for col in feature_columns if col in future_df.columns]
                missing_features = [col for col in feature_columns if col not in future_df.columns]
                
                if missing_features:
                    for feature in missing_features:
                        future_df[feature] = 0
                
                X_future = future_df[feature_columns].fillna(0)
                
                # Make predictions
                future_pred = ensemble_model.predict(X_future)
                future_pred = np.maximum(future_pred, 0)  # Ensure non-negative
                
                predictions[period_name] = pd.DataFrame({
                    'Date': dates,
                    'RouteCode': route_code,
                    'ItemCode': item_code,
                    'Predicted_Demand': future_pred
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making future predictions: {str(e)}")
            return {}

    def save_outputs(self, route_code: str, item_code: str, model_results: Dict,
                    ensemble_result: Dict, future_predictions: Dict, data_dict: Dict):
        """Save all outputs to files"""
        combo_dir = self.get_combo_dir(route_code, item_code)
        
        try:
            # Save model
            model_path = os.path.join(combo_dir, 'models', 'ensemble_model.joblib')
            joblib.dump(ensemble_result, model_path)
            
            # Save predictions to Excel
            output_path = os.path.join(combo_dir, 'outputs', 'predictions.xlsx')
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                if 'daily' in future_predictions:
                    future_predictions['daily'].to_excel(writer, sheet_name='Daily_Forecasts', index=False)
                if 'weekly' in future_predictions:
                    future_predictions['weekly'].to_excel(writer, sheet_name='Weekly_Forecasts', index=False)
                if 'monthly' in future_predictions:
                    future_predictions['monthly'].to_excel(writer, sheet_name='Monthly_Forecasts', index=False)
            
            # Save evaluation report
            eval_path = os.path.join(combo_dir, 'evaluation', 'evaluation_report.xlsx')
            eval_data = []
            for model_name, result in model_results.items():
                if 'metrics' in result:
                    metrics = result['metrics']
                    eval_data.append({
                        'Model': model_name,
                        'MAE': metrics['MAE'],
                        'RMSE': metrics['RMSE'],
                        'MAPE': metrics['MAPE'],
                        'R2': metrics['R2'],
                        'Best_Params': str(result.get('best_params', {}))
                    })
            
            if eval_data:
                eval_df = pd.DataFrame(eval_data)
                eval_df.to_excel(eval_path, index=False)
            
            # Store for consolidated reports
            for model_name, result in model_results.items():
                if 'metrics' in result:
                    self.all_evaluations.append({
                        'RouteCode': route_code,
                        'ItemCode': item_code,
                        'Model': model_name,
                        **result['metrics'],
                        'Best_Params': str(result.get('best_params', {}))
                    })
            
            # Store forecasts for consolidation
            if 'daily' in future_predictions:
                self.all_daily_forecasts.append(future_predictions['daily'])
            if 'weekly' in future_predictions:
                self.all_weekly_forecasts.append(future_predictions['weekly'])
            if 'monthly' in future_predictions:
                self.all_monthly_forecasts.append(future_predictions['monthly'])
            
            logger.info(f"Outputs saved for {route_code}-{item_code}")
            
        except Exception as e:
            logger.error(f"Error saving outputs: {str(e)}")

    def save_consolidated_reports(self):
        """Save consolidated reports across all combinations"""
        try:
            # Save consolidated forecasts
            if self.all_daily_forecasts:
                consolidated_daily = pd.concat(self.all_daily_forecasts, ignore_index=True)
                consolidated_daily.to_csv(os.path.join(self.output_dir, 'merged_daily_forecasts.csv'), index=False)
            
            if self.all_weekly_forecasts:
                consolidated_weekly = pd.concat(self.all_weekly_forecasts, ignore_index=True)
                consolidated_weekly.to_csv(os.path.join(self.output_dir, 'merged_weekly_forecasts.csv'), index=False)
            
            if self.all_monthly_forecasts:
                consolidated_monthly = pd.concat(self.all_monthly_forecasts, ignore_index=True)
                consolidated_monthly.to_csv(os.path.join(self.output_dir, 'merged_monthly_forecasts.csv'), index=False)
            
            # Save training summary
            if self.all_evaluations:
                training_summary = pd.DataFrame(self.all_evaluations)
                training_summary.to_csv(os.path.join(self.output_dir, 'training_summary.csv'), index=False)
            
            logger.info("Consolidated reports saved")
            
        except Exception as e:
            logger.error(f"Error saving consolidated reports: {str(e)}")

    def run_pipeline(self) -> Dict:
        """Run the complete modular pipeline"""
        logger.info("="*60)
        logger.info("STARTING MODULAR DEMAND FORECASTING PIPELINE")
        logger.info("="*60)
        
        # Load and prepare data using modular components
        df = self.load_and_prepare_data()
        
        # Get unique combinations
        combinations = df[['RouteCode', 'ItemCode']].drop_duplicates()
        logger.info(f"Processing {len(combinations)} Route-Item combinations")
        
        results = {'successful': 0, 'failed': 0, 'results': []}
        
        for idx, (_, row) in enumerate(combinations.iterrows(), 1):
            route_code = row['RouteCode']
            item_code = row['ItemCode']
            
            logger.info(f"\n--- Processing {idx}/{len(combinations)}: {route_code}-{item_code} ---")
            
            try:
                # Prepare combination data using modular components
                data_dict = self.prepare_combination_data_modular(df, route_code, item_code)
                if not data_dict:
                    results['failed'] += 1
                    continue
                
                # Train models using modular approach
                model_results, ensemble_result = self.train_models(data_dict)
                
                if not model_results or not ensemble_result:
                    logger.error(f"Model training failed for {route_code}-{item_code}")
                    results['failed'] += 1
                    continue
                
                # Generate future predictions
                future_predictions = self.make_future_predictions(
                    ensemble_result, route_code, item_code, data_dict['combo_data']
                )
                
                # Save all outputs
                self.save_outputs(route_code, item_code, model_results, 
                                ensemble_result, future_predictions, data_dict)
                
                results['successful'] += 1
                results['results'].append({
                    'route_code': route_code,
                    'item_code': item_code,
                    'status': 'success'
                })
                
                logger.info(f"✓ Successfully processed {route_code}-{item_code}")
                
            except Exception as e:
                logger.error(f"Error processing {route_code}-{item_code}: {str(e)}")
                results['failed'] += 1
                results['results'].append({
                    'route_code': route_code,
                    'item_code': item_code,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Save consolidated reports
        self.save_consolidated_reports()
        
        logger.info("="*60)
        logger.info("MODULAR PIPELINE COMPLETE")
        logger.info(f"Successful: {results['successful']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info("="*60)
        
        return results


def main():
    """Main execution function using modular approach"""
    # Configuration - using dynamic absolute paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, 'data', 'raw', 'static', 'training_data.csv')
    output_dir = os.path.join(project_root, 'demand_forecasting_outputs')
    
    # Initialize modular pipeline
    pipeline = ModularDemandForecastingPipeline(data_path, output_dir)
    
    # Run the complete pipeline
    results = pipeline.run_pipeline()
    
    # Example of making predictions for a specific combination
    if results['successful'] > 0:
        # Get first successful combination  
        first_result = results['results'][0]
        route_code = first_result['route_code']
        item_code = first_result['item_code']
        
        logger.info(f"\nExample: Predictions available for {route_code}-{item_code}")
    
    return results


if __name__ == "__main__":
    main()