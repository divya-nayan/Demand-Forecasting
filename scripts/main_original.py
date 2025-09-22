import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import os
from datetime import datetime, timedelta
import holidays
from typing import Dict, Tuple, Optional, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemandForecastingPipeline:
    """Production-level demand forecasting pipeline with comprehensive output generation"""
    
    def __init__(self, data_path: str, output_dir: str = 'outputs'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.holidays_set = None
        self.all_evaluations = []
        self.all_historical_predictions = []
        # Add lists to store forecast data
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

    def get_uae_holidays(self, start_year: int, end_year: int) -> set:
        """Get UAE holidays for the given year range"""
        uae_holidays = holidays.country_holidays('AE')
        holiday_dates = []
        for year in range(start_year, end_year + 3):
            for date in uae_holidays.get(year, {}):
                holiday_dates.append(date)
        return set(holiday_dates)

    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        if np.sum(mask) == 0:
            return 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def load_and_validate_data(self) -> pd.DataFrame:
        """Load and perform basic validation on the data"""
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded data: {len(df)} records")
            
            # Basic validation - updated to include new columns
            required_columns = ['TrxDate', 'WarehouseCode', 'WarehouseName', 'RouteCode', 
                              'ItemCode', 'ItemName', 'CategoryName', 'TotalQuantity', 
                              'AvgUnitPrice', 'SalesValue']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert date and handle data types
            df['TrxDate'] = pd.to_datetime(df['TrxDate'])
            df['TotalQuantity'] = pd.to_numeric(df['TotalQuantity'], errors='coerce')
            df['AvgUnitPrice'] = pd.to_numeric(df['AvgUnitPrice'], errors='coerce')
            df['SalesValue'] = pd.to_numeric(df['SalesValue'], errors='coerce')
            
            # Remove invalid records
            initial_len = len(df)
            df = df.dropna(subset=['TotalQuantity'])
            df = df[df['TotalQuantity'] >= 0]
            logger.info(f"Cleaned data: {len(df)} records (removed {initial_len - len(df)} invalid)")
            
            # Setup holidays
            start_year = df['TrxDate'].dt.year.min()
            end_year = df['TrxDate'].dt.year.max()
            self.holidays_set = self.get_uae_holidays(start_year, end_year)
            logger.info(f"Loaded UAE holidays for {start_year}-{end_year + 3}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features without data leakage"""
        df = df.copy()
        
        # Basic temporal features
        df['DayOfWeek'] = df['TrxDate'].dt.dayofweek
        df['Month'] = df['TrxDate'].dt.month
        df['Quarter'] = df['TrxDate'].dt.quarter
        df['DayOfMonth'] = df['TrxDate'].dt.day
        df['WeekOfYear'] = df['TrxDate'].dt.isocalendar().week
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['IsHoliday'] = df['TrxDate'].dt.date.isin(self.holidays_set).astype(int)
        
        # Cyclical encoding for temporal features
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfMonth_sin'] = np.sin(2 * np.pi * df['DayOfMonth'] / 31)
        df['DayOfMonth_cos'] = np.cos(2 * np.pi * df['DayOfMonth'] / 31)
        
        return df

    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'TotalQuantity') -> pd.DataFrame:
        """Create lag features with strict data leakage prevention"""
        df = df.copy()
        
        # STRICT LAG FEATURES - All shifted to prevent leakage
        lags = [1, 2, 3, 7, 14, 21, 30]
        for lag in lags:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # ROLLING STATISTICS - All with proper shifting to prevent leakage
        windows = [3, 7, 14, 30]
        for window in windows:
            # Shift by 1 and then calculate rolling stats to prevent leakage
            shifted_series = df[target_col].shift(1)
            df[f'rolling_mean_{window}'] = shifted_series.rolling(
                window=window, min_periods=max(1, window//2)
            ).mean()
            df[f'rolling_std_{window}'] = shifted_series.rolling(
                window=window, min_periods=max(1, window//2)
            ).std()
            df[f'rolling_median_{window}'] = shifted_series.rolling(
                window=window, min_periods=max(1, window//2)
            ).median()
        
        # EXPONENTIAL WEIGHTED FEATURES - All with proper shifting
        for span in [3, 7, 14, 30]:
            df[f'ewm_mean_{span}'] = df[target_col].shift(1).ewm(
                span=span, min_periods=1
            ).mean()
        
        # TREND FEATURES - All with proper shifting
        def safe_trend(x):
            if len(x) >= 2:
                return np.polyfit(range(len(x)), x, 1)[0]
            return 0
        
        df['trend_3'] = df[target_col].shift(1).rolling(
            window=3, min_periods=2
        ).apply(safe_trend, raw=False)
        
        df['trend_7'] = df[target_col].shift(1).rolling(
            window=7, min_periods=3
        ).apply(safe_trend, raw=False)
        
        return df

    def prepare_combination_data(self, df: pd.DataFrame, route_code: str, item_code: str) -> Optional[Dict]:
        """Prepare data for a specific RouteCode-ItemCode combination"""
        logger.info(f"Processing {route_code}-{item_code}")
        
        # Filter data for the specific combination
        combo_data = df[(df['RouteCode'] == route_code) & (df['ItemCode'] == item_code)].copy()
        
        if len(combo_data) < 120:
            logger.warning(f"Insufficient data ({len(combo_data)} records). Skipping.")
            return None
        
        # Sort by date to maintain temporal order
        combo_data = combo_data.sort_values('TrxDate').reset_index(drop=True)
        
        # Create features
        combo_data = self.create_time_features(combo_data)
        combo_data = self.create_lag_features(combo_data)
        
        # Time-based split - STRICT temporal split to prevent leakage
        total_days = len(combo_data)
        test_size = max(30, min(90, int(total_days * 0.25)))
        
        # Split data maintaining temporal order
        train_data = combo_data.iloc[:-test_size].copy()
        test_data = combo_data.iloc[-test_size:].copy()
        
        if len(train_data) < 60:
            logger.warning(f"Insufficient training data ({len(train_data)} records). Skipping.")
            return None
        
        # Handle outliers using ONLY training data statistics
        Q99 = train_data['TotalQuantity'].quantile(0.99)
        Q01 = train_data['TotalQuantity'].quantile(0.01)
        train_data['TotalQuantity'] = np.clip(train_data['TotalQuantity'], Q01, Q99)
        test_data['TotalQuantity'] = np.clip(test_data['TotalQuantity'], Q01, Q99)
        
        # Define feature columns (excluding metadata and target)
        feature_columns = [col for col in combo_data.columns 
                          if col not in ['TrxDate', 'WarehouseCode', 'WarehouseName', 'RouteCode', 
                                       'ItemCode', 'ItemName', 'CategoryName', 'TotalQuantity',
                                       'AvgUnitPrice', 'SalesValue']]
        
        # For training data, only use rows where we have sufficient lag features
        min_features_required = int(len(feature_columns) * 0.7)
        train_mask = train_data[feature_columns].count(axis=1) >= min_features_required
        train_data = train_data[train_mask].copy()
        
        if len(train_data) < 45:
            logger.warning(f"Insufficient clean training data ({len(train_data)} records). Skipping.")
            return None
        
        # Fill remaining NaNs using ONLY training data statistics
        train_feature_medians = train_data[feature_columns].median()
        
        # Forward fill then use training medians for any remaining NaNs
        train_data[feature_columns] = train_data[feature_columns].fillna(method='ffill').fillna(train_feature_medians)
        test_data[feature_columns] = test_data[feature_columns].fillna(method='ffill').fillna(train_feature_medians)
        
        # Final cleanup
        train_data = train_data.dropna(subset=feature_columns)
        test_data = test_data.dropna(subset=feature_columns)
        
        if len(train_data) < 30 or len(test_data) < 10:
            logger.warning(f"Insufficient data after cleaning. Train: {len(train_data)}, Test: {len(test_data)}")
            return None
        
        X_train = train_data[feature_columns].copy()
        y_train = train_data['TotalQuantity'].copy()
        X_test = test_data[feature_columns].copy()
        y_test = test_data['TotalQuantity'].copy()
        
        logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_data': train_data,
            'test_data': test_data,
            'feature_columns': feature_columns,
            'combo_data': combo_data
        }

    def get_model_configs(self) -> Dict:
        """Define optimized model configurations for Optuna"""
        return {
            'RandomForest': {
                'model_class': RandomForestRegressor,
                'param_space': {
                    'n_estimators': (50, 200),
                    'max_depth': (5, 20),
                    'min_samples_split': (5, 20),
                    'min_samples_leaf': (2, 10),
                    'max_features': ['sqrt', 'log2', 0.8],
                    'bootstrap': [True, False],
                    'random_state': 42
                }
            },
            'XGBoost': {
                'model_class': XGBRegressor,
                'param_space': {
                    'n_estimators': (50, 300),
                    'max_depth': (3, 10),
                    'learning_rate': (0.01, 0.3),
                    'subsample': (0.7, 1.0),
                    'colsample_bytree': (0.7, 1.0),
                    'reg_alpha': (0.0, 2.0),
                    'reg_lambda': (0.0, 2.0),
                    'random_state': 42,
                    'verbosity': 0
                }
            },
            'LightGBM': {
                'model_class': LGBMRegressor,
                'param_space': {
                    'n_estimators': (50, 300),
                    'max_depth': (3, 15),
                    'learning_rate': (0.01, 0.3),
                    'num_leaves': (10, 100),
                    'feature_fraction': (0.7, 1.0),
                    'bagging_fraction': (0.7, 1.0),
                    'reg_alpha': (0.0, 2.0),
                    'reg_lambda': (0.0, 2.0),
                    'random_state': 42,
                    'verbosity': -1
                }
            },
            'ElasticNet': {
                'model_class': ElasticNet,
                'param_space': {
                    'alpha': (0.001, 10.0),
                    'l1_ratio': (0.1, 0.9),
                    'max_iter': 2000,
                    'random_state': 42
                }
            },
            'Ridge': {
                'model_class': Ridge,
                'param_space': {
                    'alpha': (0.001, 100.0),
                    'random_state': 42
                }
            }
        }

    def optimize_model(self, model_name: str, model_config: Dict, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Optimize model hyperparameters using Optuna with proper cross-validation"""
        
        def objective(trial):
            params = {}
            model_class = model_config['model_class']
            param_space = model_config['param_space']
            
            for param_name, param_config in param_space.items():
                if isinstance(param_config, tuple) and len(param_config) == 2:
                    if isinstance(param_config[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_config[0], param_config[1])
                    elif isinstance(param_config[0], float):
                        params[param_name] = trial.suggest_float(param_name, param_config[0], param_config[1])
                elif isinstance(param_config, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
                else:
                    params[param_name] = param_config
            
            # Time series cross-validation
            n_splits = min(5, len(X_train) // 30)
            if n_splits < 3:
                n_splits = 3
            
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=len(X_train) // (n_splits + 1))
            cv_scores = []
            
            try:
                for train_idx, val_idx in tscv.split(X_train):
                    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    model = model_class(**params)
                    model.fit(X_cv_train, y_cv_train)
                    y_pred = model.predict(X_cv_val)
                    
                    mae = mean_absolute_error(y_cv_val, y_pred)
                    cv_scores.append(mae)
                
                return np.mean(cv_scores)
                
            except Exception as e:
                logger.warning(f"Error in trial for {model_name}: {str(e)}")
                return float('inf')
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        study.optimize(objective, n_trials=100, timeout=300, show_progress_bar=False)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }

    def train_models(self, data_dict: Dict) -> Tuple[Dict, object]:
        """Train and evaluate all models"""
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        logger.info("Training ensemble models...")
        
        # Scale features using RobustScaler (fit only on training data)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        model_configs = self.get_model_configs()
        model_results = {}
        
        for model_name, model_config in model_configs.items():
            try:
                logger.info(f"Optimizing {model_name}...")
                
                optimization_result = self.optimize_model(model_name, model_config, X_train_scaled, y_train)
                
                # Train final model with best parameters
                best_params = optimization_result['best_params']
                model = model_config['model_class'](**best_params)
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                
                # Ensure non-negative predictions
                y_train_pred = np.maximum(y_train_pred, 0)
                y_test_pred = np.maximum(y_test_pred, 0)
                
                # Calculate metrics
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_mape = self.calculate_mape(y_train, y_train_pred)
                test_mape = self.calculate_mape(y_test, y_test_pred)
                
                model_results[model_name] = {
                    'model': model,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mape': train_mape,
                    'test_mape': test_mape,
                    'best_params': best_params,
                    'cv_score': optimization_result['best_score'],
                    'n_trials': optimization_result['n_trials'],
                    'train_predictions': y_train_pred,
                    'test_predictions': y_test_pred
                }
                
                logger.info(f"{model_name}: Test MAE={test_mae:.4f}, R²={test_r2:.4f}, MAPE={test_mape:.2f}%")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        return model_results, scaler

    def create_ensemble(self, model_results: Dict, X_train_scaled: pd.DataFrame, 
                       X_test_scaled: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Dict:
        """Create weighted ensemble based on validation performance"""
        if not model_results:
            return None
        
        # Calculate weights based on inverse of CV scores and test performance
        weights = {}
        total_weight = 0
        
        for model_name, results in model_results.items():
            cv_score = results['cv_score']
            test_mae = results['test_mae']
            
            # Combine CV and test performance
            combined_score = 0.7 * cv_score + 0.3 * test_mae
            weight = 1 / (combined_score + 1e-8)
            
            weights[model_name] = weight
            total_weight += weight
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_weight
        
        logger.info(f"Ensemble weights: {weights}")
        
        # Create ensemble predictions
        train_ensemble_pred = np.zeros(len(y_train))
        test_ensemble_pred = np.zeros(len(y_test))
        
        for model_name, results in model_results.items():
            train_ensemble_pred += weights[model_name] * results['train_predictions']
            test_ensemble_pred += weights[model_name] * results['test_predictions']
        
        # Calculate ensemble metrics
        train_mae = mean_absolute_error(y_train, train_ensemble_pred)
        test_mae = mean_absolute_error(y_test, test_ensemble_pred)
        train_r2 = r2_score(y_train, train_ensemble_pred)
        test_r2 = r2_score(y_test, test_ensemble_pred)
        test_mape = self.calculate_mape(y_test, test_ensemble_pred)
        
        logger.info(f"Ensemble Test: MAE={test_mae:.4f}, R²={test_r2:.4f}, MAPE={test_mape:.2f}%")
        
        return {
            'weights': weights,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mape': test_mape,
            'train_predictions': train_ensemble_pred,
            'test_predictions': test_ensemble_pred
        }

    def create_evaluation_dataframe(self, route_code: str, item_code: str, model_results: Dict, 
                                  ensemble_info: Dict, train_data: pd.DataFrame, test_data: pd.DataFrame,
                                  train_ensemble_pred: np.array, test_ensemble_pred: np.array) -> pd.DataFrame:
        """Create comprehensive evaluation dataframe for the combination"""
        evaluation_data = []
        
        # Add individual model evaluations
        for model_name, results in model_results.items():
            train_mse = mean_squared_error(train_data['TotalQuantity'], results['train_predictions'])
            test_mse = mean_squared_error(test_data['TotalQuantity'], results['test_predictions'])
            
            evaluation_data.append({
                'RouteCode': route_code,
                'ItemCode': item_code,
                'Model': model_name,
                'Model_Type': 'Individual',
                'Train_Size': len(train_data),
                'Test_Size': len(test_data),
                'Train_MAE': results['train_mae'],
                'Test_MAE': results['test_mae'],
                'Train_MSE': train_mse,
                'Test_MSE': test_mse,
                'Train_RMSE': results['train_rmse'],
                'Test_RMSE': results['test_rmse'],
                'Train_R2': results['train_r2'],
                'Test_R2': results['test_r2'],
                'Train_MAPE': results['train_mape'],
                'Test_MAPE': results['test_mape'],
                'Generalization_Gap': abs(results['train_r2'] - results['test_r2']),
                'Ensemble_Weight': ensemble_info['weights'].get(model_name, 0),
                'Best_Parameters': str(results['best_params']),
                'Training_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Add ensemble evaluation
        train_mse_ensemble = mean_squared_error(train_data['TotalQuantity'], train_ensemble_pred)
        test_mse_ensemble = mean_squared_error(test_data['TotalQuantity'], test_ensemble_pred)
        train_rmse_ensemble = np.sqrt(train_mse_ensemble)
        test_rmse_ensemble = np.sqrt(test_mse_ensemble)
        train_mape_ensemble = self.calculate_mape(train_data['TotalQuantity'], train_ensemble_pred)
        
        evaluation_data.append({
            'RouteCode': route_code,
            'ItemCode': item_code,
            'Model': 'ENSEMBLE',
            'Model_Type': 'Ensemble',
            'Train_Size': len(train_data),
            'Test_Size': len(test_data),
            'Train_MAE': ensemble_info['train_mae'],
            'Test_MAE': ensemble_info['test_mae'],
            'Train_MSE': train_mse_ensemble,
            'Test_MSE': test_mse_ensemble,
            'Train_RMSE': train_rmse_ensemble,
            'Test_RMSE': test_rmse_ensemble,
            'Train_R2': ensemble_info['train_r2'],
            'Test_R2': ensemble_info['test_r2'],
            'Train_MAPE': train_mape_ensemble,
            'Test_MAPE': ensemble_info['test_mape'],
            'Generalization_Gap': abs(ensemble_info['train_r2'] - ensemble_info['test_r2']),
            'Ensemble_Weight': 1.0,
            'Best_Parameters': str(ensemble_info['weights']),
            'Training_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        return pd.DataFrame(evaluation_data)

    def generate_future_dates(self, last_date, periods):
        """Generate future dates for daily prediction"""
        return pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')

    def create_future_features(self, future_dates, last_known_values):
        """Create features for future prediction dates"""
        future_df = pd.DataFrame({'TrxDate': future_dates})
        
        # Temporal features
        future_df['DayOfWeek'] = future_df['TrxDate'].dt.dayofweek
        future_df['Month'] = future_df['TrxDate'].dt.month
        future_df['Quarter'] = future_df['TrxDate'].dt.quarter
        future_df['DayOfMonth'] = future_df['TrxDate'].dt.day
        future_df['WeekOfYear'] = future_df['TrxDate'].dt.isocalendar().week
        future_df['IsWeekend'] = (future_df['DayOfWeek'] >= 5).astype(int)
        future_df['IsHoliday'] = future_df['TrxDate'].dt.date.isin(self.holidays_set).astype(int)
        
        # Cyclical encoding
        future_df['DayOfWeek_sin'] = np.sin(2 * np.pi * future_df['DayOfWeek'] / 7)
        future_df['DayOfWeek_cos'] = np.cos(2 * np.pi * future_df['DayOfWeek'] / 7)
        future_df['Month_sin'] = np.sin(2 * np.pi * future_df['Month'] / 12)
        future_df['Month_cos'] = np.cos(2 * np.pi * future_df['Month'] / 12)
        future_df['DayOfMonth_sin'] = np.sin(2 * np.pi * future_df['DayOfMonth'] / 31)
        future_df['DayOfMonth_cos'] = np.cos(2 * np.pi * future_df['DayOfMonth'] / 31)
        
        # Initialize lag and rolling features with last known values
        for col in last_known_values:
            if col.startswith(('lag_', 'rolling_', 'ewm_', 'trend_')):
                future_df[col] = last_known_values[col]
        
        return future_df

    def aggregate_to_weekly(self, daily_df):
        """Aggregate daily predictions to weekly"""
        weekly_df = daily_df.copy()
        weekly_df['WeekStart'] = weekly_df['TrxDate'] - pd.to_timedelta(weekly_df['TrxDate'].dt.dayofweek, unit='d')
        weekly_df = weekly_df.groupby('WeekStart').agg({
            'WarehouseCode': 'first',
            'WarehouseName': 'first',
            'RouteCode': 'first',
            'ItemCode': 'first',
            'ItemName': 'first',
            'CategoryName': 'first',
            'Predicted_Quantity': 'sum',
            'AvgUnitPrice': 'mean',
            'SalesValue': 'sum'
        }).reset_index()
        weekly_df.rename(columns={'WeekStart': 'TrxDate'}, inplace=True)
        weekly_df['Prediction_Type'] = 'Weekly'
        return weekly_df

    def aggregate_to_monthly(self, daily_df):
        """Aggregate daily predictions to monthly"""
        monthly_df = daily_df.copy()
        monthly_df['MonthStart'] = monthly_df['TrxDate'].dt.to_period('M').dt.to_timestamp()
        monthly_df = monthly_df.groupby('MonthStart').agg({
            'WarehouseCode': 'first',
            'WarehouseName': 'first',
            'RouteCode': 'first',
            'ItemCode': 'first',
            'ItemName': 'first',
            'CategoryName': 'first',
            'Predicted_Quantity': 'sum',
            'AvgUnitPrice': 'mean',
            'SalesValue': 'sum'
        }).reset_index()
        monthly_df.rename(columns={'MonthStart': 'TrxDate'}, inplace=True)
        monthly_df['Prediction_Type'] = 'Monthly'
        return monthly_df

    def make_future_predictions(self, model_package: Dict, route_code: str, item_code: str, 
                               combo_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate daily predictions for next 3 months and aggregate to weekly/monthly"""
        logger.info(f"Generating future predictions...")
        
        # Get last known values
        last_row = combo_data.iloc[-1]
        last_date = last_row['TrxDate']
        
        last_known_values = {}
        for col in model_package['feature_columns']:
            if col in combo_data.columns:
                last_known_values[col] = last_row[col]
            else:
                last_known_values[col] = 0
        
        # Generate daily predictions for next 90 days
        future_days = 90
        daily_dates = self.generate_future_dates(last_date, future_days)
        daily_features = self.create_future_features(daily_dates, last_known_values)
        daily_features_aligned = daily_features.reindex(columns=model_package['feature_columns'], fill_value=0)
        
        # Transform features
        daily_scaled = model_package['scaler'].transform(daily_features_aligned)
        
        # Ensemble prediction for daily
        daily_predictions = np.zeros(len(daily_dates))
        for model_name, weight in model_package['ensemble_weights'].items():
            model = model_package['models'][model_name]
            daily_predictions += weight * model.predict(daily_scaled)
        
        daily_pred_df = pd.DataFrame({
            'TrxDate': daily_dates,
            'WarehouseCode': last_row['WarehouseCode'],
            'WarehouseName': last_row['WarehouseName'],
            'RouteCode': route_code,
            'ItemCode': item_code,
            'ItemName': last_row['ItemName'],
            'CategoryName': last_row['CategoryName'],
            'Predicted_Quantity': np.maximum(daily_predictions, 0),
            'AvgUnitPrice': last_row['AvgUnitPrice'],
            'SalesValue': np.maximum(daily_predictions, 0) * last_row['AvgUnitPrice'],
            'Prediction_Type': 'Daily'
        })
        
        # Aggregate to weekly and monthly
        weekly_pred_df = self.aggregate_to_weekly(daily_pred_df)
        monthly_pred_df = self.aggregate_to_monthly(daily_pred_df)
        
        logger.info(f"Daily predictions: {len(daily_pred_df)} days")
        logger.info(f"Weekly predictions: {len(weekly_pred_df)} weeks")
        logger.info(f"Monthly predictions: {len(monthly_pred_df)} months")
        
        return daily_pred_df, weekly_pred_df, monthly_pred_df

    def save_combination_outputs(self, route_code: str, item_code: str, model_results: Dict, 
                               ensemble_info: Dict, scaler: object, feature_columns: List[str],
                               train_data: pd.DataFrame, test_data: pd.DataFrame,
                               train_ensemble_pred: np.array, test_ensemble_pred: np.array,
                               combo_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Save all outputs for a combination and return evaluation and prediction dataframes"""
        
        # Create combination directory
        combo_dir = self.get_combo_dir(route_code, item_code)
        
        # Prepare model package for saving
        model_package = {
            'models': {name: results['model'] for name, results in model_results.items()},
            'ensemble_weights': ensemble_info['weights'],
            'scaler': scaler,
            'feature_columns': feature_columns,
            'metrics': {
                'train_mae': ensemble_info['train_mae'],
                'test_mae': ensemble_info['test_mae'],
                'train_r2': ensemble_info['train_r2'],
                'test_r2': ensemble_info['test_r2'],
                'test_mape': ensemble_info['test_mape']
            }
        }
        
        # Save model package
        model_filename = os.path.join(combo_dir, "models", "ensemble_model.joblib")
        joblib.dump(model_package, model_filename, compress=3)
        logger.info(f"Ensemble model saved: {model_filename}")
        
        # Create evaluation dataframe
        evaluation_df = self.create_evaluation_dataframe(
            route_code, item_code, model_results, ensemble_info,
            train_data, test_data, train_ensemble_pred, test_ensemble_pred
        )
        
        # Save individual evaluation file for this combination
        individual_eval_path = os.path.join(combo_dir, "evaluation", "model_evaluation.csv")
        evaluation_df.to_csv(individual_eval_path, index=False)
        logger.info(f"Individual evaluation saved: {individual_eval_path}")
        
        # Prepare historical predictions - Include ALL columns for readability
        hist_columns = ['TrxDate', 'WarehouseCode', 'WarehouseName', 'RouteCode', 'ItemCode', 
                       'ItemName', 'CategoryName', 'TotalQuantity', 'AvgUnitPrice', 'SalesValue']
        
        train_pred_df = train_data[hist_columns].copy()
        train_pred_df['Predicted'] = train_ensemble_pred
        train_pred_df['DataSplit'] = 'Train'
        
        test_pred_df = test_data[hist_columns].copy()
        test_pred_df['Predicted'] = test_ensemble_pred
        test_pred_df['DataSplit'] = 'Test'
        
        historical_pred_df = pd.concat([train_pred_df, test_pred_df])
        
        # Generate future predictions
        daily_pred_df, weekly_pred_df, monthly_pred_df = self.make_future_predictions(
            model_package, route_code, item_code, combo_data
        )
        
        # Create metrics dataframe for Excel output
        metrics_data = []
        for model_name, results in model_results.items():
            metrics_data.append({
                'Model': model_name,
                'Train_MAE': results['train_mae'],
                'Test_MAE': results['test_mae'],
                'Train_R2': results['train_r2'],
                'Test_R2': results['test_r2'],
                'Generalization_Gap': abs(results['train_r2'] - results['test_r2']),
                'Parameters': str(results['best_params'])
            })
        
        # Add ensemble metrics
        metrics_data.append({
            'Model': 'ENSEMBLE',
            'Train_MAE': ensemble_info['train_mae'],
            'Test_MAE': ensemble_info['test_mae'],
            'Train_R2': ensemble_info['train_r2'],
            'Test_R2': ensemble_info['test_r2'],
            'Generalization_Gap': abs(ensemble_info['train_r2'] - ensemble_info['test_r2']),
            'Parameters': str(ensemble_info['weights'])
        })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Save all outputs to Excel
        output_path = os.path.join(combo_dir, "outputs", "predictions.xlsx")
        with pd.ExcelWriter(output_path) as writer:
            historical_pred_df.to_excel(writer, sheet_name='Historical_Predictions', index=False)
            daily_pred_df.to_excel(writer, sheet_name='Future_Daily', index=False)
            weekly_pred_df.to_excel(writer, sheet_name='Future_Weekly', index=False)
            monthly_pred_df.to_excel(writer, sheet_name='Future_Monthly', index=False)
            metrics_df.to_excel(writer, sheet_name='Model_Metrics', index=False)
            evaluation_df.to_excel(writer, sheet_name='Detailed_Evaluation', index=False)
        
        logger.info(f"All outputs saved to: {output_path}")
        
        return evaluation_df, historical_pred_df, daily_pred_df, weekly_pred_df, monthly_pred_df

    def process_combination(self, df: pd.DataFrame, route_code: str, item_code: str) -> Optional[Dict]:
        """Process a single RouteCode-ItemCode combination"""
        
        # Prepare data
        data_dict = self.prepare_combination_data(df, route_code, item_code)
        if not data_dict:
            return None
        
        # Train models
        model_results, scaler = self.train_models(data_dict)
        if not model_results:
            logger.error("No models trained successfully")
            return None
        
        # Create ensemble
        X_train_scaled = scaler.transform(data_dict['X_train'])
        X_test_scaled = scaler.transform(data_dict['X_test'])
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=data_dict['feature_columns'], 
                                     index=data_dict['X_train'].index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=data_dict['feature_columns'], 
                                    index=data_dict['X_test'].index)
        
        ensemble_info = self.create_ensemble(
            model_results, X_train_scaled, X_test_scaled, 
            data_dict['y_train'], data_dict['y_test']
        )
        
        if not ensemble_info:
            logger.error("Failed to create ensemble")
            return None
        
        # Save outputs and get all dataframes
        evaluation_df, historical_pred_df, daily_pred_df, weekly_pred_df, monthly_pred_df = self.save_combination_outputs(
            route_code, item_code, model_results, ensemble_info, scaler, 
            data_dict['feature_columns'], data_dict['train_data'], data_dict['test_data'],
            ensemble_info['train_predictions'], ensemble_info['test_predictions'],
            data_dict['combo_data']
        )
        
        # Store for later merging
        self.all_evaluations.append(evaluation_df)
        self.all_historical_predictions.append(historical_pred_df)
        self.all_daily_forecasts.append(daily_pred_df)
        self.all_weekly_forecasts.append(weekly_pred_df)
        self.all_monthly_forecasts.append(monthly_pred_df)
        
        return {
            'route_code': route_code,
            'item_code': item_code,
            'test_mae': ensemble_info['test_mae'],
            'test_r2': ensemble_info['test_r2'],
            'test_mape': ensemble_info['test_mape'],
            'data_points': len(data_dict['train_data']) + len(data_dict['test_data']),
            'evaluation_df': evaluation_df,
            'historical_predictions_df': historical_pred_df
        }

    def create_merged_outputs(self):
        """Create merged evaluation and forecast files"""
        if not self.all_evaluations:
            logger.warning("No evaluation data to merge")
            return
        
        # Create merged evaluation file
        merged_evaluation_df = pd.concat(self.all_evaluations, ignore_index=True)
        merged_evaluation_path = os.path.join(self.output_dir, 'evaluation', 'merged_model_evaluation.csv')
        merged_evaluation_df.to_csv(merged_evaluation_path, index=False)
        logger.info(f"Merged evaluation file saved: {merged_evaluation_path}")
        
        # Create comprehensive evaluation Excel file
        evaluation_excel_path = os.path.join(self.output_dir, 'evaluation', 'comprehensive_model_evaluation.xlsx')
        with pd.ExcelWriter(evaluation_excel_path) as writer:
            # All evaluations
            merged_evaluation_df.to_excel(writer, sheet_name='All_Model_Evaluations', index=False)
            
            # Summary by combination (ensemble only)
            ensemble_summary = merged_evaluation_df[
                merged_evaluation_df['Model'] == 'ENSEMBLE'
            ][['RouteCode', 'ItemCode', 'Train_Size', 'Test_Size', 'Test_MAE', 
               'Test_RMSE', 'Test_R2', 'Test_MAPE', 'Generalization_Gap']].copy()
            ensemble_summary.to_excel(writer, sheet_name='Ensemble_Summary', index=False)
            
            # Summary by model type
            model_summary = merged_evaluation_df.groupby(['Model']).agg({
                'Test_MAE': ['mean', 'std', 'min', 'max'],
                'Test_R2': ['mean', 'std', 'min', 'max'],
                'Test_MAPE': ['mean', 'std', 'min', 'max'],
                'Generalization_Gap': ['mean', 'std', 'min', 'max']
            }).round(4)
            model_summary.columns = ['_'.join(col).strip() for col in model_summary.columns.values]
            model_summary = model_summary.reset_index()
            model_summary.to_excel(writer, sheet_name='Model_Type_Summary', index=False)
            
            # Best performing combinations (by ensemble R2)
            best_combinations = ensemble_summary.nlargest(20, 'Test_R2')
            best_combinations.to_excel(writer, sheet_name='Top_20_Combinations', index=False)
            
            # Worst performing combinations (by ensemble R2)
            worst_combinations = ensemble_summary.nsmallest(20, 'Test_R2')
            worst_combinations.to_excel(writer, sheet_name='Bottom_20_Combinations', index=False)
        
        logger.info(f"Comprehensive evaluation Excel saved: {evaluation_excel_path}")
        
        # Create merged historical predictions file
        if self.all_historical_predictions:
            merged_historical_df = pd.concat(self.all_historical_predictions, ignore_index=True)
            merged_historical_path = os.path.join(self.output_dir, 'merged_historical_predictions.csv')
            merged_historical_df.to_csv(merged_historical_path, index=False)
            logger.info(f"Merged historical predictions saved: {merged_historical_path}")
            
            # Also save as Excel with additional analysis
            historical_excel_path = os.path.join(self.output_dir, 'historical_predictions_analysis.xlsx')
            with pd.ExcelWriter(historical_excel_path) as writer:
                # All historical predictions
                merged_historical_df.to_excel(writer, sheet_name='All_Historical_Predictions', index=False)
                
                # Summary statistics by combination
                historical_summary = merged_historical_df.groupby(['RouteCode', 'ItemCode', 'DataSplit']).agg({
                    'TotalQuantity': ['mean', 'std', 'min', 'max', 'sum'],
                    'Predicted': ['mean', 'std', 'min', 'max', 'sum'],
                    'SalesValue': ['mean', 'std', 'min', 'max', 'sum']
                }).round(4)
                historical_summary.columns = ['_'.join(col).strip() for col in historical_summary.columns.values]
                historical_summary = historical_summary.reset_index()
                historical_summary.to_excel(writer, sheet_name='Summary_by_Combination', index=False)
                
                # Calculate prediction errors
                error_analysis = merged_historical_df.copy()
                error_analysis['Error'] = error_analysis['TotalQuantity'] - error_analysis['Predicted']
                error_analysis['Absolute_Error'] = np.abs(error_analysis['Error'])
                error_analysis['Percentage_Error'] = np.where(
                    error_analysis['TotalQuantity'] != 0,
                    (error_analysis['Error'] / error_analysis['TotalQuantity']) * 100,
                    0
                )
                error_analysis['Absolute_Percentage_Error'] = np.abs(error_analysis['Percentage_Error'])
                
                # Error summary by combination and split
                error_summary = error_analysis.groupby(['RouteCode', 'ItemCode', 'DataSplit']).agg({
                    'Error': ['mean', 'std'],
                    'Absolute_Error': ['mean', 'max'],
                    'Absolute_Percentage_Error': ['mean', 'max']
                }).round(4)
                error_summary.columns = ['_'.join(col).strip() for col in error_summary.columns.values]
                error_summary = error_summary.reset_index()
                error_summary.to_excel(writer, sheet_name='Error_Analysis', index=False)
            
            logger.info(f"Historical predictions analysis Excel saved: {historical_excel_path}")
        
        # Create merged daily forecasts file
        if self.all_daily_forecasts:
            merged_daily_df = pd.concat(self.all_daily_forecasts, ignore_index=True)
            merged_daily_path = os.path.join(self.output_dir, 'merged_daily_forecasts.csv')
            merged_daily_df.to_csv(merged_daily_path, index=False)
            logger.info(f"Merged daily forecasts saved: {merged_daily_path}")
        
        # Create merged weekly forecasts file
        if self.all_weekly_forecasts:
            merged_weekly_df = pd.concat(self.all_weekly_forecasts, ignore_index=True)
            merged_weekly_path = os.path.join(self.output_dir, 'merged_weekly_forecasts.csv')
            merged_weekly_df.to_csv(merged_weekly_path, index=False)
            logger.info(f"Merged weekly forecasts saved: {merged_weekly_path}")
        
        # Create merged monthly forecasts file
        if self.all_monthly_forecasts:
            merged_monthly_df = pd.concat(self.all_monthly_forecasts, ignore_index=True)
            merged_monthly_path = os.path.join(self.output_dir, 'merged_monthly_forecasts.csv')
            merged_monthly_df.to_csv(merged_monthly_path, index=False)
            logger.info(f"Merged monthly forecasts saved: {merged_monthly_path}")
        
        # Create comprehensive forecasts Excel file
        if self.all_daily_forecasts and self.all_weekly_forecasts and self.all_monthly_forecasts:
            forecasts_excel_path = os.path.join(self.output_dir, 'comprehensive_forecasts.xlsx')
            with pd.ExcelWriter(forecasts_excel_path) as writer:
                # All forecasts
                merged_daily_df.to_excel(writer, sheet_name='All_Daily_Forecasts', index=False)
                merged_weekly_df.to_excel(writer, sheet_name='All_Weekly_Forecasts', index=False)
                merged_monthly_df.to_excel(writer, sheet_name='All_Monthly_Forecasts', index=False)
                
                # Daily forecast summary
                daily_summary = merged_daily_df.groupby(['RouteCode', 'ItemCode']).agg({
                    'Predicted_Quantity': ['sum', 'mean', 'std', 'min', 'max'],
                    'SalesValue': ['sum', 'mean', 'std', 'min', 'max']
                }).round(4)
                daily_summary.columns = ['_'.join(col).strip() for col in daily_summary.columns.values]
                daily_summary = daily_summary.reset_index()
                daily_summary.to_excel(writer, sheet_name='Daily_Summary_by_Item', index=False)
                
                # Weekly forecast summary
                weekly_summary = merged_weekly_df.groupby(['RouteCode', 'ItemCode']).agg({
                    'Predicted_Quantity': ['sum', 'mean', 'std', 'min', 'max'],
                    'SalesValue': ['sum', 'mean', 'std', 'min', 'max']
                }).round(4)
                weekly_summary.columns = ['_'.join(col).strip() for col in weekly_summary.columns.values]
                weekly_summary = weekly_summary.reset_index()
                weekly_summary.to_excel(writer, sheet_name='Weekly_Summary_by_Item', index=False)
                
                # Monthly forecast summary
                monthly_summary = merged_monthly_df.groupby(['RouteCode', 'ItemCode']).agg({
                    'Predicted_Quantity': ['sum', 'mean', 'std', 'min', 'max'],
                    'SalesValue': ['sum', 'mean', 'std', 'min', 'max']
                }).round(4)
                monthly_summary.columns = ['_'.join(col).strip() for col in monthly_summary.columns.values]
                monthly_summary = monthly_summary.reset_index()
                monthly_summary.to_excel(writer, sheet_name='Monthly_Summary_by_Item', index=False)
            
            logger.info(f"Comprehensive forecasts Excel saved: {forecasts_excel_path}")

    def run_pipeline(self) -> Dict:
        """Run the complete pipeline"""
        logger.info("Starting Production Demand Forecasting Pipeline")
        
        # Load and validate data
        df = self.load_and_validate_data()
        
        # Get unique combinations
        combinations = df[['RouteCode', 'ItemCode']].drop_duplicates()
        logger.info(f"Found {len(combinations)} unique RouteCode-ItemCode combinations")
        
        # Process each combination
        results = []
        successful = 0
        failed = 0
        
        for idx, (_, row) in enumerate(combinations.iterrows(), 1):
            route_code = row['RouteCode']
            item_code = row['ItemCode']
            
            logger.info(f"[{idx}/{len(combinations)}] Processing {route_code}-{item_code}")
            
            try:
                result = self.process_combination(df, route_code, item_code)
                if result:
                    results.append(result)
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error processing {route_code}-{item_code}: {str(e)}")
                failed += 1
                continue
        
        # Create merged outputs
        self.create_merged_outputs()
        
        # Save summary
        if results:
            summary_df = pd.DataFrame(results)
            summary_path = os.path.join(self.output_dir, 'training_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            
            logger.info(f"\nPipeline completed!")
            logger.info(f"Successfully processed: {successful}/{len(combinations)} combinations")
            logger.info(f"Failed: {failed}/{len(combinations)} combinations")
            logger.info(f"Average Test MAE: {summary_df['test_mae'].mean():.4f}")
            logger.info(f"Average Test R²: {summary_df['test_r2'].mean():.4f}")
            logger.info(f"Summary saved: {summary_path}")
            
            # Show evaluation statistics
            if self.all_evaluations:
                merged_eval = pd.concat(self.all_evaluations, ignore_index=True)
                ensemble_eval = merged_eval[merged_eval['Model'] == 'ENSEMBLE']
                
                logger.info(f"\nEvaluation Statistics:")
                logger.info(f"  - Total model evaluations: {len(merged_eval)}")
                logger.info(f"  - Unique combinations: {ensemble_eval['RouteCode'].nunique()} routes × {ensemble_eval['ItemCode'].nunique()} items")
                logger.info(f"  - Best ensemble R²: {ensemble_eval['Test_R2'].max():.4f}")
                logger.info(f"  - Worst ensemble R²: {ensemble_eval['Test_R2'].min():.4f}")
                logger.info(f"  - Median ensemble MAPE: {ensemble_eval['Test_MAPE'].median():.2f}%")
            
            # Show historical predictions statistics
            if self.all_historical_predictions:
                merged_historical = pd.concat(self.all_historical_predictions, ignore_index=True)
                logger.info(f"\nHistorical Predictions Statistics:")
                logger.info(f"  - Total historical predictions: {len(merged_historical):,}")
                logger.info(f"  - Date range: {merged_historical['TrxDate'].min()} to {merged_historical['TrxDate'].max()}")
                logger.info(f"  - Train records: {len(merged_historical[merged_historical['DataSplit'] == 'Train']):,}")
                logger.info(f"  - Test records: {len(merged_historical[merged_historical['DataSplit'] == 'Test']):,}")
            
            # Show forecast statistics
            if self.all_daily_forecasts:
                merged_daily = pd.concat(self.all_daily_forecasts, ignore_index=True)
                logger.info(f"\nForecast Statistics:")
                logger.info(f"  - Total daily forecasts: {len(merged_daily):,}")
                logger.info(f"  - Daily forecast date range: {merged_daily['TrxDate'].min()} to {merged_daily['TrxDate'].max()}")
                logger.info(f"  - Total weekly forecasts: {len(pd.concat(self.all_weekly_forecasts)):,}")
                logger.info(f"  - Total monthly forecasts: {len(pd.concat(self.all_monthly_forecasts)):,}")
        
        return {
            'total_combinations': len(combinations),
            'successful': successful,
            'failed': failed,
            'results': results,
            'output_dir': self.output_dir
        }

    def load_model_for_prediction(self, route_code: str, item_code: str) -> Optional[Dict]:
        """Load trained model for making predictions"""
        model_path = os.path.join(self.output_dir, f"{route_code}_{item_code}", 'models', 'ensemble_model.joblib')
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return None
        
        try:
            model_package = joblib.load(model_path)
            logger.info(f"Loaded model for {route_code}-{item_code}")
            return model_package
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

    def predict_for_combination(self, route_code: str, item_code: str):
        """Load saved model and generate predictions for a specific combination"""
        combo_dir = self.get_combo_dir(route_code, item_code)
        output_path = os.path.join(combo_dir, "outputs", "predictions.xlsx")
        
        if not os.path.exists(output_path):
            logger.error(f"Predictions not found: {output_path}")
            return None
        
        try:
            logger.info(f"Loading predictions from: {output_path}")
            return pd.read_excel(output_path, sheet_name=None)
        except Exception as e:
            logger.error(f"Error loading predictions: {str(e)}")
            return None

# Usage example and main execution
def main():
    """Main execution function"""
    
    # Configuration
    data_path = r'C:\Users\divya\Desktop\Winit\Yaumi_mine\1004\Yaumi\forecasting\static\training_data.csv'
    output_dir = r'original_demand_forecasting_outputs'
    
    # Initialize pipeline
    pipeline = DemandForecastingPipeline(data_path, output_dir)
    
    # Run the complete pipeline
    results = pipeline.run_pipeline()
    
    # Example of making predictions for a specific combination
    if results['successful'] > 0:
        # Get first successful combination
        first_result = results['results'][0]
        route_code = first_result['route_code']
        item_code = first_result['item_code']
        
        logger.info(f"\nGenerating future predictions for {route_code}-{item_code}...")
        future_predictions = pipeline.predict_for_combination(route_code, item_code)
        
        if future_predictions is not None:
            logger.info(f"Available prediction sheets: {list(future_predictions.keys())}")
    
    return results

if __name__ == "__main__":
    main()