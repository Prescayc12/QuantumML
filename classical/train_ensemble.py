"""
Gradient Boosting Ensemble Trainer
Train 12 XGBoost and LightGBM models with meta-learner stacking
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import warnings
import time
import pickle
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

def engineer_features(df):
    """Engineer temporal and static feature"""
    start_time = time.time()
    df_eng = df.copy()
    
    # Verify required column exist for temporal feature
    if 'datetime' not in df_eng.columns or 'latitude' not in df_eng.columns:
        return df_eng
    
    # Parse datetime and sort by location then time
    # Sorting is required for rolling window calculation
    df_eng['datetime_parsed'] = pd.to_datetime(df_eng['datetime'])
    df_eng = df_eng.sort_values(['latitude', 'longitude', 'datetime_parsed'])
    df_eng = df_eng.reset_index(drop=True)
    
    # Temperature feature: range, average, variance
    if 'tmmx' in df.columns and 'tmmn' in df.columns:
        df_eng['temp_range'] = df['tmmx'] - df['tmmn']
        df_eng['temp_avg'] = (df['tmmx'] + df['tmmn']) / 2
        df_eng['temp_variance'] = (df['tmmx'] - df['tmmn']) ** 2
    
    # Humidity feature: range, average, dryness index
    if 'rmax' in df.columns and 'rmin' in df.columns:
        df_eng['humidity_range'] = df['rmax'] - df['rmin']
        df_eng['humidity_avg'] = (df['rmax'] + df['rmin']) / 2
        df_eng['dryness_index'] = 100 - df['rmin']  # Higher value = drier condition
    
    # Vapor pressure deficit interaction with temperature
    # VPD is key fire danger indicator
    if 'vpd' in df.columns and 'tmmx' in df.columns:
        df_eng['vpd_temp'] = df['vpd'] * df['tmmx']
        df_eng['vpd_squared'] = df['vpd'] ** 2
    
    # Wind and energy release component interaction
    # Higher wind + higher ERC = increased fire spread potential
    if 'vs' in df.columns and 'erc' in df.columns:
        df_eng['wind_erc'] = df['vs'] * df['erc']
        df_eng['wind_squared'] = df['vs'] ** 2
    
    # Fuel moisture feature: ratio, difference, average
    # Lower fuel moisture = higher fire risk
    if 'fm100' in df.columns and 'fm1000' in df.columns:
        df_eng['fuel_moisture_ratio'] = df['fm100'] / (df['fm1000'] + 1e-6)
        df_eng['fuel_moisture_diff'] = df['fm1000'] - df['fm100']
        df_eng['fuel_moisture_avg'] = (df['fm100'] + df['fm1000']) / 2
    
    # Drought stress: VPD relative to precipitation
    if 'pr' in df.columns and 'vpd' in df.columns:
        df_eng['drought_stress'] = df['vpd'] / (df['pr'] + 1e-6)
    
    # Heat load from solar radiation and temperature
    if 'srad' in df.columns and 'tmmx' in df.columns:
        df_eng['heat_load'] = df['srad'] * df['tmmx']
    
    # Energy release component squared (amplify high value)
    if 'erc' in df.columns:
        df_eng['erc_squared'] = df['erc'] ** 2
    
    # Fire spread potential from burning index and wind
    if 'bi' in df.columns and 'vs' in df.columns:
        df_eng['fire_spread_potential'] = df['bi'] * df['vs']
    
    # Geographic feature: location and interaction
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df_eng['lat_squared'] = df['latitude'] ** 2
        df_eng['lon_squared'] = df['longitude'] ** 2
        df_eng['lat_lon_interaction'] = df['latitude'] * df['longitude']
    
    # Seasonal feature: cyclical encoding and fire season indicator
    # Use sin/cos to preserve cyclical nature of day of year
    df_eng['day_of_year'] = df_eng['datetime_parsed'].dt.dayofyear
    df_eng['day_sin'] = np.sin(2 * np.pi * df_eng['day_of_year'] / 365.25)
    df_eng['day_cos'] = np.cos(2 * np.pi * df_eng['day_of_year'] / 365.25)
    df_eng['month'] = df_eng['datetime_parsed'].dt.month
    df_eng['is_fire_season'] = ((df_eng['month'] >= 5) & (df_eng['month'] <= 10)).astype(int)
    
    # Temporal rolling feature: capture recent weather trend
    # Group by location to ensure rolling window respect spatial boundary
    print("Computing temporal feature...")
    grouped = df_eng.groupby(['latitude', 'longitude'], group_keys=False)
    
    # Temperature rolling average: 3-day and 7-day window
    df_eng['temp_3day_avg'] = grouped['temp_avg'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df_eng['temp_7day_avg'] = grouped['temp_avg'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # VPD rolling average: capture atmospheric drying trend
    if 'vpd' in df_eng.columns:
        df_eng['vpd_3day_avg'] = grouped['vpd'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df_eng['vpd_7day_avg'] = grouped['vpd'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
    
    # Humidity rolling average
    if 'humidity_avg' in df_eng.columns:
        df_eng['humidity_3day_avg'] = grouped['humidity_avg'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
    
    # Lag feature: previous day value
    df_eng['temp_lag1'] = grouped['temp_avg'].shift(1)
    df_eng['vpd_lag1'] = grouped['vpd'].shift(1) if 'vpd' in df_eng.columns else 0
    df_eng['humidity_lag1'] = grouped['humidity_avg'].shift(1) if 'humidity_avg' in df_eng.columns else 0
    
    # Rate of change: day-to-day delta
    df_eng['temp_change'] = grouped['temp_avg'].diff()
    df_eng['humidity_change'] = grouped['humidity_avg'].diff() if 'humidity_avg' in df_eng.columns else 0
    df_eng['vpd_change'] = grouped['vpd'].diff() if 'vpd' in df_eng.columns else 0
    
    # Fill missing value from lag operation
    lag_cols = ['temp_lag1', 'vpd_lag1', 'humidity_lag1', 'temp_change', 'humidity_change', 'vpd_change']
    for col in lag_cols:
        if col in df_eng.columns:
            df_eng[col] = grouped[col].transform(lambda x: x.fillna(method='bfill').fillna(0))
    
    elapsed = (time.time() - start_time) / 60
    print(f"Feature engineering complete: {elapsed:.1f} min")
    
    df_eng = df_eng.drop(columns=['datetime_parsed', 'day_of_year'], errors='ignore')
    return df_eng

def preprocess_data(df, target_column):
    """Preprocess data for model training"""
    df_processed = df.copy()
    
    # Fill missing value in numeric column with median
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Fill missing value in categorical column with mode
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    # Encode categorical variable to numeric value
    label_encoders = {}
    for col in categorical_columns:
        if col != target_column:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Separate feature from target
    feature_columns = [col for col in df_processed.columns if col != target_column]
    X = df_processed[feature_columns].values
    y = df_processed[target_column].values
    
    # Encode target if categorical
    le_target = None
    if df_processed[target_column].dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    
    # Split into train/val/test with stratification
    # 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    # Standardize feature to zero mean and unit variance
    # Fit only on training set to prevent data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, le_target, label_encoders

def train_xgboost(model_num, X_train, y_train, X_val, y_val, params):
    """Train XGBoost model"""
    print(f"Training XGBoost {model_num}/6...")
    start_time = time.time()
    
    # Calculate class weight to handle imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Initialize XGBoost classifier with early stopping
    model = xgb.XGBClassifier(
        n_estimators=500,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',  # Faster training method
        random_state=42 + model_num,
        eval_metric='logloss',
        n_jobs=-1,  # Use all CPU core
        early_stopping_rounds=50,
        **params
    )
    
    # Train with validation set for early stopping
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Calculate validation metric
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs > 0.5).astype(int)
    
    val_f1 = f1_score(y_val, val_preds)
    val_recall = recall_score(y_val, val_preds)
    val_precision = precision_score(y_val, val_preds, zero_division=0)
    
    elapsed = (time.time() - start_time) / 60
    print(f"  Complete: {elapsed:.1f} min | F1={val_f1:.4f} Recall={val_recall:.4f}")
    
    return {
        'model': model,
        'model_type': 'xgboost',
        'params': params,
        'val_f1': val_f1,
        'val_recall': val_recall,
        'val_precision': val_precision,
        'model_num': model_num
    }

def train_lightgbm(model_num, X_train, y_train, X_val, y_val, params):
    """Train LightGBM model"""
    print(f"Training LightGBM {model_num}/6...")
    start_time = time.time()
    
    # Calculate class weight to handle imbalance
    # LightGBM requires dictionary format for class_weight
    n_samples = len(y_train)
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    class_weight = {0: n_samples / (2 * n_neg), 1: n_samples / (2 * n_pos)}
    
    # Initialize LightGBM classifier with class weighting
    model = lgb.LGBMClassifier(
        n_estimators=500,
        random_state=42 + model_num,
        n_jobs=-1,  # Use all CPU core
        verbosity=-1,  # Suppress output
        class_weight=class_weight,  # Handle class imbalance with proper weighting
        **params
    )
    
    # Train with validation set and early stopping
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
    
    # Calculate validation metric
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs > 0.5).astype(int)
    
    val_f1 = f1_score(y_val, val_preds)
    val_recall = recall_score(y_val, val_preds)
    val_precision = precision_score(y_val, val_preds, zero_division=0)
    
    elapsed = (time.time() - start_time) / 60
    print(f"  Complete: {elapsed:.1f} min | F1={val_f1:.4f} Recall={val_recall:.4f}")
    
    return {
        'model': model,
        'model_type': 'lightgbm',
        'params': params,
        'val_f1': val_f1,
        'val_recall': val_recall,
        'val_precision': val_precision,
        'model_num': model_num
    }

def train_ensemble(X_train, y_train, X_val, y_val):
    """Train ensemble of 6 XGBoost and 6 LightGBM model"""
    print("\nTraining ensemble...")
    all_models = []
    
    # Train 6 XGBoost model with diverse hyperparameter configuration
    # Diversity improves ensemble performance by capturing different pattern
    xgb_configs = [
        {'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 1},
        {'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 3},
        {'max_depth': 10, 'learning_rate': 0.1, 'min_child_weight': 5},
        {'max_depth': 12, 'learning_rate': 0.05, 'min_child_weight': 7},
        {'max_depth': 8, 'learning_rate': 0.15, 'min_child_weight': 2},
        {'max_depth': 10, 'learning_rate': 0.08, 'min_child_weight': 4}
    ]
    
    for i, params in enumerate(xgb_configs, 1):
        model_info = train_xgboost(i, X_train, y_train, X_val, y_val, params)
        all_models.append(model_info)
    
    # Train 6 LightGBM model with diverse hyperparameter configuration
    # LightGBM uses different algorithm than XGBoost for additional diversity
    lgb_configs = [
        {'num_leaves': 31, 'learning_rate': 0.1, 'min_child_samples': 20},
        {'num_leaves': 63, 'learning_rate': 0.05, 'min_child_samples': 30},
        {'num_leaves': 127, 'learning_rate': 0.1, 'min_child_samples': 40},
        {'num_leaves': 255, 'learning_rate': 0.05, 'min_child_samples': 50},
        {'num_leaves': 95, 'learning_rate': 0.08, 'min_child_samples': 25},
        {'num_leaves': 191, 'learning_rate': 0.12, 'min_child_samples': 35}
    ]
    
    for i, params in enumerate(lgb_configs, 1):
        model_info = train_lightgbm(6 + i, X_train, y_train, X_val, y_val, params)
        all_models.append(model_info)
    
    return all_models

def get_predictions(models, X):
    """Get prediction from all model in ensemble"""
    all_preds = []
    for model_info in models:
        # Get probability for positive class (fire)
        probs = model_info['model'].predict_proba(X)[:, 1]
        all_preds.append(probs)
    # Return shape: (num_models, num_samples)
    return np.array(all_preds)

def train_meta_learner(models, X_val, y_val, X_test, y_test):
    """Train meta-learner on base model prediction"""
    print("\nTraining meta-learner...")
    
    # Get prediction from all base model on validation set
    # Meta-learner learn optimal way to combine base model
    val_preds = get_predictions(models, X_val)
    
    # Train logistic regression as meta-learner
    # Input: base model prediction, Output: final prediction
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    meta_learner.fit(val_preds.T, y_val)
    
    # Evaluate stacked ensemble on test set
    test_preds = get_predictions(models, X_test)
    stacked_probs = meta_learner.predict_proba(test_preds.T)[:, 1]
    stacked_preds = (stacked_probs >= 0.42).astype(int)
    
    # Calculate performance metric
    tn, fp, fn, tp = confusion_matrix(y_test, stacked_preds).ravel()
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    balanced_acc = (recall + specificity) / 2
    
    print(f"\nTest set performance:")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    
    return meta_learner, stacked_probs

def save_models(models, meta_learner, X_test, y_test, scaler, le_target, label_encoders, input_size):
    """Save trained model and configuration"""
    print("\nSaving models...")
    
    # Create directory structure for organized storage
    os.makedirs('models', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Save preprocessing configuration for use on new data
    # This ensure new data is transformed identically to training data
    preprocessing_data = {
        'scaler': scaler,
        'le_target': le_target,
        'label_encoders': label_encoders,
        'input_size': input_size
    }
    with open('config/preprocessing.pkl', 'wb') as f:
        pickle.dump(preprocessing_data, f)
    
    # Save test data for evaluation and optimization
    test_data = {'X_test': X_test, 'y_test': y_test}
    with open('results/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    # Save each individual model
    for i, model_info in enumerate(models, 1):
        filepath = f'models/model_{i}_{model_info["model_type"]}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(model_info['model'], f)
    
    # Save meta-learner for stacking prediction
    with open('models/meta_learner.pkl', 'wb') as f:
        pickle.dump(meta_learner, f)
    
    # Save prediction for analysis and optimization
    test_preds = get_predictions(models, X_test)
    stacked_probs = meta_learner.predict_proba(test_preds.T)[:, 1]
    
    predictions_data = {
        'individual_predictions': test_preds,
        'stacked_predictions': stacked_probs,
        'targets': y_test,
        'model_info': models,
        'input_size': input_size,
        'num_models': len(models)
    }
    
    with open('results/predictions.pkl', 'wb') as f:
        pickle.dump(predictions_data, f)
    
    print("Save complete")

def main():
    DATA_FILE = 'Wildfire_Dataset.csv'
    TARGET_COLUMN = 'Wildfire'
    
    print("Wildfire Ensemble Trainer")
    print("=" * 60)
    
    # Load data
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] File not found: {DATA_FILE}")
        print("Download from: [URL will be in README]")
        return
    
    print(f"Loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df):,} rows")
    
    # Engineer feature
    df = engineer_features(df)
    
    # Preprocess
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, le_target, label_encoders = preprocess_data(df, TARGET_COLUMN)
    input_size = X_train.shape[1]
    
    # Train ensemble
    total_start = time.time()
    models = train_ensemble(X_train, y_train, X_val, y_val)
    
    # Train meta-learner
    meta_learner, stacked_probs = train_meta_learner(models, X_val, y_val, X_test, y_test)
    
    # Save
    save_models(models, meta_learner, X_test, y_test, scaler, le_target, label_encoders, input_size)
    
    total_time = (time.time() - total_start) / 60
    print(f"\nTraining complete: {total_time:.1f} min")
    print("=" * 60)

if __name__ == "__main__":
    main()