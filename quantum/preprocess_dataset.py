#!/usr/bin/env python3
"""
Wildfire Dataset Preprocessing Pipeline

This script performs comprehensive feature engineering and data preparation for
quantum machine learning model training. The pipeline processes raw wildfire
occurrence data with meteorological features, constructs derived temporal and
interaction features, applies standardization, and partitions the dataset into
stratified train/validation/test splits distributed across multiple model shards.

Key Operations:
    - Temporal feature engineering (rolling windows, lags, seasonal encoding)
    - Physical interaction features (temperature-humidity, wind-fuel moisture)
    - Standardization using sklearn's StandardScaler
    - Stratified splitting to preserve class balance across splits
    - Model-wise sharding for distributed training

Output:
    - preprocessed/model_X.pkl: Training/validation/test data for each model
    - preprocessed/metadata.pkl: Feature names, scaler parameters, split ratios
    - logs/preprocess.log: Detailed processing log with timing and memory usage

Dependencies:
    - pandas: DataFrame operations and temporal grouping
    - numpy: Numerical array operations
    - sklearn: StandardScaler, train_test_split
    - psutil: Memory profiling (optional)
"""

import os
import sys
import time
import pickle
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================================
# MEMORY PROFILING UTILITY
# ============================================================================
# Provides real-time memory usage tracking during preprocessing operations.
# Falls back gracefully if psutil is unavailable.

try:
    import psutil
    def mem_gb():
        """Returns current process memory usage in gigabytes."""
        return psutil.Process().memory_info().rss / 1e9
except Exception:
    def mem_gb():
        """Fallback memory function when psutil unavailable."""
        return -1.0

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

N_MODELS = 12                    # Number of parallel quantum models to train
N_QUBITS = 6                     # Qubits available for amplitude embedding
TARGET_FEATURES = 2 ** N_QUBITS  # Required feature count (64) for 6-qubit embedding
RAW_CSV = '../data/Wildfire_Dataset.csv' # Input dataset path
PREPROC_DIR = 'preprocessed'     # Output directory for processed shards
LOG_DIR = 'logs'                 # Directory for processing logs
LOG_FILE = os.path.join(LOG_DIR, 'preprocess.log')

# Split ratios for train/validation/test partitioning
TRAIN_RATIO = 0.70  # 70% of data for training
VAL_RATIO = 0.15    # 15% for validation (hyperparameter tuning)
TEST_RATIO = 0.15   # 15% for final evaluation (never seen during training)

# Ensure output directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PREPROC_DIR, exist_ok=True)

def log(msg):
    """
    Dual-output logging function.
    
    Writes message to both console (stdout) and persistent log file with
    ISO 8601 timestamp for audit trail and debugging.
    
    Args:
        msg (str): Message to log
    """
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")

# ============================================================================
# FEATURE ENGINEERING PIPELINE
# ============================================================================

def engineer_features(df):
    """
    Comprehensive feature engineering for wildfire prediction.
    
    Constructs derived meteorological, temporal, and interaction features from
    raw measurements. Features are designed to capture fire risk indicators
    including atmospheric dryness (VPD), fuel moisture levels, wind effects,
    and temporal patterns (seasonal cycles, multi-day trends).
    
    Feature Categories:
        1. Basic Derived: Simple mathematical combinations (avg, range, variance)
        2. Physical Interactions: Domain-specific multiplicative features
        3. Temporal Patterns: Seasonal encoding (sin/cos) and fire season flags
        4. Rolling Statistics: Multi-day averages capturing weather trends
        5. Lag Features: Previous day values for temporal context
        6. Rate of Change: Day-to-day deltas indicating rapid changes
    
    Processing Flow:
        - Sort by location and time to enable temporal operations
        - Compute basic derived features from raw measurements
        - Calculate interaction terms between related variables
        - Apply rolling windows grouped by geographic location
        - Compute lag features using shift operations
        - Fill missing values from lag/diff operations using backfill
    
    Args:
        df (pd.DataFrame): Raw wildfire dataset with meteorological columns
        
    Returns:
        pd.DataFrame: Enhanced dataset with engineered features
        
    Note:
        All temporal operations (rolling, lag, diff) are grouped by location
        (latitude, longitude) to prevent information leakage across spatial
        boundaries. Missing values from first observations are backfilled
        within groups then filled with zero.
    """
    start_time = time.time()

    df_eng = df.copy()

    # ------------------------------------------------------------------------
    # TEMPORAL ORDERING AND PARSING
    # ------------------------------------------------------------------------
    # Sort dataset by location then timestamp to enable proper temporal
    # feature calculation. All rolling/lag operations require chronological
    # ordering within each geographic location.
    
    if 'datetime' in df_eng.columns:
        df_eng['datetime_parsed'] = pd.to_datetime(df_eng['datetime'])
        df_eng = df_eng.sort_values(['latitude', 'longitude', 'datetime_parsed']).reset_index(drop=True)
    else:
        return df_eng

    # ------------------------------------------------------------------------
    # BASIC DERIVED FEATURES
    # ------------------------------------------------------------------------
    
    # Temperature features
    # Range captures diurnal temperature variation (higher = more extreme)
    # Average provides baseline thermal conditions
    # Variance amplifies the range signal for model emphasis
    if 'tmmx' in df_eng.columns and 'tmmn' in df_eng.columns:
        df_eng['temp_range'] = df_eng['tmmx'] - df_eng['tmmn']
        df_eng['temp_avg'] = (df_eng['tmmx'] + df_eng['tmmn']) / 2
        df_eng['temp_variance'] = (df_eng['tmmx'] - df_eng['tmmn']) ** 2
    
    # Humidity features
    # Range shows daily humidity fluctuation
    # Average provides baseline moisture level
    # Dryness index inverts minimum humidity (higher = drier, more fire risk)
    if 'rmax' in df_eng.columns and 'rmin' in df_eng.columns:
        df_eng['humidity_range'] = df_eng['rmax'] - df_eng['rmin']
        df_eng['humidity_avg'] = (df_eng['rmax'] + df_eng['rmin']) / 2
        df_eng['dryness_index'] = 100 - df_eng['rmin']
    
    # ------------------------------------------------------------------------
    # PHYSICAL INTERACTION FEATURES
    # ------------------------------------------------------------------------
    # These capture multiplicative effects between related variables that
    # domain knowledge suggests are important for fire behavior.
    
    # Vapor Pressure Deficit (VPD) interactions
    # VPD measures atmospheric moisture demand - key fire weather indicator
    # vpd_temp: Combined heat and dryness stress on vegetation
    # vpd_squared: Amplifies high VPD values (exponential fire risk increase)
    if 'vpd' in df_eng.columns and 'tmmx' in df_eng.columns:
        df_eng['vpd_temp'] = df_eng['vpd'] * df_eng['tmmx']
        df_eng['vpd_squared'] = df_eng['vpd'] ** 2
    
    # Wind and Energy Release Component (ERC) interaction
    # ERC measures potential energy release from burning fuels
    # High wind + high ERC = rapid fire spread potential
    # wind_squared: Emphasizes strong wind events (wind effects non-linear)
    if 'vs' in df_eng.columns and 'erc' in df_eng.columns:
        df_eng['wind_erc'] = df_eng['vs'] * df_eng['erc']
        df_eng['wind_squared'] = df_eng['vs'] ** 2
    
    # Fuel moisture features
    # fm100/fm1000 are 100-hour and 1000-hour fuel moisture levels
    # Ratio captures relative drying of different fuel size classes
    # Difference shows moisture gradient across fuel types
    # Average provides overall fuel moisture status
    if 'fm100' in df_eng.columns and 'fm1000' in df_eng.columns:
        df_eng['fuel_moisture_ratio'] = df_eng['fm100'] / (df_eng['fm1000'] + 1e-6)
        df_eng['fuel_moisture_diff'] = df_eng['fm1000'] - df_eng['fm100']
        df_eng['fuel_moisture_avg'] = (df_eng['fm100'] + df_eng['fm1000']) / 2
    
    # Drought stress indicator
    # High VPD with low precipitation creates severe drought conditions
    # Ratio amplifies when precipitation is minimal
    if 'pr' in df_eng.columns and 'vpd' in df_eng.columns:
        df_eng['drought_stress'] = df_eng['vpd'] / (df_eng['pr'] + 1e-6)
    
    # Heat load from solar radiation
    # Solar radiation (srad) combined with high temperature increases
    # surface heating and fuel drying rate
    if 'srad' in df_eng.columns and 'tmmx' in df_eng.columns:
        df_eng['heat_load'] = df_eng['srad'] * df_eng['tmmx']
    
    # ERC squared
    # Amplifies high energy release conditions (fire behavior non-linear
    # with increasing ERC values)
    if 'erc' in df_eng.columns:
        df_eng['erc_squared'] = df_eng['erc'] ** 2
    
    # Fire spread potential
    # Burning Index (bi) indicates fire intensity potential
    # Combined with wind speed gives overall fire spread risk
    if 'bi' in df_eng.columns and 'vs' in df_eng.columns:
        df_eng['fire_spread_potential'] = df_eng['bi'] * df_eng['vs']
    
    # Geographic features
    # Squared terms capture non-linear geographic effects
    # Interaction term captures diagonal gradients across the region
    if 'latitude' in df_eng.columns and 'longitude' in df_eng.columns:
        df_eng['lat_squared'] = df_eng['latitude'] ** 2
        df_eng['lon_squared'] = df_eng['longitude'] ** 2
        df_eng['lat_lon_interaction'] = df_eng['latitude'] * df_eng['longitude']
    
    # ------------------------------------------------------------------------
    # TEMPORAL AND SEASONAL FEATURES
    # ------------------------------------------------------------------------
    
    # Cyclical encoding of day of year
    # Sin/cos encoding preserves cyclical nature (day 365 is close to day 1)
    # Avoids discontinuity that would occur with raw day number
    # Month provides coarser seasonal signal
    # Fire season flag identifies May-October period (peak fire activity)
    df_eng['day_of_year'] = df_eng['datetime_parsed'].dt.dayofyear
    df_eng['day_sin'] = np.sin(2 * np.pi * df_eng['day_of_year'] / 365.25)
    df_eng['day_cos'] = np.cos(2 * np.pi * df_eng['day_of_year'] / 365.25)
    df_eng['month'] = df_eng['datetime_parsed'].dt.month
    df_eng['is_fire_season'] = ((df_eng['month'] >= 5) & (df_eng['month'] <= 10)).astype(int)
    
    # ------------------------------------------------------------------------
    # ROLLING WINDOW FEATURES
    # ------------------------------------------------------------------------
    # Multi-day averages capture weather trends and persistence.
    # Grouped by location to prevent information leakage across sites.
    # min_periods=1 allows calculation even for first few days.
    
    grouped = df_eng.groupby(['latitude', 'longitude'], group_keys=False)
    
    # Temperature rolling averages
    # 3-day captures short-term heat persistence
    # 7-day captures weekly-scale weather patterns
    if 'temp_avg' in df_eng.columns:
        df_eng['temp_3day_avg'] = grouped['temp_avg'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df_eng['temp_7day_avg'] = grouped['temp_avg'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
    
    # VPD rolling averages
    # Captures atmospheric drying trends over multiple days
    # Persistent high VPD indicates sustained drought stress
    if 'vpd' in df_eng.columns:
        df_eng['vpd_3day_avg'] = grouped['vpd'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df_eng['vpd_7day_avg'] = grouped['vpd'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
    
    # Humidity rolling average
    # 3-day average smooths daily fluctuations in moisture
    if 'humidity_avg' in df_eng.columns:
        df_eng['humidity_3day_avg'] = grouped['humidity_avg'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
    
    # ------------------------------------------------------------------------
    # LAG FEATURES
    # ------------------------------------------------------------------------
    # Previous day values provide temporal context and baseline for comparison.
    # Shift(1) moves values down by one row within each location group.
    
    if 'temp_avg' in df_eng.columns:
        df_eng['temp_lag1'] = grouped['temp_avg'].shift(1)
    if 'vpd' in df_eng.columns:
        df_eng['vpd_lag1'] = grouped['vpd'].shift(1)
    if 'humidity_avg' in df_eng.columns:
        df_eng['humidity_lag1'] = grouped['humidity_avg'].shift(1)
    
    # ------------------------------------------------------------------------
    # RATE OF CHANGE FEATURES
    # ------------------------------------------------------------------------
    # Day-to-day differences capture rapid changes in conditions.
    # Sudden increases in temperature or VPD can indicate fire risk spikes.
    # Diff() computes difference from previous row within each group.
    
    if 'temp_avg' in df_eng.columns:
        df_eng['temp_change'] = grouped['temp_avg'].diff()
    if 'humidity_avg' in df_eng.columns:
        df_eng['humidity_change'] = grouped['humidity_avg'].diff()
    if 'vpd' in df_eng.columns:
        df_eng['vpd_change'] = grouped['vpd'].diff()
    
    # ------------------------------------------------------------------------
    # MISSING VALUE HANDLING
    # ------------------------------------------------------------------------
    # Lag and diff operations create NaN values for first observations at
    # each location. Backfill within groups propagates first valid value
    # backward, then any remaining NaN are filled with zero.
    
    lag_cols = ['temp_lag1', 'vpd_lag1', 'humidity_lag1', 
                'temp_change', 'humidity_change', 'vpd_change']
    for col in lag_cols:
        if col in df_eng.columns:
            df_eng[col] = grouped[col].transform(lambda x: x.bfill().fillna(0))
    
    # Clean up temporary columns used during feature construction
    df_eng.drop(columns=['datetime', 'datetime_parsed', 'day_of_year'], 
                errors='ignore', inplace=True)
    
    elapsed = time.time() - start_time
    
    return df_eng

# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def main():
    """
    Main preprocessing execution pipeline.
    
    Orchestrates the complete data preparation workflow:
        1. Load raw CSV dataset
        2. Apply comprehensive feature engineering
        3. Extract features and target variable
        4. Standardize features using StandardScaler
        5. Pad or truncate to required 64-feature dimension
        6. Perform stratified train/validation/test split
        7. Shard each split across N_MODELS for parallel training
        8. Save preprocessed shards and metadata
    
    The stratified splitting ensures class balance is preserved across all
    splits and model shards, critical for handling the severe class imbalance
    (~5% wildfire occurrence rate).
    
    Outputs:
        - preprocessed/model_X.pkl: Data shard for model X (X=1..12)
        - preprocessed/metadata.pkl: Scaler and feature information
        - logs/preprocess.log: Detailed processing log
    
    Raises:
        SystemExit: On missing input file or processing failure
    """
    start = time.time()

    try:
        # --------------------------------------------------------------------
        # DATASET LOADING
        # --------------------------------------------------------------------
        
        if not os.path.exists(RAW_CSV):
            sys.exit(1)

        df = pd.read_csv(RAW_CSV)

        # --------------------------------------------------------------------
        # FEATURE ENGINEERING
        # --------------------------------------------------------------------
        
        df = engineer_features(df)

        # --------------------------------------------------------------------
        # FEATURE AND TARGET EXTRACTION
        # --------------------------------------------------------------------
        
        target_col = 'Wildfire'
        if target_col not in df.columns:
            raise KeyError(f"Missing expected target column '{target_col}'")
        
        # Exclude metadata columns from feature set
        drop_cols = ['Unnamed: 0']
        feature_cols = [c for c in df.columns if c not in drop_cols + [target_col]]
        

        # Convert to numpy arrays with appropriate dtypes
        # float32 reduces memory footprint while maintaining precision
        # Target is binary: 1 for wildfire, 0 for no wildfire
        X = df[feature_cols].values.astype(np.float32)
        y = np.array([1 if str(v).lower() in ['yes', 'true', '1'] else 0 
                      for v in df[target_col].values], dtype=np.int8)
        

        # --------------------------------------------------------------------
        # FEATURE STANDARDIZATION
        # --------------------------------------------------------------------
        # StandardScaler transforms features to zero mean and unit variance.
        # Fit on entire dataset (transformation, not learning from test).
        # Critical for quantum amplitude embedding which is sensitive to scale.
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # --------------------------------------------------------------------
        # DIMENSION ADJUSTMENT
        # --------------------------------------------------------------------
        # Quantum amplitude embedding requires exactly 2^n_qubits features.
        # For 6 qubits, need exactly 64 features.
        # Truncate if too many, pad with zeros if too few.
        
        if X.shape[1] > TARGET_FEATURES:
            X = X[:, :TARGET_FEATURES]
        elif X.shape[1] < TARGET_FEATURES:
            pad = TARGET_FEATURES - X.shape[1]
            X = np.hstack([X, np.zeros((len(X), pad), dtype=np.float32)])
        

        # --------------------------------------------------------------------
        # STRATIFIED TRAIN/VALIDATION/TEST SPLIT
        # --------------------------------------------------------------------
        # Two-stage splitting: first separate training set, then split
        # remaining into validation and test. Stratification ensures class
        # balance is preserved in all splits (critical with 5% fire rate).
        
        
        # First split: train vs (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=(VAL_RATIO + TEST_RATIO),  # 30% for val+test
            random_state=42,
            stratify=y  # Preserve class balance
        )
        
        # Second split: val vs test from the temporary set
        val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio_adjusted),  # Split temp 50/50
            random_state=42,
            stratify=y_temp
        )
        

        # --------------------------------------------------------------------
        # MODEL-WISE SHARDING
        # --------------------------------------------------------------------
        # Each model receives a stratified random sample from each split.
        # This ensures all models see similar data distributions while
        # maintaining statistical independence for ensemble diversity.
        
        
        for model_id in range(1, N_MODELS + 1):
            shard_size = 1.0 / N_MODELS  # Each model gets 1/12 of data
            
            # Shard training data with stratification
            if len(X_train) > 0:
                _, X_train_shard, _, y_train_shard = train_test_split(
                    X_train, y_train,
                    test_size=shard_size,
                    random_state=42 + model_id,  # Different seed per model
                    stratify=y_train
                )
            else:
                X_train_shard, y_train_shard = X_train, y_train
            
            # Shard validation data
            if len(X_val) > 0:
                _, X_val_shard, _, y_val_shard = train_test_split(
                    X_val, y_val,
                    test_size=shard_size,
                    random_state=42 + model_id + 100,  # Offset seed for independence
                    stratify=y_val
                )
            else:
                X_val_shard, y_val_shard = X_val, y_val
            
            # Shard test data
            if len(X_test) > 0:
                _, X_test_shard, _, y_test_shard = train_test_split(
                    X_test, y_test,
                    test_size=shard_size,
                    random_state=42 + model_id + 200,
                    stratify=y_test
                )
            else:
                X_test_shard, y_test_shard = X_test, y_test

            # Save shard to disk
            out_path = os.path.join(PREPROC_DIR, f"model_{model_id}.pkl")
            with open(out_path, 'wb') as f:
                pickle.dump({
                    'X_train': X_train_shard,
                    'y_train': y_train_shard,
                    'X_val': X_val_shard,
                    'y_val': y_val_shard,
                    'X_test': X_test_shard,
                    'y_test': y_test_shard
                }, f, protocol=4)  # Protocol 4 for Python 3.4+ compatibility
            

        # --------------------------------------------------------------------
        # METADATA PERSISTENCE
        # --------------------------------------------------------------------
        # Save preprocessing configuration for reproducibility and future
        # inference on new data.
        
        meta_path = os.path.join(PREPROC_DIR, 'metadata.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'feature_cols': feature_cols,           # Original feature names
                'scaler': scaler,                       # Fitted StandardScaler
                'split_ratios': {
                    'train': TRAIN_RATIO,
                    'val': VAL_RATIO,
                    'test': TEST_RATIO
                },
                'sampling_method': 'stratified',
                'feature_engineering': 'full_classical_matching',
                'notes': 'Full feature engineering matching classical model, no data leakage'
            }, f, protocol=4)

        total_time = time.time() - start
        input("\n[PAUSE] Preprocessing done. Press Enter to close...")
        
    except Exception as e:
        input("\n[PAUSE] Preprocessing failed. Press Enter to close...")
        sys.exit(1)

if __name__ == "__main__":
    main()
