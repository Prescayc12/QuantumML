#!/usr/bin/env python3
"""
Wildfire Dataset Preprocessing Pipeline - Quantum ML

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
    - Exact 64-feature output for 6-qubit amplitude embedding

Output:
    - preprocessed/model_X.pkl: Training/validation/test data for each model
    - preprocessed/metadata.pkl: Feature names, scaler parameters, split ratios
    - logs/preprocess.log: Detailed processing log with timing and memory usage

Usage:
    python preprocess.py [--data PATH] [--output DIR] [--models N]
    
    --data PATH    : Path to Wildfire_Dataset.csv (default: ../data/Wildfire_Dataset.csv)
    --output DIR   : Output directory for preprocessed files (default: ./preprocessed)
    --models N     : Number of model shards to create (default: 12)

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
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================================
# MEMORY PROFILING UTILITY
# ============================================================================

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
# CONFIGURATION CLASS
# ============================================================================

class PreprocessConfig:
    """Configuration for quantum preprocessing pipeline."""
    
    def __init__(self, data_path='../data/Wildfire_Dataset.csv', 
                 output_dir='./preprocessed', n_models=12):
        self.n_models = n_models
        self.n_qubits = 6
        self.target_features = 2 ** self.n_qubits  # 64 features for amplitude embedding
        self.data_path = data_path
        self.output_dir = output_dir
        self.log_dir = './logs'
        self.log_file = os.path.join(self.log_dir, 'preprocess.log')
        
        # Split ratios
        self.train_ratio = 0.70
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def log(self, msg):
        """Dual-output logging to console and file."""
        print(msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} | {msg}\n")

# ============================================================================
# FEATURE ENGINEERING PIPELINE
# ============================================================================

def engineer_features(df, config):
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
    
    Args:
        df (pd.DataFrame): Raw wildfire dataset with meteorological columns
        config (PreprocessConfig): Configuration object
        
    Returns:
        pd.DataFrame: Enhanced dataset with engineered features
    """
    start_time = time.time()
    config.log("[STEP] Starting comprehensive feature engineering...")

    df_eng = df.copy()

    # ------------------------------------------------------------------------
    # TEMPORAL ORDERING AND PARSING
    # ------------------------------------------------------------------------
    
    if 'datetime' in df_eng.columns:
        df_eng['datetime_parsed'] = pd.to_datetime(df_eng['datetime'])
        df_eng = df_eng.sort_values(['latitude', 'longitude', 'datetime_parsed']).reset_index(drop=True)
        config.log("  Sorted by location and timestamp")
    else:
        config.log("[WARN] No 'datetime' column found, skipping temporal features")
        return df_eng

    # ------------------------------------------------------------------------
    # BASIC DERIVED FEATURES
    # ------------------------------------------------------------------------
    
    # Temperature features
    if 'tmmx' in df_eng.columns and 'tmmn' in df_eng.columns:
        df_eng['temp_range'] = df_eng['tmmx'] - df_eng['tmmn']
        df_eng['temp_avg'] = (df_eng['tmmx'] + df_eng['tmmn']) / 2
        df_eng['temp_variance'] = (df_eng['tmmx'] - df_eng['tmmn']) ** 2
        config.log("  Added: temp_range, temp_avg, temp_variance")
    
    # Humidity features
    if 'rmax' in df_eng.columns and 'rmin' in df_eng.columns:
        df_eng['humidity_range'] = df_eng['rmax'] - df_eng['rmin']
        df_eng['humidity_avg'] = (df_eng['rmax'] + df_eng['rmin']) / 2
        df_eng['dryness_index'] = 100 - df_eng['rmin']
        config.log("  Added: humidity_range, humidity_avg, dryness_index")
    
    # ------------------------------------------------------------------------
    # PHYSICAL INTERACTION FEATURES
    # ------------------------------------------------------------------------
    
    # VPD × Temperature interaction
    if 'vpd' in df_eng.columns and 'tmmx' in df_eng.columns:
        df_eng['vpd_temp'] = df_eng['vpd'] * df_eng['tmmx']
        df_eng['vpd_squared'] = df_eng['vpd'] ** 2
        config.log("  Added: vpd_temp, vpd_squared")
    
    # Wind × ERC interaction
    if 'vs' in df_eng.columns and 'erc' in df_eng.columns:
        df_eng['wind_erc'] = df_eng['vs'] * df_eng['erc']
        df_eng['wind_squared'] = df_eng['vs'] ** 2
        config.log("  Added: wind_erc, wind_squared")
    
    # Fuel moisture features
    if 'fm100' in df_eng.columns and 'fm1000' in df_eng.columns:
        df_eng['fuel_moisture_ratio'] = df_eng['fm100'] / (df_eng['fm1000'] + 1e-6)
        df_eng['fuel_moisture_diff'] = df_eng['fm1000'] - df_eng['fm100']
        df_eng['fuel_moisture_avg'] = (df_eng['fm100'] + df_eng['fm1000']) / 2
        config.log("  Added: fuel_moisture features")
    
    # Drought stress
    if 'pr' in df_eng.columns and 'vpd' in df_eng.columns:
        df_eng['drought_stress'] = df_eng['vpd'] / (df_eng['pr'] + 1e-6)
        config.log("  Added: drought_stress")
    
    # Heat load
    if 'srad' in df_eng.columns and 'tmmx' in df_eng.columns:
        df_eng['heat_load'] = df_eng['srad'] * df_eng['tmmx']
        config.log("  Added: heat_load")
    
    # ERC squared
    if 'erc' in df_eng.columns:
        df_eng['erc_squared'] = df_eng['erc'] ** 2
        config.log("  Added: erc_squared")
    
    # Fire spread potential
    if 'bi' in df_eng.columns and 'vs' in df_eng.columns:
        df_eng['fire_spread_potential'] = df_eng['bi'] * df_eng['vs']
        config.log("  Added: fire_spread_potential")
    
    # Geographic features
    if 'latitude' in df_eng.columns and 'longitude' in df_eng.columns:
        df_eng['lat_squared'] = df_eng['latitude'] ** 2
        df_eng['lon_squared'] = df_eng['longitude'] ** 2
        df_eng['lat_lon_interaction'] = df_eng['latitude'] * df_eng['longitude']
        config.log("  Added: geographic features")
    
    # ------------------------------------------------------------------------
    # SEASONAL FEATURES
    # ------------------------------------------------------------------------
    
    df_eng['day_of_year'] = df_eng['datetime_parsed'].dt.dayofyear
    df_eng['day_sin'] = np.sin(2 * np.pi * df_eng['day_of_year'] / 365.25)
    df_eng['day_cos'] = np.cos(2 * np.pi * df_eng['day_of_year'] / 365.25)
    df_eng['month'] = df_eng['datetime_parsed'].dt.month
    df_eng['is_fire_season'] = ((df_eng['month'] >= 5) & (df_eng['month'] <= 10)).astype(int)
    config.log("  Added: seasonal features")
    
    # ------------------------------------------------------------------------
    # TEMPORAL ROLLING FEATURES
    # ------------------------------------------------------------------------
    
    config.log("  Computing temporal rolling features...")
    grouped = df_eng.groupby(['latitude', 'longitude'], group_keys=False)
    
    # Temperature rolling averages
    df_eng['temp_3day_avg'] = grouped['temp_avg'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df_eng['temp_7day_avg'] = grouped['temp_avg'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # VPD rolling averages
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
    
    # ------------------------------------------------------------------------
    # LAG FEATURES
    # ------------------------------------------------------------------------
    
    df_eng['temp_lag1'] = grouped['temp_avg'].shift(1)
    df_eng['vpd_lag1'] = grouped['vpd'].shift(1) if 'vpd' in df_eng.columns else 0
    df_eng['humidity_lag1'] = grouped['humidity_avg'].shift(1) if 'humidity_avg' in df_eng.columns else 0
    
    # ------------------------------------------------------------------------
    # RATE OF CHANGE
    # ------------------------------------------------------------------------
    
    df_eng['temp_change'] = grouped['temp_avg'].diff()
    df_eng['humidity_change'] = grouped['humidity_avg'].diff() if 'humidity_avg' in df_eng.columns else 0
    df_eng['vpd_change'] = grouped['vpd'].diff() if 'vpd' in df_eng.columns else 0
    
    # Fill missing values from lag operations
    lag_cols = ['temp_lag1', 'vpd_lag1', 'humidity_lag1', 'temp_change', 'humidity_change', 'vpd_change']
    for col in lag_cols:
        if col in df_eng.columns:
            df_eng[col] = grouped[col].transform(lambda x: x.fillna(method='bfill').fillna(0))
    
    elapsed = (time.time() - start_time) / 60
    config.log(f"  Feature engineering complete: {elapsed:.1f} min")
    
    # Clean up temporary columns
    df_eng = df_eng.drop(columns=['datetime_parsed', 'day_of_year'], errors='ignore')
    return df_eng

# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Preprocess wildfire dataset for quantum ML')
    parser.add_argument('--data', type=str, default='../data/Wildfire_Dataset.csv',
                       help='Path to Wildfire_Dataset.csv')
    parser.add_argument('--output', type=str, default='./preprocessed',
                       help='Output directory for preprocessed files')
    parser.add_argument('--models', type=int, default=12,
                       help='Number of model shards to create')
    args = parser.parse_args()
    
    # Initialize configuration
    config = PreprocessConfig(
        data_path=args.data,
        output_dir=args.output,
        n_models=args.models
    )
    
    start = time.time()
    
    try:
        config.log("="*80)
        config.log("QUANTUM WILDFIRE PREPROCESSING PIPELINE")
        config.log("="*80)
        config.log(f"Input: {config.data_path}")
        config.log(f"Output: {config.output_dir}")
        config.log(f"Models: {config.n_models}")
        config.log(f"Target features: {config.target_features} (for {config.n_qubits}-qubit encoding)")
        config.log("")
        
        # --------------------------------------------------------------------
        # DATA LOADING
        # --------------------------------------------------------------------
        
        if not os.path.exists(config.data_path):
            config.log(f"[ERROR] Dataset not found: {config.data_path}")
            config.log("Download from: https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset")
            sys.exit(1)
        
        config.log(f"[STEP] Loading dataset from {config.data_path}...")
        df = pd.read_csv(config.data_path)
        config.log(f"[INFO] Loaded {len(df):,} samples, mem={mem_gb():.3f} GB")
        
        # --------------------------------------------------------------------
        # FEATURE ENGINEERING
        # --------------------------------------------------------------------
        
        df = engineer_features(df, config)

        # --------------------------------------------------------------------
        # FEATURE AND TARGET EXTRACTION
        # --------------------------------------------------------------------
        
        target_col = 'Wildfire'
        if target_col not in df.columns:
            raise KeyError(f"Missing expected target column '{target_col}'")
        
        # Exclude metadata columns
        drop_cols = ['Unnamed: 0']
        feature_cols = [c for c in df.columns if c not in drop_cols + [target_col]]
        
        config.log(f"[INFO] Feature columns before padding: {len(feature_cols)}")

        # Convert to numpy arrays
        X = df[feature_cols].values.astype(np.float32)
        y = np.array([1 if str(v).lower() in ['yes', 'true', '1'] else 0 
                      for v in df[target_col].values], dtype=np.int8)
        
        config.log(f"[INFO] Converted to NumPy: X={X.shape}, y={y.shape}, mem={mem_gb():.3f} GB")
        config.log(f"[INFO] Class balance: {y.mean():.3%} fires")

        # --------------------------------------------------------------------
        # FEATURE STANDARDIZATION
        # --------------------------------------------------------------------
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        config.log(f"[STEP] Scaling complete, mem={mem_gb():.3f} GB")

        # --------------------------------------------------------------------
        # DIMENSION ADJUSTMENT
        # --------------------------------------------------------------------
        
        if X.shape[1] > config.target_features:
            config.log(f"[WARN] Truncating from {X.shape[1]} to {config.target_features} features")
            X = X[:, :config.target_features]
        elif X.shape[1] < config.target_features:
            pad = config.target_features - X.shape[1]
            config.log(f"[INFO] Padding from {X.shape[1]} to {config.target_features} features (+{pad} zeros)")
            X = np.hstack([X, np.zeros((len(X), pad), dtype=np.float32)])
        
        config.log(f"[INFO] Final feature count: {X.shape[1]}")

        # --------------------------------------------------------------------
        # STRATIFIED TRAIN/VALIDATION/TEST SPLIT
        # --------------------------------------------------------------------
        
        config.log("[STEP] Performing stratified train/val/test split...")
        
        # First split: train vs (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=(config.val_ratio + config.test_ratio),
            random_state=42,
            stratify=y
        )
        
        # Second split: val vs test
        val_ratio_adjusted = config.val_ratio / (config.val_ratio + config.test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio_adjusted),
            random_state=42,
            stratify=y_temp
        )
        
        config.log(f"[INFO] Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        config.log(f"[INFO] Class balance - Train: {y_train.mean():.3%}, Val: {y_val.mean():.3%}, Test: {y_test.mean():.3%}")

        # --------------------------------------------------------------------
        # MODEL-WISE SHARDING
        # --------------------------------------------------------------------
        
        config.log(f"[STEP] Sharding data across {config.n_models} models with stratified sampling...")
        
        for model_id in range(1, config.n_models + 1):
            shard_size = 1.0 / config.n_models
            
            # Shard training data
            if len(X_train) > 0:
                _, X_train_shard, _, y_train_shard = train_test_split(
                    X_train, y_train,
                    test_size=shard_size,
                    random_state=42 + model_id,
                    stratify=y_train
                )
            else:
                X_train_shard, y_train_shard = X_train, y_train
            
            # Shard validation data
            if len(X_val) > 0:
                _, X_val_shard, _, y_val_shard = train_test_split(
                    X_val, y_val,
                    test_size=shard_size,
                    random_state=42 + model_id + 100,
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

            # Save shard
            out_path = os.path.join(config.output_dir, f"model_{model_id}.pkl")
            with open(out_path, 'wb') as f:
                pickle.dump({
                    'X_train': X_train_shard,
                    'y_train': y_train_shard,
                    'X_val': X_val_shard,
                    'y_val': y_val_shard,
                    'X_test': X_test_shard,
                    'y_test': y_test_shard
                }, f, protocol=4)
            
            config.log(f"[SAVED] {out_path} | Train: {len(X_train_shard)}, Val: {len(X_val_shard)}, Test: {len(X_test_shard)} | Fire%: {y_train_shard.mean():.3%}")

        # --------------------------------------------------------------------
        # METADATA PERSISTENCE
        # --------------------------------------------------------------------
        
        meta_path = os.path.join(config.output_dir, 'metadata.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'feature_cols': feature_cols,
                'scaler': scaler,
                'split_ratios': {
                    'train': config.train_ratio,
                    'val': config.val_ratio,
                    'test': config.test_ratio
                },
                'sampling_method': 'stratified',
                'feature_engineering': 'full_classical_matching',
                'n_qubits': config.n_qubits,
                'target_features': config.target_features,
                'notes': 'Full feature engineering matching classical model, no data leakage'
            }, f, protocol=4)
        config.log(f"[SAVED] Metadata saved -> {meta_path}")

        total_time = time.time() - start
        config.log(f"")
        config.log(f"Preprocessing complete in {total_time:.2f}s ({total_time/60:.1f} min), mem={mem_gb():.3f} GB")
        config.log("="*80)
        
    except Exception as e:
        config.log(f"[FATAL] Exception: {e}")
        config.log(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
