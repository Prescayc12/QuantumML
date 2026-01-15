# Classical Ensemble Implementation

High-performance wildfire prediction using gradient boosting ensemble methods.

## Overview

This implementation achieves **81.89% balanced accuracy** using an ensemble of 12 gradient boosting models (6 XGBoost + 6 LightGBM) with optimized decision thresholds and F1-weighted ensemble averaging.

**Key Results:**
- Balanced Accuracy: **81.89%**
- Overall Accuracy: 76.47%
- Recall (Fire Detection): 86.86%
- Specificity: 76.92%
- F1 Score: 28.91%

## Quick Start

```bash
# 1. Preprocess the dataset
python preprocess.py

# 2. Train the 12-model ensemble (~30-60 minutes)
python train_ensemble.py

# 3. Optimize thresholds and ensemble strategy
python optimize_ensemble.py
```

## File Descriptions

### `preprocess.py`
Comprehensive feature engineering and data preparation.

**What it does:**
- Loads raw wildfire dataset from `../data/Wildfire_Dataset.csv`
- Engineers 60+ derived features (temporal, interaction, seasonal)
- Splits into train (70%), validation (15%), test (15%) with stratification
- Scales features using StandardScaler
- Saves preprocessed data to `preprocessed/` directory

**Output:**
- `preprocessed/train_data.pkl` - Training features and labels
- `preprocessed/val_data.pkl` - Validation features and labels  
- `preprocessed/test_data.pkl` - Test features and labels
- `preprocessed/scaler.pkl` - Fitted StandardScaler for inference

**Run time:** ~10-20 minutes

### `train_ensemble.py`
Trains 12 diverse gradient boosting models with early stopping.

**Architecture:**
- **6 XGBoost models** with varied configurations:
  - Max depths: 6, 8, 10, 12
  - Learning rates: 0.05 - 0.15
  - Min child weights: 1 - 7
  
- **6 LightGBM models** with varied configurations:
  - Num leaves: 31, 63, 127, 255
  - Learning rates: 0.05 - 0.12
  - Min child samples: 20 - 50

**What it does:**
- Loads preprocessed data
- Trains each model with validation-based early stopping
- Saves individual models to `models/` directory
- Saves predictions and metadata to `results/` directory
- Tracks validation metrics (F1, recall, precision) for each model

**Output:**
- `models/model_1_xgboost.pkl` through `models/model_12_lightgbm.pkl`
- `config/preprocessing.pkl` - Scaler and encoder configuration
- `results/test_data.pkl` - Test set for evaluation
- `results/predictions.pkl` - Model predictions and metadata

**Run time:** ~30-60 minutes (dataset dependent)

**Memory:** ~8-16GB RAM recommended

### `optimize_ensemble.py`
Post-training threshold and ensemble strategy optimization.

**What it does:**
- Loads trained models and test predictions
- Tests thresholds from 0.20 to 0.60 (41 values)
- Identifies optimal thresholds for:
  - Highest overall accuracy
  - Highest balanced accuracy
- Evaluates ensemble strategies:
  - Simple average (equal weighting)
  - F1-weighted (weight by validation F1 score)
  - Recall-weighted (weight by validation recall)
  - Meta-learner (Random Forest stacking)
- Selects best strategy based on balanced accuracy
- Generates visualization of threshold optimization

**Output:**
- `config/optimal_config.pkl` - Best threshold and strategy
- `config/threshold_optimization.png` - Visualization plot

**Run time:** <1 minute

**Key finding:** F1-weighted strategy at 0.48 threshold achieves highest balanced accuracy (81.89%)

## Feature Engineering Details

The preprocessing pipeline creates the following feature categories:

### Basic Derived Features
- **Temperature**: range, average, variance
- **Humidity**: range, average, dryness index
- **Geographic**: latitude/longitude squared, interactions

### Physical Interaction Features
- **VPD × Temperature**: Atmospheric drying × heat load
- **Wind × ERC**: Wind speed × energy release (fire spread potential)
- **Fuel Moisture**: 100hr/1000hr ratios and differences
- **Drought Stress**: VPD / precipitation ratio
- **Heat Load**: Solar radiation × temperature

### Temporal Features
- **Seasonal Encoding**: Sin/cos of day-of-year (preserves cyclical nature)
- **Fire Season**: Binary indicator (May-October)
- **Rolling Averages**: 3-day and 7-day windows for temperature, VPD, humidity
- **Lag Features**: Previous day values for key variables
- **Rate of Change**: Day-to-day deltas (temperature, humidity, VPD changes)

All temporal operations are grouped by (latitude, longitude) to prevent spatial information leakage.

## Model Configuration Details

### XGBoost Models (1-6)

| Model | Max Depth | Learning Rate | Min Child Weight |
|-------|-----------|---------------|------------------|
| 1     | 6         | 0.10          | 1                |
| 2     | 8         | 0.05          | 3                |
| 3     | 10        | 0.10          | 5                |
| 4     | 12        | 0.05          | 7                |
| 5     | 8         | 0.15          | 2                |
| 6     | 10        | 0.08          | 4                |

**Common parameters:**
- Trees: 1000 (with early stopping)
- Subsample: 0.8
- Col sample by tree: 0.8
- Objective: Binary logistic

### LightGBM Models (7-12)

| Model | Num Leaves | Learning Rate | Min Child Samples |
|-------|------------|---------------|-------------------|
| 7     | 31         | 0.10          | 20                |
| 8     | 63         | 0.05          | 30                |
| 9     | 127        | 0.10          | 40                |
| 10    | 255        | 0.05          | 50                |
| 11    | 95         | 0.08          | 25                |
| 12    | 191        | 0.12          | 35                |

**Common parameters:**
- Trees: 1000 (with early stopping)
- Subsample: 0.8
- Feature fraction: 0.8
- Objective: Binary

### Ensemble Strategy

**F1-Weighted Averaging** (Best Performance):
- Each model's predictions weighted by its validation F1 score
- Higher-performing models have more influence
- Final prediction: Weighted average ≥ 0.48 threshold

**Rationale:** Models with better F1 scores demonstrate superior balance between precision and recall, making them more reliable for final predictions.

## Performance Breakdown

### Threshold Analysis

| Threshold | Metric Focus | Overall Acc | Balanced Acc | Recall | Specificity | False Alarms |
|-----------|-------------|-------------|--------------|--------|-------------|--------------|
| **0.60**  | Max Accuracy | **86.52%** | 77.38% | 67.16% | 87.60% | 223,398 |
| **0.48**  | Max Balanced | 76.47% | **81.89%** | **86.86%** | 76.92% | 433,788 |

**Trade-off:** 
- Higher threshold (0.60) → Fewer false alarms but misses more fires
- Lower threshold (0.48) → Catches more fires but more false alarms

**Recommendation:** Use 0.48 threshold (balanced accuracy optimized) for operational deployment to maximize fire detection while maintaining acceptable false alarm rate.

### Ensemble Strategy Comparison

| Strategy | Balanced Accuracy | F1 Score | Recall | Specificity |
|----------|-------------------|----------|--------|-------------|
| Simple Average | 81.08% | 28.81% | 86.24% | 75.92% |
| **F1-Weighted** | **81.89%** | **28.91%** | **86.86%** | **76.92%** |
| Recall-Weighted | 81.26% | 28.85% | 86.38% | 76.14% |
| Meta-Learner | 63.05% | 15.23% | 26.50% | 99.59% |

**Winner:** F1-weighted strategy provides best overall performance across metrics.

## System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 2 CPU cores
- 5GB disk space (with dataset)

### Recommended
- Python 3.10+
- 16GB RAM
- 4+ CPU cores
- Multi-threading enabled
- 10GB disk space

## Dependencies

Core libraries (see `../requirements.txt` for versions):
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Preprocessing, metrics, meta-learning
- `xgboost` - XGBoost models
- `lightgbm` - LightGBM models
- `matplotlib` - Visualization

## Troubleshooting

### "File not found: Wildfire_Dataset.csv"
- Download dataset from Kaggle (see `../data/README.md`)
- Place in `../data/` directory
- Run `preprocess.py` before training

### "Out of memory" during training
- Close other applications
- Reduce dataset size in `preprocess.py` (sample fewer rows)
- Train models sequentially instead of in parallel (modify `train_ensemble.py`)

### "Module not found" errors
```bash
pip install -r ../requirements.txt
```

### Training is very slow
- Expected: 30-60 minutes for full ensemble
- Check CPU usage (should be high during training)
- Early stopping will terminate models that aren't improving
- Consider reducing `n_estimators` in model configs for faster testing

## Advanced Usage

### Custom Model Configuration

Edit `train_ensemble.py` to modify model hyperparameters:

```python
# Example: Add a new XGBoost configuration
xgb_configs = [
    {'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 1},
    # Add your custom config here
    {'max_depth': 14, 'learning_rate': 0.03, 'min_child_weight': 10},
]
```

### Custom Threshold

To use a different threshold without re-optimizing:

```python
import pickle
with open('config/optimal_config.pkl', 'rb') as f:
    config = pickle.load(f)

config['threshold'] = 0.55  # Your custom threshold

with open('config/optimal_config.pkl', 'wb') as f:
    pickle.dump(config, f)
```

### Inference on New Data

```python
import pickle
import pandas as pd
from preprocess import engineer_features

# Load trained models and preprocessing config
models = []
for i in range(1, 13):
    with open(f'models/model_{i}_*.pkl', 'rb') as f:
        models.append(pickle.load(f))

with open('config/preprocessing.pkl', 'rb') as f:
    prep_config = pickle.load(f)

# Prepare new data
new_data = pd.read_csv('new_wildfire_data.csv')
new_data = engineer_features(new_data)
X_new = prep_config['scaler'].transform(new_data)

# Get predictions
predictions = []
for model in models:
    pred = model.predict_proba(X_new)[:, 1]
    predictions.append(pred)

# Apply F1-weighted ensemble
weights = config['weights']  # From optimal_config.pkl
ensemble_probs = sum(p * w for p, w in zip(predictions, weights))
final_predictions = (ensemble_probs >= config['threshold']).astype(int)
```

## Performance Analysis

### Why This Approach Works

1. **Ensemble Diversity**: Different algorithms (XGBoost vs LightGBM) and hyperparameters capture different patterns
2. **Class Weighting**: Built-in handling of severe class imbalance (~5% fires)
3. **Feature Engineering**: Domain-specific features (VPD interactions, fuel moisture ratios) encode fire risk
4. **Temporal Features**: Rolling averages and lag features capture weather trends leading to fires
5. **Threshold Optimization**: Separate optimization step allows tuning for specific operational goals

### Comparison to Quantum Approach

| Aspect | Classical | Quantum |
|--------|-----------|---------|
| Training Time | 30-60 min | 48+ hours |
| Balanced Accuracy | 81.89% | ~50% |
| Scalability | Excellent | Limited (simulation) |
| Interpretability | High (feature importance) | Low (quantum states) |

Classical gradient boosting is highly effective for tabular data with clear feature engineering opportunities, as demonstrated by these results.

## Citation

If you use this implementation, please cite:

```
Trepanier, J., Aycock, P., & Ayala-Lagunas, A. (2025). 
Wildfire Prediction Using Quantum Machine Learning. 
Classical Ensemble Implementation.
NC State University.
```

## License

MIT License - See repository root LICENSE file

## Questions?

See main repository README or open an issue on GitHub.
