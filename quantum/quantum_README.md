# Quantum Machine Learning Implementation

Variational quantum circuit (VQC) ensemble for wildfire prediction using PennyLane.

## Overview

This implementation uses quantum computing to classify wildfire risk, achieving **52.88% balanced accuracy** on test data. The quantum approach underperformed compared to classical methods (81.89% BA) due to computational constraints and the classical nature of the dataset.

**Key Results:**
- Balanced Accuracy: **52.88%** (~random performance)
- Overall Accuracy: 64.48%
- Recall (Fire Detection): 38.02%
- Specificity: 65.95%
- Training Time: ~48+ hours for full 12-model ensemble

## Architecture

**Quantum Circuit Design:**
- **Input Encoding**: 64 classical features → 6 data qubits via amplitude embedding
- **Processing**: 6 additional qubits for expressivity (12 total)
- **Circuit Depth**: 2-3 layers of parameterized gates (RY/RZ rotations + CNOT entanglement)
- **Measurement**: Single expectation value on qubit 0 for binary classification
- **Parameters**: 48-72 trainable parameters per model

**Ensemble:**
- 12 diverse quantum models with varied hyperparameters
- Each model trains on stratified 1/12th data shard
- Session-based training (4 sessions × 5000 samples = 20k total per model)

## Quick Start

### Simple: Train Single Model

For demonstration or testing:

```bash
# 1. Preprocess data (one-time, ~90 seconds)
python preprocess_dataset.py

# 2. Train a single quantum model (~2.5 hours)
python train_model.py

# Model trains through 4 sessions automatically
# Results saved to ./checkpoints/model_1_final.pkl
```

### Advanced: Train Full Ensemble

For reproducing paper results (requires 48+ hours):

```bash
# Launch parallel training of all 12 models
python orchestrator.py

# Models train in batches of 4
# Progress logged to ./logs/model_X/session_Y.log
# Checkpoints saved to ./checkpoints/
```

## File Descriptions

### Core Training Files

**`preprocess_dataset.py`**
- Loads raw wildfire dataset from `../data/Wildfire_Dataset.csv`
- Performs comprehensive feature engineering (60+ derived features)
- Standardizes features using StandardScaler
- Pads/truncates to exactly 64 features for 6-qubit amplitude embedding
- Creates 12 stratified model shards with 70/15/15 train/val/test split
- Output: `preprocessed/model_1.pkl` through `model_12.pkl` + `metadata.pkl`
- Runtime: ~90 seconds

**`train_model.py`**
- **Simple single-model training script**
- Trains one quantum model through all 4 sessions
- Uses default configuration (model 1, 5000 samples/session, 30 iterations/session)
- Generates test predictions and calculates final metrics
- Output: Session checkpoints + `model_1_final.pkl` with results
- Runtime: ~2.5 hours
- **Use this for quick tests and demonstrations**

**`train_worker.py`**
- **Advanced worker for parallel ensemble training**
- Trains individual model for one session with configurable parameters
- Called by orchestrator for distributed training
- Supports loading weights from previous sessions
- Caches test predictions in final session for fast evaluation
- Output: `checkpoints/model_X_session_Y.pkl`
- Runtime: ~20 minutes per session
- **Use this with orchestrator for full ensemble**

**`orchestrator.py`**
- **Parallel training manager for full 12-model ensemble**
- Launches workers in batches (default: 4 models at a time)
- Monitors completion via checkpoint polling
- Automatically runs preprocessing if needed
- Output: All checkpoints across all models/sessions
- Runtime: ~48+ hours for full ensemble (12 models × 4 sessions)
- **Use this to reproduce paper results**

**`optimize_ensemble.py`**
- Post-training threshold and ensemble strategy optimization
- Re-runs quantum circuits to generate test predictions
- Tests 41 thresholds (0.20 to 0.60) to find optimal decision boundary
- Evaluates 4 ensemble strategies (simple average, F1-weighted, recall-weighted, meta-learner)
- Output: `config/optimal_config.pkl` + `config/threshold_optimization.png`
- Runtime: ~2 hours (re-runs circuits on test set)
- **Run after full ensemble training completes**

### Configuration Files

All scripts accept command-line arguments for customization:

```bash
# Preprocessing
python preprocess_dataset.py --output ./preprocessed --models 12

# Single model training
python train_model.py --model-id 3 --samples 2000 --iterations 20 --sessions 4

# Parallel ensemble training
python orchestrator.py --models 12 --batch-size 4 --checkpoint-dir ./checkpoints

# Optimization
python optimize_ensemble.py --checkpoint-dir ./checkpoints --test-samples 1000
```

See `--help` on each script for full options.

## Workflow

**Standard workflow for full ensemble:**

1. **Preprocess** (one-time): `python preprocess_dataset.py` → Creates 12 model shards
2. **Train ensemble**: `python orchestrator.py` → Trains all 12 models across 4 sessions
3. **Optimize**: `python optimize_ensemble.py` → Finds best threshold and ensemble strategy

**Quick test workflow:**

1. **Preprocess** (one-time): `python preprocess_dataset.py`
2. **Train one model**: `python train_model.py` → Fast single-model test

## Model Configurations

Each model has unique hyperparameters for ensemble diversity:

| Model | Learning Rate | Circuit Layers | Initialization Scale |
|-------|---------------|----------------|---------------------|
| 1     | 0.01          | 2              | 2π                  |
| 2     | 0.02          | 2              | π                   |
| 3     | 0.015         | 3              | 1.5π                |
| 4     | 0.005         | 2              | 0.5π                |
| 5     | 0.01          | 3              | 2.5π                |
| 6     | 0.025         | 2              | 1.2π                |
| 7     | 0.008         | 3              | 1.8π                |
| 8     | 0.012         | 2              | 0.8π                |
| 9     | 0.018         | 3              | 2.2π                |
| 10    | 0.015         | 2              | π                   |
| 11    | 0.01          | 3              | 1.5π                |
| 12    | 0.02          | 2              | 0.7π                |

**Common parameters:**
- Optimizer: Gradient Descent
- Early stopping: Patience = 5 iterations
- Loss: Class-weighted MSE (handles 5% fire imbalance)
- Training: Session-based (4 sessions × 5000 samples)

## Performance Analysis

### Why Quantum Underperformed

**Computational Limitations:**
- Limited to 20k training samples per model (vs 550k available in shard)
- Session-based training fragmented learning across 4 separate runs
- No weight persistence between sessions initially
- Total training time: 48+ hours on CPU simulator

**Algorithmic Challenges:**
- **Barren plateau problem**: Gradients vanish with random initialization
- **Classical data**: Tabular features lack quantum structure for advantage
- **Limited expressivity**: 2-3 layer circuits may be insufficient
- **No global view**: Session-based approach prevents seeing full data distribution

**Dataset Characteristics:**
- Severe class imbalance (5% fires) despite class weighting
- High-dimensional classical features (64) not naturally quantum
- Temporal/spatial correlations lost in quantum encoding

### Comparison to Classical Approach

| Aspect | Quantum | Classical |
|--------|---------|-----------|
| Training Time | 48+ hours | 30-60 min |
| Balanced Accuracy | 52.88% | 81.89% |
| Samples Used | 20k/model | 550k/model |
| Scalability | Limited (simulation) | Excellent |
| Interpretability | Low | High |

**Conclusion:** For this tabular wildfire dataset, classical gradient boosting (XGBoost/LightGBM) vastly outperforms quantum approaches. Quantum advantage may emerge with:
- Access to quantum hardware (not simulators)
- Datasets with inherent quantum structure
- Improved initialization strategies (avoiding barren plateaus)
- Hybrid quantum-classical architectures

## System Requirements

### Minimum
- Python 3.8+
- 16GB RAM (for preprocessing)
- 4 CPU cores
- 10GB disk space

### Recommended
- Python 3.10+
- 32GB RAM
- 8+ CPU cores
- Multi-threading enabled (`OMP_NUM_THREADS`)

### For Full Ensemble Training
- 48+ hours of uninterrupted compute time
- 64GB RAM recommended
- 12-16 CPU cores for efficient parallel training

## Dependencies

See `../requirements.txt` for full list. Key packages:

- **pennylane**: Quantum circuit framework
- **pennylane-lightning**: Fast CPU-based quantum simulator
- **numpy**: Numerical operations (PennyLane-compatible version)
- **scikit-learn**: Preprocessing, metrics
- **pandas**: Data manipulation

Install:
```bash
cd ..
pip install -r requirements.txt
```

## Troubleshooting

### "Out of memory" during preprocessing
- Preprocessing uses ~10GB RAM for full 9.5M dataset
- Increase WSL2 memory limit in `.wslconfig`:
  ```
  [wsl2]
  memory=16GB
  ```
- Restart WSL: `wsl --shutdown` in Windows CMD

### Training is very slow
- Expected: ~40 seconds per iteration on modern CPU
- Each session takes ~20 minutes (30 iterations)
- Full model (4 sessions): ~2.5 hours
- Full ensemble (12 models): ~48 hours
- Use `train_model.py` with `--samples 2000 --iterations 10` for faster testing

### "Checkpoint not found" in optimize_ensemble.py
- Ensure all 12 models completed session 4
- Check `./checkpoints/` for `model_X_session_4.pkl` files
- Use `--checkpoint-dir` to specify correct location

### Models not training in parallel
- Verify orchestrator shows different PIDs for each model
- Check `./logs/model_X/session_Y.log` for individual progress
- Reduce `--batch-size` if hitting memory limits

### Poor accuracy (stuck at ~50%)
- This is expected behavior for this dataset
- Quantum approach struggles with classical tabular data
- Try increasing `--samples` or `--iterations` for marginal improvements
- For better results, use classical implementation (`../classical/`)

## Advanced Usage

### Custom Model Training

Train specific model with custom parameters:

```bash
python train_model.py \
  --model-id 5 \
  --samples 10000 \
  --iterations 50 \
  --sessions 4 \
  --output ./custom_checkpoints
```

### Parallel Training Configuration

Adjust batch size and resource allocation:

```bash
python orchestrator.py \
  --models 12 \
  --sessions 4 \
  --batch-size 2 \
  --samples 3000 \
  --iterations 20
```

### Quick Test Run

Fast training for testing:

```bash
python train_model.py --samples 1000 --iterations 10 --sessions 2
```

Completes in ~30 minutes vs 2.5 hours.

## Output Files

**After preprocessing:**
```
preprocessed/
  ├── model_1.pkl through model_12.pkl  # Model shards
  └── metadata.pkl                       # Scaler and feature info
```

**After training single model:**
```
checkpoints/
  ├── model_1_session_1.pkl  # Session checkpoints
  ├── model_1_session_2.pkl
  ├── model_1_session_3.pkl
  ├── model_1_session_4.pkl
  └── model_1_final.pkl      # Final results with metrics
```

**After ensemble training:**
```
checkpoints/
  ├── model_1_session_1.pkl through model_12_session_4.pkl
logs/
  └── model_X/session_Y.log  # Training logs
```

**After optimization:**
```
config/
  ├── optimal_config.pkl              # Best threshold and strategy
  └── threshold_optimization.png      # Visualization
```

## Citation

If you use this quantum implementation, please cite:

```
Trepanier, J., Aycock, P., & Ayala-Lagunas, A. (2025). 
Wildfire Prediction Using Quantum Machine Learning. 
Quantum Variational Circuit Implementation.
NC State University.
```

## License

MIT License - See repository root LICENSE file

## Further Reading

- **Paper**: See `../paper/Quantum_Final_Project.pdf` for full methodology and analysis
- **Classical baseline**: See `../classical/README.md` for high-performing classical approach
- **PennyLane docs**: https://pennylane.ai/
