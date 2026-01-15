# Wildfire Prediction Using Quantum and Classical Machine Learning

Comparative study of classical ensemble methods and variational quantum circuits for wildfire ignition prediction using meteorological data.

## Overview

This repository contains the complete implementation of a wildfire prediction system comparing two approaches:

- **Classical Ensemble**: XGBoost/LightGBM models achieving **81.89% balanced accuracy**
- **Quantum ML**: Variational quantum circuits achieving **~50% balanced accuracy** (limited by computational constraints)

This work was conducted as part of a graduate quantum computing course at NC State University. The quantum approach, while theoretically promising, was constrained by simulation resources - training required distributed processing across multiple sessions due to the computational expense of simulating 12-qubit circuits.

## Repository Structure

```
wildfire-qml/
├── classical/          # Classical ensemble implementation (81.89% BA)
├── quantum/            # Quantum variational circuit implementation (~50% BA)
├── data/               # Dataset information and preprocessing notes
├── paper/              # Research paper and detailed analysis
└── README.md           # This file
```

## Quick Start

### Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended for quantum models)
- Dataset: [US Wildfire Dataset (2014-2025)](https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset)

### Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/wildfire-qml.git
cd wildfire-qml

# Install dependencies
pip install -r requirements.txt
```

### Running Classical Models

```bash
cd classical

# Preprocess data
python preprocess.py

# Train ensemble (12 models: 6 XGBoost + 6 LightGBM)
python train_ensemble.py

# Optimize thresholds
python optimize_ensemble.py
```

### Running Quantum Models

**Simple version (single model demonstration):**
```bash
cd quantum

# Preprocess data
python preprocess.py

# Train a single quantum model
python train_model.py

# Evaluate ensemble (if you've trained multiple models)
python evaluate_ensemble.py
```

**Advanced version (full 12-model ensemble):**
See `quantum/advanced/README.md` for distributed training instructions.

## Results Summary

| Model Type | Balanced Accuracy | Overall Accuracy | Recall | Specificity |
|------------|-------------------|------------------|--------|-------------|
| **Classical Ensemble** | **81.89%** | 76.47% | 86.86% | 76.92% |
| **Quantum Ensemble** | 52.88% | 17.50% | 92.45% | 13.31% |

The classical ensemble significantly outperformed the quantum approach. Key findings:

### Classical Success Factors
- Gradient boosting handles high-dimensional tabular data effectively
- Ensemble diversity through varied hyperparameters
- Efficient training on full 9.7M sample dataset
- F1-weighted ensemble strategy optimized performance

### Quantum Limitations
- **Computational constraints**: 12-qubit simulation too expensive for continuous training
- **Data starvation**: Fragmented training sessions (5k samples/session) prevented convergence
- **Barren plateaus**: Random initialization led to flat loss landscapes
- Models clustered around 50% BA (chance performance) regardless of hyperparameters

### Key Insight
The quantum approach's failure was primarily resource-limited, not algorithmic. With access to actual quantum hardware or larger-scale simulation, the theoretical advantages of quantum entanglement for capturing complex meteorological interactions could be realized.

## Methodology

### Classical Approach
- **Feature Engineering**: 64+ derived features including rolling averages, interaction terms, seasonal encoding
- **Models**: 12 gradient boosting models with diverse hyperparameters
- **Optimization**: F1-weighted ensemble with threshold tuning (0.48 for balanced accuracy)

### Quantum Approach  
- **Circuit Architecture**: 12 qubits (6 data + 6 processing), 2-3 layer variational circuit
- **Encoding**: Amplitude embedding for efficient quantum state preparation
- **Training**: Class-weighted MSE loss with gradient descent
- **Ensemble**: 12 models with varied learning rates and circuit depths

## Dataset

**Source**: [US Wildfire Dataset on Kaggle](https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset)

**Features**:
- Meteorological variables (temperature, humidity, wind, precipitation)
- Fire danger indices (ERC, BI, fuel moisture)
- Temporal and geographic information
- ~9.7 million samples with ~5% positive class (wildfires)

**Note**: Dataset not included in repository due to size. Download and place `Wildfire_Dataset.csv` in the `data/` directory before running preprocessing.

## Project Structure Details

### Classical Directory
- `preprocess.py` - Feature engineering and train/val/test splitting
- `train_ensemble.py` - Trains 12 gradient boosting models
- `optimize_ensemble.py` - Threshold and ensemble strategy optimization

### Quantum Directory
- `preprocess.py` - Feature engineering with 64-feature padding for quantum encoding
- `train_model.py` - Simplified single-model training
- `evaluate_ensemble.py` - Ensemble evaluation with cached predictions
- `optimize_ensemble.py` - Threshold optimization (requires re-running circuits)
- `advanced/` - Distributed training system for full 12-model ensemble

### Paper
- `Quantum_Final_Project.pdf` - Complete analysis, methodology, and discussion

## Hardware Requirements

### Classical Models
- **Minimum**: 8GB RAM, 2+ CPU cores
- **Recommended**: 16GB RAM, 4+ CPU cores
- **Training Time**: ~30-60 minutes for full ensemble

### Quantum Models
- **Minimum**: 16GB RAM, 4+ CPU cores
- **Recommended**: 32GB RAM, 8+ CPU cores, WSL2 (for distributed training)
- **Training Time**: ~2-4 hours per model for 4 sessions (8-16 hours total for simple version)
- **Full Ensemble**: ~48+ hours with distributed training

## Future Directions

1. **Quantum Hardware Access**: Test on actual quantum computers to eliminate simulation overhead
2. **Improved Initialization**: Use classical pre-training or transfer learning for better parameter initialization
3. **Hybrid Approaches**: Combine classical feature extraction with quantum classification layers
4. **Advanced Ansätze**: Explore hardware-efficient ansätze designed for NISQ devices

## Citation

If you use this code or reference this work, please cite:

```
Trepanier, J., Aycock, P., & Ayala-Lagunas, A. (2025). 
Wildfire Prediction Using Quantum Machine Learning. 
NC State University Graduate Quantum Computing Course.
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Course: ECE 569 - Quantum Computing, NC State University
- Dataset: FirecastRL US Wildfire Dataset
- Libraries: PennyLane, XGBoost, LightGBM, scikit-learn

## Contact

For questions or collaboration:
- Preston Aycock - pcaycock@ncsu.edu
- Repository: [github.com/[your-username]/wildfire-qml](https://github.com/[your-username]/wildfire-qml)

---

**Note**: This project demonstrates both the potential and current limitations of quantum machine learning for real-world classification tasks. While the classical approach proved more effective given current computational constraints, the quantum methodology provides a foundation for future work as quantum hardware becomes more accessible.
