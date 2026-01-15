# Dataset Information

## Overview

This project uses the **US Wildfire Dataset (2014-2025)** from Kaggle, containing meteorological records and wildfire occurrence data for wildfire prediction.

## Download Instructions

**Dataset Source**: [US Wildfire Dataset on Kaggle](https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset)

### Steps to Download

1. **Create a Kaggle account** (if you don't have one): https://www.kaggle.com/

2. **Download the dataset**:
   - Visit: https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset
   - Click "Download" button
   - Extract the ZIP file

3. **Place the CSV file**:
   ```bash
   # Place the extracted CSV in this directory
   wildfire-qml/data/Wildfire_Dataset.csv
   ```

**File size**: ~1-2 GB (CSV format)

**Note**: The dataset is NOT included in this repository due to its size. You must download it separately before running any preprocessing or training scripts.

## Dataset Description

### Samples
- **Total records**: ~9.7 million meteorological observations
- **Date range**: 2014-2025
- **Geographic coverage**: United States
- **Class distribution**: ~5% wildfire events (severe class imbalance)

### Target Variable
- `Wildfire`: Binary classification
  - `1` / `Yes` / `True` = Wildfire ignition occurred
  - `0` / `No` / `False` = No wildfire

### Features

#### Meteorological Variables
- `tmmx` - Maximum temperature (°F)
- `tmmn` - Minimum temperature (°F)
- `pr` - Precipitation (mm)
- `rmax` - Maximum relative humidity (%)
- `rmin` - Minimum relative humidity (%)
- `sph` - Specific humidity (kg/kg)
- `vs` - Wind speed (m/s)
- `th` - Wind direction (degrees)
- `pdsi` - Palmer Drought Severity Index
- `vpd` - Vapor pressure deficit (kPa)
- `srad` - Solar radiation (W/m²)

#### Fire Danger Indices
- `erc` - Energy Release Component
- `bi` - Burning Index
- `fm100` - 100-hour fuel moisture (%)
- `fm1000` - 1000-hour fuel moisture (%)

#### Geographic & Temporal
- `latitude` - Location latitude
- `longitude` - Location longitude
- `datetime` - Timestamp of observation

### Feature Engineering

Both classical and quantum models apply extensive feature engineering to this raw data:

**Derived Features**:
- Temperature range, average, variance
- Humidity range, average, dryness index
- Physical interactions (VPD×temperature, wind×ERC, etc.)
- Fuel moisture ratios and differences

**Temporal Features**:
- 3-day and 7-day rolling averages
- Lag features (previous day values)
- Rate of change (day-to-day deltas)
- Seasonal encoding (sin/cos of day-of-year)
- Fire season indicator (May-October)

**Final Feature Count**:
- Classical: 64+ features (variable based on engineering)
- Quantum: Exactly 64 features (required for 6-qubit amplitude embedding)

## Data Splits

Both models use stratified train/validation/test splits to preserve class balance:

- **Training**: 70% (~6.8M samples)
- **Validation**: 15% (~1.5M samples)  
- **Test**: 15% (~1.5M samples)

Stratification ensures the ~5% wildfire rate is maintained across all splits.

## Preprocessing

Each model has its own preprocessing script:

### Classical Preprocessing
```bash
cd classical
python preprocess.py
```
- Applies full feature engineering
- Creates train/val/test splits
- Scales features using StandardScaler
- Saves to `classical/preprocessed/`

### Quantum Preprocessing
```bash
cd quantum
python preprocess.py
```
- Applies same feature engineering as classical
- Pads/truncates to exactly 64 features (2^6 for amplitude embedding)
- Creates stratified splits
- Shards data across 12 models (for distributed training)
- Saves to `quantum/preprocessed/model_X.pkl`

## Data Usage Notes

### Class Imbalance
The severe class imbalance (~5% fires) requires special handling:
- **Classical**: Uses class weights in XGBoost/LightGBM
- **Quantum**: Uses class-weighted loss function (higher weight for fires)
- Both use stratified splitting to maintain balance

### Memory Considerations
- Full dataset: ~10M samples × 64 features × 4 bytes/float = ~2.5 GB in memory
- Classical: Trains on full dataset using efficient gradient boosting
- Quantum: Trains on 1/12 shards per model (~200-800 MB per shard)

### Temporal Ordering
- Data must be sorted by location then timestamp for rolling features
- Both preprocessing scripts handle this automatically
- Important: Rolling windows are grouped by (latitude, longitude) to prevent spatial leakage

## File Structure After Preprocessing

```
data/
├── README.md                           # This file
├── Wildfire_Dataset.csv                # Raw data (you download this)
│
├── ../classical/preprocessed/          # Classical preprocessed data
│   ├── train_data.pkl
│   ├── val_data.pkl
│   └── test_data.pkl
│
└── ../quantum/preprocessed/            # Quantum preprocessed data
    ├── model_1.pkl                     # Shard for model 1
    ├── model_2.pkl
    ├── ...
    ├── model_12.pkl
    └── metadata.pkl                    # Scaler and feature info
```

## Citation

If you use this dataset, please cite the original source:

```
FirecastRL. (2025). US Wildfire Dataset (2014-2025). 
Kaggle. https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset
```

## Troubleshooting

### "File not found: Wildfire_Dataset.csv"
- Make sure you downloaded the dataset from Kaggle
- Verify the file is named exactly `Wildfire_Dataset.csv`
- Check it's in the `data/` directory

### "Out of memory during preprocessing"
- Close other applications
- Preprocessing is memory-intensive due to rolling window calculations
- Consider using a machine with 16GB+ RAM
- Alternatively, modify preprocessing scripts to process in chunks

### "Preprocessing takes too long"
- Expected time: 10-30 minutes depending on hardware
- Rolling feature calculations are computationally expensive
- Progress is logged to console during processing
- Be patient - this only needs to be done once

## Questions?

For dataset-specific questions, refer to the Kaggle dataset page or contact the dataset authors.

For preprocessing questions related to this project, see the main repository README or open an issue.
