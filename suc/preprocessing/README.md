# SU-C Preprocessing Module

## Overview

**SU-C** = Supervised Unsupervised Clustering

This preprocessing module prepares **clean spray parcel training data** from raw CFD outputs, currently focusing on data preparation and physics-informed clustering.

## Quick Start

### 1. Run Full Preprocessing Pipeline

```bash
cd model/suc/preprocessing/
python run_preprocessing_pipeline.py
```

This will:
- ✓ Load raw data from `../../data/case1/processed_data/step3_with_injection_labels.csv`
- ✓ Create paired t→t+1 transitions (clean, no synthetic samples)
- ✓ Split into train/val/test by particle ID (no temporal leakage)
- ✓ Normalize features (StandardScaler on training data only)
- ✓ Compute physics features (7 dimensions)
- ✓ Fit 3-component Gaussian Mixture Model
- ✓ Assign cluster IDs to all samples
- ✓ Save processed data to `./data/` folder

### 2. Outputs Are Ready for Training

Generated files in `./data/`:
- `train_paired.csv` → Training data (cluster + features + outputs)
- `val_paired.csv` → Validation data
- `test_paired.csv` → Test data
- `gmm_model.pkl` → Fitted clustering model
- `metadata.pkl` → Scalers and feature metadata

Done! Next step: train your model using these datasets.

---

## Key Features

### ✓ Data Quality
- **No contamination**: Removes synthetic "disappearing particle" samples
- **Real physics only**: All samples are genuine t→t+1 transitions
- **Proper normalization**: Train scaler fitted only on training data

### ✓ Smart Splitting
- **By particle ID**: All timesteps of same particle stay together
- **No temporal leakage**: Validation can't see future from training
- **Balanced distribution**: 70% train, 15% val, 15% test

### ✓ Physics-Informed Clustering
- **7-dimensional features**: Reynolds, Weber, velocity, temperature, diameter, particle count changes
- **Denormalized** (absolute values for physics interpretation)
- **3 clusters**: Typically stable, chaotic, and moderate regimes

---

## File Structure

```
preprocessing/
├── prepare_gnn_data_clean.py             # Phase 1: Data preparation
├── create_physics_clusters.py            # Phase 2: Clustering
├── run_preprocessing_pipeline.py         # Master orchestration
├── PREPROCESSING_GUIDE.md                # Detailed documentation
├── README.md                             # This file
└── data/                                 # Output directory
    ├── train_paired.csv
    ├── val_paired.csv
    ├── test_paired.csv
    ├── metadata.pkl
    └── gmm_model.pkl
```

---

## What Each Script Does

### `prepare_gnn_data_clean.py`
**Input:** Raw spray data CSV  
**Output:** Clean paired data (train/val/test CSVs + metadata)

Handles:
- Loading and validating input
- Creating t→t+1 pairs for persistent particles only
- Computing feature deltas
- Normalizing with StandardScaler
- Splitting by particle ID

### `create_physics_clusters.py`
**Input:** Normalized paired data from Phase 1  
**Output:** Same data with cluster assignments

Handles:
- Denormalizing to absolute values
- Computing physics features (Re, We, ΔT, Δd, ΔnParticle, etc.)
- Fitting 3-component GMM
- Assigning cluster IDs

### `run_preprocessing_pipeline.py`
**Master script** that runs both phases sequentially.

```bash
python run_preprocessing_pipeline.py                    # Default paths
python run_preprocessing_pipeline.py --skip-cluster     # Phase 1 only
python run_preprocessing_pipeline.py --input-csv PATH   # Custom input
```

---

## Data Formats

### Input CSV (`step3_with_injection_labels.csv`)
Required columns:
- `timestep` (int): CFD timestep
- `origId` (float): Particle identifier
- `d`, `T`, `nParticle` (float): Droplet diameter, temperature, count
- `U:0`, `U:1`, `U:2` (float): Velocity components
- `rho`, `mu`, `sigma` (float): Material properties
- `euler_*` (float): Eulerian field values (temperature, velocity, moisture, pressure, density)
- `Points:0/1/2` (float): Spatial coordinates

### Output CSV (train/val/test_paired.csv)

**Both normalized AND absolute value columns in one CSV:**

| Type | Column | Description |
|------|--------|-------------|
| Metadata | `origId` | Particle ID |
| Metadata | `timestep` | Current timestep t |
| Metadata | `timestep_next` | Next timestep t+1 |
| **Normalized** | `in_*` (17) | Normalized input features at t (for training) |
| **Normalized** | `out_*` (6) | Normalized deltas t+1-t (for training) |
| **Absolute** | `in_*_abs` (17) | Absolute input values at t (for clustering/analysis) |
| **Absolute** | `out_*_abs` (6) | Absolute delta values (for clustering/analysis) |
| Positions | `pos_*` (3) | Spatial coordinates (not normalized) |
| Clustering | `cluster_id_physics` | Cluster assignment (0, 1, or 2) after Phase 2 |

**Benefits:**
- ✓ No need to denormalize during clustering
- ✓ No need to denormalize during analysis
- ✓ No extra preprocessing steps needed
- ✓ Single CSV file has everything you need

### Scalers (metadata.pkl)
Dictionary with:
- `scaler_input`: StandardScaler for 17 input features
- `scaler_output`: StandardScaler for 6 output features
- `feature_cols`: Lists of input/output column names
- `data_type`: `'clean_persistent_only'` (marks this as clean data)

---

## Configuration

### Modify Default Paths

If your data is in a different location, edit `run_preprocessing_pipeline.py`:

```python
parser.add_argument(
    '--input-csv',
    default='../../data/case1/processed_data/step3_with_injection_labels.csv',  # ← Change here
    help='Input CSV path'
)
parser.add_argument(
    '--output-dir',
    default='./data',  # ← Or here
    help='Output directory'
)
```

Or pass as command-line arguments:
```bash
python run_preprocessing_pipeline.py \
    --input-csv /absolute/path/to/input.csv \
    --output-dir /absolute/path/to/output/
```

### Modify Hyperparameters

#### Number of clusters (default: 3)
Edit `run_preprocessing_pipeline.py`:
```python
clusterer = PhysicsFeatureClusterer(
    ...
    n_clusters=3  # ← Change to 4, 5, etc.
)
```

#### Train/val/test split ratio (default: 70/15/15)
Edit `prepare_gnn_data_clean.py`, method `create_splits()`:
```python
def create_splits(self, paired_df, 
                 train_frac=0.7,    # ← Modify
                 val_frac=0.15,     # ← Modify
                 test_frac=0.15):   # ← Modify
```

#### Random seed (default: 42)
Pass to constructor:
```python
preparator = CleanGNNDataPreparator(..., random_seed=42)  # ← Change
```

---

## Example Run

```bash
$ cd model/suc/preprocessing/
$ python run_preprocessing_pipeline.py

================================================================================
STOCHASTIC ML SPRAY - PREPROCESSING PIPELINE
================================================================================

Phase 1: Data Preparation (Clean - no disappearing particles)
Phase 2: Physics-Informed Clustering (3 clusters)

--------------------------------------------------------------------------------
PHASE 1: DATA PREPARATION
--------------------------------------------------------------------------------
Loading main dataset...
  Rows: 3,200,000
  Timesteps: 0 to 123
  Unique particles: 3,270

Creating paired timesteps (CLEAN - only persistent particles)...
  Computing mass proxy feature...
    ✓ mass_proxy = d³·nP / 1e12 (range: 0.000001 to 0.450000)
  Total rows: 3,200,000
  Valid pairs (persisting particles): 2,500,000
  Shuffled 2,500,000 samples

Creating train/val/test splits (by particle)...
  Train: 1,750,000 samples from 2,289 particles
  Val:   375,000 samples from 491 particles
  Test:  375,000 samples from 490 particles

Normalizing features...
  Input features: 17
  Output features: 6
  Fitting scalers on TRAINING data only...
  ...

✓ Phase 1 complete!

--------------------------------------------------------------------------------
PHASE 2: PHYSICS-INFORMED CLUSTERING
--------------------------------------------------------------------------------
Loading data from ./data...
  Train: 1,750,000 samples
  Val:   375,000 samples
  Test:  375,000 samples
  Loaded scalers from metadata

Computing physics features (denormalized)...
  Features computed: ['Re', 'We', 'delta_T', 'delta_Urel_mag', 'delta_d', 'delta_nParticle', 'Urel_mag']
  Train shape: (1750000, 7)
  ...

Fitting 3-component GMM...
  BIC: 123456789.01
  AIC: 123456750.01
  
  Cluster distribution on training data:
    Cluster 0: 1,050,000 (60%)
    Cluster 1: 175,000 (10%)
    Cluster 2: 525,000 (30%)

✓ Phase 2 complete!

================================================================================
PREPROCESSING PIPELINE - COMPLETE
================================================================================

Output Directory: ./data

Generated Files:
  • train_paired.csv       (1,750,000 samples)
  • val_paired.csv         (375,000 samples)
  • test_paired.csv        (375,000 samples)
  • gmm_model.pkl          (3-cluster GMM model)
  • metadata.pkl           (feature names, scalers, metadata)

✓ Data is ready for model training!
================================================================================
```

---

## Troubleshooting

### Q: Where is the input data file?
**A:** Should be at `../../data/case1/processed_data/step3_with_injection_labels.csv` relative to this folder.

Check it exists:
```bash
ls -lh ../../data/case1/processed_data/step3_with_injection_labels.csv
```

If missing, update the path using `--input-csv` flag.

### Q: "ModuleNotFoundError" when running
**A:** Make sure you're in the right directory:
```bash
cd model/suc/preprocessing/
python run_preprocessing_pipeline.py
```

Or use absolute imports:
```bash
cd /home/rmishra/projects/stochasticMLSpray/model/suc/preprocessing/
python run_preprocessing_pipeline.py
```

### Q: How long does preprocessing take?
**A:** Depends on input size:
- **Phase 1** (data prep): ~5-15 minutes for 3M rows
- **Phase 2** (clustering): ~2-5 minutes for 2M samples

### Q: How much disk space?
**A:** Typically 2-3x the input file size:
- Input: ~300 MB
- Output: ~800 MB (train + val + test + metadata)

### Q: Can I run only Phase 1?
**A:** Yes:
```bash
python run_preprocessing_pipeline.py --skip-cluster
```

Then run Phase 2 separately if needed:
```bash
python create_physics_clusters.py
```

---

## Next Steps

After running preprocessing:

1. **Verify outputs**
   ```bash
   ls -lh data/
   wc -l data/train_paired.csv data/val_paired.csv data/test_paired.csv
   ```

2. **Check cluster distribution**
   ```python
   import pandas as pd
   df = pd.read_csv('data/train_paired.csv')
   print(df['cluster_id_physics'].value_counts())
   ```

3. **Train your model**
   Use `data/train_paired.csv` and `data/val_paired.csv` with your model code

4. **Denormalize predictions**
   Use `data/metadata.pkl` to load scalers for inverse_transform

---

## Documentation

For detailed explanations of the pipeline, see:
- **PREPROCESSING_GUIDE.md** — Comprehensive documentation
- **Code comments** — In-line explanations in each Python file

---

## Summary

| Step | Script | Input | Output | Time |
|------|--------|-------|--------|------|
| Phase 1 | `prepare_gnn_data_clean.py` | Raw CSV | Clean paired data | ~5-15 min |
| Phase 2 | `create_physics_clusters.py` | Phase 1 output | Clustered data | ~2-5 min |
| Orchestration | `run_preprocessing_pipeline.py` | Raw CSV | Everything | ~10-20 min |

**Key Innovation:** No synthetic samples, real physics only → cleaner clusters → better model training!

