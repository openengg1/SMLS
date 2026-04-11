# Preprocessing Pipeline Guide

## Overview

This folder contains a **clean preprocessing pipeline** for preparing spray parcel training data. The key innovation: **no synthetic disappearing particle samples**, which means we're working with real physical data only.

### Problem We Solved

The original preprocessing pipeline (`preprocess/prepare_gnn_data_simple.py`) had a critical bug:
- When a particle disappeared (had no t+1 data), it created a **synthetic sample** with all outputs set to zero
- These fake samples were concatenated into the training data
- Result: ~50% of training data was artificial, contaminating the clustering and model training

**This pipeline fixes that problem by keeping ONLY real t→t+1 transitions.**

---

## Pipeline Structure

```
preprocessing/
├── prepare_gnn_data_clean.py           # Phase 1: Clean data preparation
├── create_physics_clusters.py          # Phase 2: Physics-informed clustering
├── run_preprocessing_pipeline.py       # Master orchestration script
├── PREPROCESSING_GUIDE.md              # This file
└── data/                               # Output directory (created on first run)
    ├── train_paired.csv                # Training data
    ├── val_paired.csv                  # Validation data
    ├── test_paired.csv                 # Test data
    ├── gmm_model.pkl                   # Fitted 3-cluster GMM
    └── metadata.pkl                    # Scalers and feature metadata
```

---

## Phase 1: Data Preparation (`prepare_gnn_data_clean.py`)

### Input
- Path: `../../data/case1/processed_data/step3_with_injection_labels.csv`
- Format: Preprocessed spray data with droplet properties and field values
- Columns: `timestep`, `origId`, `d`, `nParticle`, `T`, `U:0/1/2`, `euler_*`, `rho`, `mu`, `sigma`, `Points:*`

### Processing

1. **Load Data**
   - Reads input CSV
   - Checks for missing columns or malformed data

2. **Create Paired Timesteps**
   - Sorts all particles by `(origId, timestep)`
   - Creates pairs where: same `origId` appears at time `t` and `t+1`
   - **Excludes** particles that disappear (no t+1 data)
   - Computes deltas: `Δfeature = feature(t+1) - feature(t)`

3. **Compute Mass Proxy**
   - Formula: `mass_proxy = d³ × nParticle / 1e12`
   - Represents parcel mass without numerical overflow

4. **Create Train/Val/Test Splits**
   - Splits by **particle ID** (not individual samples)
   - Default: 70% train, 15% val, 15% test
   - Prevents temporal leakage (same particle doesn't cross splits)

5. **Normalize Features**
   - StandardScaler fitted **ONLY on training data**
   - Applied to validation and test to prevent data leakage
   - Positions (`Points:*`) NOT normalized (kept in physical units)

### Output
Three CSV files in `./data/`:

#### `train_paired.csv` / `val_paired.csv` / `test_paired.csv`

**Key Feature: Dual-Format Columns**

Each CSV contains BOTH normalized (for training) and absolute/denormalized (for clustering) versions:

| Column Type | Columns | Purpose | Example |
|-------------|---------|---------|---------|
| **Metadata** | `origId`, `timestep`, `timestep_next` | Sample identification | — |
| **Normalized** | `in_d`, `in_T`, ... (17 cols) | Scaled inputs (μ≈0, σ≈1) | -0.523 |
| **Absolute** | `in_d_abs`, `in_T_abs`, ... (17 cols) | Real physical values | 1.234e-5 m |
| **Normalized** | `out_d`, `out_T`, ... (6 cols) | Scaled deltas | 0.156 |
| **Absolute** | `out_d_abs`, `out_T_abs`, ... (6 cols) | Real delta values | 1.2e-6 m |
| **Positions** | `pos_Points:0`, ... (3 cols) | Spatial coords (NOT scaled) | 0.001 m |
| **Clustering** | `cluster_id_physics` | Cluster ID (0/1/2) | After Phase 2 |

**Benefits:**
- ✓ **No denormalization needed**: Both forms in the same CSV
- ✓ **Training uses normalized**: `in_*` and `out_*` columns
- ✓ **Clustering uses absolute**: `in_*_abs` and `out_*_abs` columns
- ✓ **Analysis ready**: Absolute values for physics interpretation
- ✓ **Single file**: Everything in one place, no extra preprocessing

#### `metadata.pkl`

Dictionary containing:
- `scaler_input`: StandardScaler fitted on training inputs
- `scaler_output`: StandardScaler fitted on training outputs
- `feature_cols`: Lists of input/output column names
- `n_samples`: Train/val/test counts
- `data_type`: `'clean_persistent_only'`

---

## Phase 2: Physics-Informed Clustering (`create_physics_clusters.py`)

### Input
- Phase 1 outputs: `train_paired.csv`, `val_paired.csv`, `test_paired.csv`
- Contains BOTH normalized and absolute (_abs) columns

### Processing

1. **Read Absolute Value Columns**
   - Directly uses `in_*_abs` and `out_*_abs` columns
   - No denormalization step needed
   - Faster and simpler than computing inverse_transform

2. **Compute Physics Features (7D)**
   ```
   Δ nParticle     - From out_nParticle_abs
   Δ T             - From out_T_abs
   Δ d             - From out_d_abs
   Δ Urel_mag      - Computed from out_U:*_abs
   Re              - Reynolds: (ρ·U·d/μ) from in_*_abs
   We              - Weber: (ρ·U²·d/σ) from in_*_abs
   Urel_mag        - Velocity magnitude from in_U:*_abs
   ```

3. **Fit 3-Component Gaussian Mixture Model**
   - Trains on training data only
   - 10 random initializations, up to 100 iterations
   - Uses BIC/AIC for model assessment

4. **Assign Clusters**
   - Predicts cluster ID for each sample in train/val/test
   - Adds `cluster_id_physics` column

### Output

Updated CSV files with new column:

| Column | Type | Description |
|--------|------|-------------|
| ... (all Phase 1 columns) | ... | ... |
| `cluster_id_physics` | int | Cluster ID (0, 1, or 2) |

Also saves:
- `gmm_model.pkl`: Fitted GMM model + feature names

### Cluster Interpretation

The 3 clusters typically represent different spray regimes:
- **Cluster 0**: Stable evaporation (high evaporation rate)
- **Cluster 1**: Chaotic/breakup regime (high variance)
- **Cluster 2**: Moderate evaporation (intermediate behavior)

(Actual interpretation depends on data - check cluster stats after running!)

---

## How to Use

### Option 1: Run Full Pipeline (Recommended)

```bash
cd model/suc/preprocessing/
python run_preprocessing_pipeline.py
```

This will:
1. Prepare clean paired data in `./data/`
2. Compute physics clusters
3. Save clustered datasets ready for training

### Option 2: Run Phases Separately

Phase 1 only:
```bash
python prepare_gnn_data_clean.py
```

Then Phase 2:
```bash
python create_physics_clusters.py
```

### Option 3: Custom Input Path

```bash
python run_preprocessing_pipeline.py \
  --input-csv /path/to/input.csv \
  --output-dir /path/to/output/
```

### Option 4: Skip Clustering

```bash
python run_preprocessing_pipeline.py --skip-cluster
```

---

## Key Design Decisions

### 1. **No Disappearing Particles**
Only real t→t+1 transitions are included. No synthetic "particle evaporated" samples.

**Rationale:**
- Training on fake samples biases the model
- The true evaporation physics is captured in continuous transitions
- Clustering on clean data reveals real physical regimes, not preprocessing artifacts

### 2. **Cluster on Absolute Features**
Features are denormalized before clustering.

**Rationale:**
- Normalized values are for model training (scale-free)
- Physics interpretation requires absolute values (actual evaporation rates, temp changes, etc.)
- Domain experts understand "particle lost 20 units" more than "z-score -0.5"

### 3. **Split by Particle ID**
Train/val/test splits group all timesteps of the same particle together.

**Rationale:**
- Temporal leakage: if consecutive timesteps (t, t+1) are in different splits, validation becomes unreliable
- Physical continuity: droplet evolution should stay together

### 4. **Normalize After Splitting**
Train/val/test splits are created on raw data, then normalized using training stats.

**Rationale:**
- Prevents data leakage: val/test statistics don't influence train normalizer
- Realistic: in production, we only have training data to set scaling

---

## Data Flow Diagram

```
Raw Input CSV
    ↓
[Phase 1] Prepare Paired Data
    • Load & sort by (origId, timestep)
    • Create pairs: sample(t) → deltas to sample(t+1)
    • Exclude disappearing particles
    • Compute mass_proxy
    ↓
Unnormalized Paired Data
    ↓
[Intermediate] Split Train/Val/Test
    (by particle ID)
    ↓
Raw Train / Val / Test
    ↓
[Normalization] Fit scaler on train, apply to all
    ↓
Normalized Train / Val / Test
    ↓
[Phase 2] Compute Physics Features
    • Denormalize outputs/inputs
    • Calculate Re, We, physics deltas
    ↓
Physics Features (7D)
    ↓
[Clustering] Fit 3-component GMM
    ↓
Clustered Train / Val / Test
    + GMM Model
    ↓
✓ Ready for Model Training!
```

---

## Output Data Statistics

After running the full pipeline, you should see:

### Phase 1 Output
```
Train: 500,000 samples from 2,500 particles
Val:   107,000 samples from 540 particles
Test:  107,000 samples from 540 particles
```
(Exact numbers vary with input data)

### Phase 2 Output
```
Cluster 0: 300,000 samples (60%)
Cluster 1:  50,000 samples (10%)
Cluster 2: 157,000 samples (30%)
```
(Typical distribution for evaporation-dominated spray)

---

## Troubleshooting

### Error: "Input file not found"
- Check that `step3_with_injection_labels.csv` exists
- Path is relative to `./preprocessing/`
- Use `--input-csv` to specify absolute path

### Error: "No paired data created"
- Input CSV might not have required columns
- Check for `timestep`, `origId`, `d`, `nParticle`, `T`, `U:0/1/2`
- Run sample of CSV: `head -20 ../../../data/case1/processed_data/step3_with_injection_labels.csv`

### Clustering Results Look Off
- Check cluster statistics: `pandas.read_csv('./data/train_paired.csv'); print(df.groupby('cluster_id_physics').size())`
- Verify denormalization is working: examine raw physics feature values
- Try different n_clusters: edit `run_preprocessing_pipeline.py` line with `n_clusters=3`

### Memory Error on Large Data
- Reduce chunk_size in `prepare_gnn_data_clean.py` line 129
- Process only subset of timesteps first

---

## Next Steps

After preprocessing, use the output data for:

1. **Model Training** (Hybrid MOE with cluster routing)
   - Load from `./data/train_paired.csv`, `./data/val_paired.csv`
   - Use `cluster_id_physics` for gating network
   - Train separate experts per cluster

2. **Data Analysis** (Understand cluster properties)
   - Load clustered CSVs
   - Compute per-cluster statistics
   - Visualize feature distributions

3. **Inference** (Predict on new data)
   - Load `gmm_model.pkl` for cluster assignment
   - Route through appropriate expert
   - Denormalize predictions using `scaler_output`

---

## References

- **Input Format**: `step3_with_injection_labels.csv` from `data/case1/processed_data/`
- **Clustering Method**: Gaussian Mixture Model (sklearn)
- **Physics Features**: Reynolds number, Weber number (dimensionless spray parameters)
- **Denormalization**: StandardScaler.inverse_transform()

