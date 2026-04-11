# Pre-Processing Verification Checklist

Before creating full processed data, verify you have everything needed for **clustering** and **training**.

---

## ✓ 1. Input Features (17 total)

The preprocessing captures **17 input features** per droplet:

```python
self.input_features = [
    'd',                                    # Droplet diameter (m)
    'U:0', 'U:1', 'U:2',                   # Droplet velocity components (m/s)
    'T',                                    # Droplet temperature (K)
    'nParticle',                            # Parcels per droplet (count)
    
    'rho',                                  # Gas density at droplet location (kg/m³)
    'mu',                                   # Gas dynamic viscosity (Pa·s)
    'sigma',                                # Surface tension (N/m)
    
    'euler_T',                              # Eulerian field temperature (K)
    'euler_U:0', 'euler_U:1', 'euler_U:2', # Eulerian field velocity (m/s)
    'euler_H2O',                            # Water vapor concentration (kg/m³)
    'euler_p',                              # Pressure (Pa)
    'euler_rho',                            # Field density (kg/m³)
    
    'mass_proxy'                            # Proxy mass for weighted operations
]
```

**Status**: ✓ **Defined in [raw_to_paired.py](raw_to_paired.py)**

---

## ✓ 2. Output Features (6 total)

Models predict **6 output features** at timestep t+1:

```python
self.output_features = ['d', 'U:0', 'U:1', 'U:2', 'T', 'nParticle']
```

These are the **actual values** (not deltas) at the next timestep.

**Status**: ✓ **Defined in [raw_to_paired.py](raw_to_paired.py)**

---

## ✓ 3. Clustering Features (6 absolute physics features)

The preprocessing **automatically computes** 6 physics-informed clustering features:

| Feature | Formula | Meaning |
|---------|---------|---------|
| **We_abs** | ρ × \|U\|² × d / σ | Weber: breakup intensity |
| **Re_abs** | ρ × \|U\| × d / μ | Reynolds: flow regime |
| **del_T_abs** | out_T_abs - in_T_abs | Temperature change (K) |
| **del_d_abs** | out_d_abs - in_d_abs | Diameter change (m) — evaporation |
| **delnParticle_abs** | out_nParticle_abs - in_nParticle_abs | Particle count change — breakup |
| **delUrelMag_abs** | \|U(t+1)\| - \|U(t)\| | Velocity magnitude change — acceleration **← NEW** |

**Status**: ✓ **Computed in [raw_to_paired.py](raw_to_paired.py#L315-L345)**

---

## ✓ 4. Data Normalization Strategy

Both **normalized** and **absolute** columns are saved:

### Normalized Columns (for model training)
- `in_d`, `in_U:0`, `in_U:1`, etc.
- `out_d`, `out_U:0`, `out_U:1`, etc.
- **Fitted using StandardScaler on training data only**
- Scalers saved as `metadata.pkl`

### Absolute Columns (for clustering & analysis)
- `in_d_abs`, `in_U:0_abs`, `in_U:1_abs`, etc.
- `out_d_abs`, `out_U:0_abs`, `out_U:1_abs`, etc.
- Original physical units (meters, K, m/s, etc.)
- **Used by clustering module**

**Status**: ✓ **Implemented in [raw_to_paired.py](raw_to_paired.py#L243-L280)**

---

## ✓ 5. Train/Val/Test Split

Data is split into three separate CSV files:

| Split | Purpose | Data Source |
|-------|---------|-------------|
| **train_paired.csv** | Fit scalers + train model | 70% of data |
| **val_paired.csv** | Tune hyperparameters | 15% of data |
| **test_paired.csv** | Final evaluation | 15% of data |

**Scaler Fitting**: Training data only → applied to all splits

**Status**: ✓ **Implemented in [raw_to_paired.py](raw_to_paired.py#L380-L420)**

---

## ✓ 6. Persistent Droplet Pairing

Only droplets that **persist across sequential timesteps** are paired:

```python
valid_pairs = all_data['origId'].shift(-1) == all_data['origId']
```

This ensures:
- ✓ Each pair (t → t+1) is from **same droplet**
- ✓ **No orphaned samples** or impossible transitions
- ✓ **Linear causality**: output at t+1 depends on input at t

**Status**: ✓ **Implemented in [raw_to_paired.py](raw_to_paired.py#L195-L207)**

---

## ✓ 7. Clustering Module (3-Cluster GMM)

The clustering pipeline creates **3 spray phase clusters**:

```python
Cluster Assignment Logic:
├─ Early Phase (t=0-10):    High breakup → high delnParticle_abs
├─ Peak Phase (t=21-29):    Moderate breakup → moderate delnParticle_abs  
└─ Post Phase (t≥50):       Stable → delnParticle_abs ≈ 0
```

Features used (computed from absolutes):
- ΔnParticle, ΔT, Δd
- ΔU_mag (velocity change magnitude)
- Re (Reynolds number)
- We (Weber number)
- U_mag (velocity magnitude)

**Status**: ✓ **Implemented in [create_physics_clusters.py](create_physics_clusters.py)**

---

## ✓ 8. Output Data Format

### Train/Val/Test CSVs contain:

**Metadata Columns (3)**:
- `origId` — Droplet identifier
- `timestep` — Current timestep
- `timestep_next` — Next timestep

**Normalized Features (23)**:
- `in_*` (17 input features)
- `out_*` (6 output features)

**Absolute Features (23)**:
- `in_*_abs` (17 input features, original units)
- `out_*_abs` (6 output features, original units)

**Clustering Features (6)**:
- `We_abs`, `Re_abs`, `del_T_abs`, `del_d_abs`, `delnParticle_abs`, `delUrelMag_abs`

**Position Columns (3)** — for analysis:
- `pos_Points:0`, `pos_Points:1`, `pos_Points:2`

**Total: ~58 columns**

**Status**: ✓ **Defined in [raw_to_paired.py](raw_to_paired.py#L355-L365)**

---

## ✓ 9. Metadata & Scalers

Preprocessing saves:

```
data/
├─ train_paired.csv          (normalized + absolute features)
├─ val_paired.csv
├─ test_paired.csv
├─ metadata.pkl              (scaler_input, scaler_output, feature lists, sample counts)
└─ [clustering output]
    ├─ train_paired_clustered.csv
    ├─ val_paired_clustered.csv
    └─ test_paired_clustered.csv
```

**Status**: ✓ **Implemented in [raw_to_paired.py](raw_to_paired.py#L405-L420)**

---

## ✓ 10. Pipeline Orchestration

Master script coordinates all steps:

```bash
python run_preprocessing_pipeline.py
```

**Stages**:
1. **Prepare paired data** → train/val/test CSVs
2. **Create physics clusters** → cluster assignments
3. **Merge cluster IDs** → train/val/test_clustered.csv

**Status**: ✓ **Implemented in [run_preprocessing_pipeline.py](run_preprocessing_pipeline.py)**

---

## ✓ 11. Data Quality Checks

Before full processing, verify:

- [ ] Raw Lagrangian CSV files exist in `../../data/case1/VTK/lagrangian/sprayCloud/`
- [ ] Raw Eulerian CSV files exist in `../../data/case1/VTK/eulerian/`
- [ ] At least 20+ timesteps available for training
- [ ] Droplet columns: `origId`, `d`, `U:0`, `U:1`, `U:2`, `T`, `nParticle`, `Points:0`, `Points:1`, `Points:2`
- [ ] Eulerian columns: `T`, `U:0`, `U:1`, `U:2`, `H2O`, `p`, `rho`

---

## ✓ 12. Ready for Training

Once preprocessing completes, you'll have:

✓ **For model training**:
- Normalized input features (fitted scaler → reproducible)
- Normalized output targets
- Stratified train/val/test splits

✓ **For physics-informed clustering**:
- 5 physics-based clustering features (We, Re, ΔT, Δd, ΔnParticle)
- 3-cluster GMM assignments
- Droplet phase labels (Early/Peak/Post)

✓ **For analysis & debugging**:
- Absolute physical units (meters, K, m/s)
- Droplet tracking IDs (origId)
- Position coordinates for spatial analysis

---

## Summary: Ready to Process?

**Yes, if you have:**

- [x] All 17 input features defined & validated
- [x] All 6 output features defined & validated  
- [x] 5 clustering features auto-computed
- [x] Normalization strategy defined (StandardScaler, fit on train only)
- [x] Persistent droplet pairing logic implemented
- [x] 3-cluster GMM ready for assignment
- [x] Output CSVs with normalized + absolute + clustering columns
- [x] Metadata & scalers saved for model deployment
- [x] Master pipeline script ready

**Recommendation**: Run on 5-10 timesteps first as a test:

```bash
python raw_to_paired.py --test-timesteps 0 1 2 3 4
```

Then expand to full dataset once verified.

---

**Last Updated**: March 23, 2026
