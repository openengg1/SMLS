# Data Preparation Dependencies for SUC Folder

**Question:** What files are used for getting the CSV that are in the SUC folder?

**Answer:** A multi-stage preprocessing pipeline converts raw CFD data to the training CSVs.

---

## Full Data Flow

```
RAW DATA                          PREPROCESSING                   SUC INPUT
================================================================================

data/case1/                      preprocess/                      model/suc/
├── VTK/                         ├── step1_load_lagrangian.py      └── data/
│   ├── lagrangian/              │   └─> step1_lagrangian_base.csv    ├── train_paired.csv
│   │   └── sprayCloud/          │
│   │       └── Lagrangian_*.csv ├── step3_interpolate_eulerian.py
│   │                            │   └─> step2_with_eulerian.csv
│   │
│   └── eulerian/                ├── step4_identify_injection.py
│       └── cell_*.csv (or similar)
│                                ├── step5_add_dimensionless_numbers.py
                                 │   └─> step3_with_injection_labels.csv
                                 │
                                 ├── prepare_gnn_data_simple.py
                                 │   ├─> train_paired.csv  ───┐
                                 │   ├─> val_paired.csv    ───┼──> COPY TO model/suc/data/
                                 │   └─> test_paired.csv  ────┘
                                 │
                                 └── processed_data/
                                     └── gnn_training_simple/
```

---

## Complete Preprocessing Pipeline

### **Stage 0: Raw Data (Input)**
Location: `data/case1/VTK/`

**Lagrangian files:**
```
data/case1/VTK/lagrangian/sprayCloud/
├── Lagrangian_1.csv   (timestep 1 particle snapshot: 100k+ rows)
├── Lagrangian_2.csv   (timestep 2 particle snapshot)
├── ...
└── Lagrangian_N.csv   (final timestep)

Total: ~150-300 timesteps × ~100k-1M particles = 1.95 million particle records
```

**Eulerian files:**
```
data/case1/VTK/eulerian/
├── cell_*.csv or similar (surrounding gas properties at cell locations)
└── [Structure depends on your CFD output format]
```

---

### **Stage 1: Core Preprocessing Scripts**

| Step | File | Input | Output | Purpose |
|------|------|-------|--------|---------|
| **1** | `step1_load_lagrangian.py` | `Lagrangian_*.csv` files | `step1_lagrangian_base.csv` | Combine all timesteps into single CSV |
| **2** | `step3_interpolate_eulerian.py` | step1 + Eulerian data | `step2_with_eulerian.csv` | Add gas environment properties to each particle |
| **3** | `step4_identify_injection.py` | step2 output | `step2_with_injection.csv` | Tag injection events (new drop births) |
| **4** | `step5_add_dimensionless_numbers.py` | step3 + step2 | `step3_with_injection_labels.csv` | Compute Weber, Reynolds, Ohnesorge numbers |
| **5** | `prepare_gnn_data_simple.py` | step3 output | `train/val/test_paired.csv` | Create timestep pairs, normalize, split data |

---

### **Stage 2: Master Orchestrator**

**File:** `preprocess/run_preprocessing_pipeline.py`

Runs all 5 steps in sequence:
```bash
python preprocess/run_preprocessing_pipeline.py
```

Outputs:
```
preprocess/processed_data/gnn_training_simple/
├── train_paired.csv      (~1.9M rows, ~493 MB, normalized 17D input + 7D output)
├── val_paired.csv        (~242k rows, ~95 MB)
├── test_paired.csv       (~242k rows, ~95 MB)
└── metadata.pkl          (StandardScaler objects, feature mappings)
```

---

## Supporting/Utility Scripts (Not in Main Pipeline)

These are available but NOT used by the main pipeline:

| File | Purpose | Status |
|------|---------|--------|
| `calculate_dimensionless_numbers.py` | Helper to compute We, Re, Oh numbers | Used by step 5 |
| `create_injection_events_csv.py` | Alternative injection event detection | Alternative (not default) |
| `prepare_gnn_training_data.py` | Older version of data prep | Deprecated |
| `prepare_gnn_data_simple_v2_backup.py` | Backup of simple version | Backup only |
| `phase3_structured_preprocessing.py` | Alternative approach | Experimental |
| `step2_load_eulerian.py` | Legacy (step3 does this now) | Deprecated |
| `step5_classify_events.py` | Alternative event classification | Not used |
| `step6_separate_injection_events.py` | Separate injection from regular drops | Not used |

---

## Data Dependency Tree for SUC

```
RAW INPUT                     STAGE                          STAGE                          OUTPUT
====================================================================================

Lagrangian_1.csv  ┐
Lagrangian_2.csv  │  Step 1         step1_lagrangian_base.csv
...               │  ───────────→ (1.95M particle records)
Lagrangian_N.csv  ┘

                                        Step 2
Eulerian data ────────────────────────────────→ step2_with_eulerian.csv
(cell properties)                                (add gas context)

                                        Step 3
                        ────────────────────────→ step2_with_injection.csv
                                                 (tag injection events)

                                        Step 4
                        ────────────────────────→ step3_with_injection_labels.csv
                                                 (add dimensionless numbers)

                                        Step 5
                        ────────────────────────→ train_paired.csv
                        (Prepare GNN data)       val_paired.csv
                                                 test_paired.csv
                                                 (normalized, split)

                    ┌─ COPY TO SUC
                    │
                    └─→ model/suc/data/
                        ├── train_paired.csv  ✓ Can now train SUC model
                        ├── val_paired.csv    ✓
                        └── test_paired.csv   ✓
```

---

## Files Needed to Reproduce SUC's Input Data

### **Minimum Required Files**

To regenerate the CSVs in `model/suc/data/`:

```
preprocess/
├── step1_load_lagrangian.py          ✓ Required
├── step3_interpolate_eulerian.py     ✓ Required
├── step4_identify_injection.py       ✓ Required
├── step5_add_dimensionless_numbers.py ✓ Required
├── prepare_gnn_data_simple.py        ✓ Required
├── calculate_dimensionless_numbers.py ✓ Required (used by step 5)
├── run_preprocessing_pipeline.py     ✓ Recommended (orchestrator)
└── processed_data/                   (intermediate outputs)
```

### **Raw Data Required**

```
data/case1/
├── VTK/lagrangian/sprayCloud/Lagrangian_*.csv     ✓ Essential
└── VTK/eulerian/[cell_*.csv or similar]           ✓ Essential
```

---

## Current State: Data Already Prepared

**Good news:** You don't need to run the full preprocessing pipeline.

The CSVs are already prepared and in place:

```
✓ model/suc/data/train_paired.csv       (ready to train)
✓ model/suc/data/val_paired.csv
✓ model/suc/data/test_paired.csv

You CAN directly run:
  python model/suc/train_supervised_cluster_routing.py
```

---

## If You Need to Regenerate Input Data

**From project root:**

```bash
# Run full preprocessing pipeline (takes ~30-60 minutes)
python preprocess/run_preprocessing_pipeline.py

# OR run individual steps manually
python preprocess/step1_load_lagrangian.py
python preprocess/step3_interpolate_eulerian.py
python preprocess/step4_identify_injection.py
python preprocess/step5_add_dimensionless_numbers.py
python preprocess/prepare_gnn_data_simple.py

# Then copy to SUC folder
cp preprocess/processed_data/gnn_training_simple/*.csv model/suc/data/
```

---

## Feature Engineering Details

After all preprocessing, the final CSVs contain:

**INPUT FEATURES (17D):**
- Lagrangian (6): `d`, `U:0`, `U:1`, `U:2`, `T`, `nParticle`
- Material (3): `rho`, `mu`, `sigma` (dependent on T)
- Eulerian (7): `euler_T`, `euler_U:0`, `euler_U:1`, `euler_U:2`, `euler_H2O`, `euler_p`, `euler_rho`
- Derived (1): `mass_proxy` = d³ · n_Particle / 10¹²

**OUTPUT FEATURES (7D):**
- Deltas: Δd, ΔU:0, ΔU:1, ΔU:2, ΔT, Δn_Particle
- Classification: `persists` (1 = survives to t+1, 0 = evaporates)

---

## SUC Folder Self-Containment Assessment

| Aspect | Status |
|--------|--------|
| Training code | ✅ SELF-CONTAINED (no external imports) |
| Training data CSVs | ✅ INCLUDED (already in `./data/`) |
| Feature extraction | ✅ SELF-CONTAINED (in `feature_engineering.py`) |
| Preprocessing pipeline | ⚠️ SEPARATE (in `/preprocess/` folder) |
| Raw data | ⚠️ SEPARATE (in `/data/case1/`) |

**Conclusion:** SUC is **self-contained for training**. The preprocessing pipeline is **separate and optional** since prepared CSVs are included.

---

## What's in `model/suc/data/` folder

```
model/suc/data/
├── train_paired.csv          (1.9M rows, training data with cluster_id column)
├── val_paired.csv            (242k rows, validation data)
├── test_paired.csv           (242k rows, test data)
└── metadata.pkl              (scalers + feature mappings from preprocessing)
```

These are **copies** of preprocessed data from:
```
preprocess/processed_data/gnn_training_simple/
```

The preprocessing stage is **independent of SUC** and can be run separately if you need to regenerate the data from raw CFD outputs.
