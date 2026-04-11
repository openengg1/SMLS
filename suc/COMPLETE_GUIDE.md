# Stochastic ML Spray - SUC Module Complete Guide

**Status:** Complete, self-contained, production-ready (March 27, 2026)

This guide provides **step-by-step instructions** to run the complete pipeline from raw CFD data to trained model and publication-ready plots.

---

## Table of Contents

1. [Setup & Prerequisites](#setup--prerequisites)
2. [Quick Start (5 minutes)](#quick-start-5-minutes)
3. [Complete Workflow (2-3 hours)](#complete-workflow-2-3-hours)
4. [Detailed Stage-by-Stage Guide](#detailed-stage-by-stage-guide)
5. [Training in Detail](#training-in-detail)
6. [Analysis & Plotting](#analysis--plotting)
7. [Architecture Overview](#architecture-overview)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### **Option 1: Train with Existing Data (5-10 minutes)**
```bash
cd model/suc
python train_supervised_cluster_routing.py
```

### **Option 2: Full Reproducibility from Raw Data (1-2 hours)**
```bash
cd model/suc
bash run_full_pipeline_from_raw.sh
```

### **Option 3: Workflow with Clustering (if needed)**
```bash
cd model/suc
bash run_suc_workflow.sh
```

---

## Architecture Overview

### **Supervised Cluster Routing (SUC) - Physics-Informed Mixture of Experts**

**Core Design:**
- **Clustering:** Unsupervised GMM on 7D physics features → ground truth cluster labels
- **Experts:** 4 separate networks, each trained on cluster-specific data
- **Gating:** Classifier network (CrossEntropyLoss) learns to predict cluster from 17D inputs
- **Routing:** Hard selection - each sample routed to exactly one expert

**Model Architecture:**
| Component | Input | Hidden | Output |
|-----------|-------|--------|--------|
| **Gating** | 17D | 128→64 | 4 logits (cluster prediction) |
| **Expert 0** | 17D | 128→64 | 6D (property deltas) |
| **Expert 1** | 17D | 128→64 | 6D (property deltas) |
| **Expert 2** | 17D | 128→64 | 6D (property deltas) |
| **Expert 3** | 17D | 128→64 | 6D (property deltas) |

**Performance (Test Set - 242K samples):**
```
Gating Accuracy:     99.42% (cluster prediction)
Expert 0 R²:         0.9875 (265K small-drop samples)
Expert 1 R²:         0.9990 (1.674M large-drop samples)
Ensemble R²:         0.9974 (overall performance)
```

---

## Folder Structure

```
model/suc/ (16 GB - Fully Self-Contained)
│
├── TRAINING WORKFLOW (Production Ready)
│   ├── train_supervised_cluster_routing.py     ⭐ Main entry point
│   ├── add_clustering_to_csv.py               (Preprocessing helper)
│   ├── hybrid_ruc_supervised.py               (Model class)
│   ├── feature_engineering.py                 (17D feature extraction)
│   ├── run_suc_workflow.sh                    (Orchestration script)
│   └── run_full_pipeline_from_raw.sh          (Complete RAW→MODEL)
│
├── DATA (Prepared Training Data - ~700 MB)
│   ├── train_paired.csv                       (1.9M rows, normalized)
│   ├── val_paired.csv                         (242k rows, normalized)
│   ├── test_paired.csv                        (242k rows, normalized)
│   └── metadata.pkl                           (Scalers, feature mappings)
│
├── RAW_DATA (CFD Simulation Output - 5.4 GB) [NEW]
│   └── VTK/
│       ├── lagrangian/sprayCloud/             (Lagrangian_1.csv → Lagrangian_N.csv)
│       └── eulerian/                          (Gas field properties)
│
├── PREPROCESSING (Complete Pipeline)
│   ├── step1_load_lagrangian.py
│   ├── step3_interpolate_eulerian.py
│   ├── step4_identify_injection.py
│   ├── step5_add_dimensionless_numbers.py
│   ├── prepare_gnn_data_simple.py
│   ├── calculate_dimensionless_numbers.py
│   └── create_physics_clusters_gmm_subsampled.py
│
├── RESULTS & CHECKPOINTS
│   ├── checkpoints/suc_best_model.pt          (Trained model)
│   ├── results/scaler_*.pkl                   (Feature scalers)
│   └── results/gmm_*.pkl                      (Clustering models)
│
├── ANALYSIS
│   ├── analysis/plot_cluster_distributions.py
│   └── generate_paper_figures.py
│
└── DOCUMENTATION [THIS FILE COMBINES ALL]
    ├── README.md (original)
    ├── CLEANUP_PLAN.md (consolidated)
    ├── CLEANUP_COMPLETE.md (consolidated)
    ├── INDEPENDENCE_ANALYSIS.md (consolidated)
    ├── DATA_GENERATION_PIPELINE.md (consolidated)
    └── RAW_DATA_ADDED.md (consolidated)
```

---

## Self-Contained Status

### ✅ FULLY SELF-CONTAINED

The SUC folder now contains **everything needed** to:
1. Regenerate training data from raw CFD simulation output
2. Train the model from scratch
3. Evaluate and analyze results
4. Generate paper figures

### **External Dependencies: NONE** (except Python packages)

| Component | Location | Status |
|-----------|----------|--------|
| Raw CFD data | `model/suc/raw_data/VTK/` | ✅ Included (5.4 GB) |
| Preprocessing scripts | `model/suc/preprocessing/` | ✅ Included (all 7 steps) |
| Training code | `model/suc/` | ✅ Included |
| Trained model | `model/suc/checkpoints/` | ✅ Included (2 MB) |
| Prepared training data | `model/suc/data/` | ✅ Included (~700 MB) |
| Python dependencies | System packages | ✅ torch, pandas, sklearn, etc |

**What This Means:**
- ✅ Can copy `model/suc/` anywhere (self-contained)
- ✅ No need to access `data/case1/` or `preprocess/` folders
- ✅ Complete reproducibility from raw simulation data
- ✅ Verifiable results matching paper claims

---

## Data Pipeline

### **Complete Data Flow**

```
STAGE 0: Raw CFD Data
├── Input: Lagrangian_*.csv (150+ timesteps, ~100k-1M particles each)
└── Input: Eulerian fields (surrounding gas properties)

                    ↓

STAGE 1: Load Lagrangian Base
├── Script: preprocessing/step1_load_lagrangian.py
├── Input: raw_data/VTK/lagrangian/sprayCloud/Lagrangian_*.csv
└── Output: data/step1_lagrangian_base.csv (1.95M particle records)

                    ↓

STAGE 2: Interpolate Eulerian Fields
├── Script: preprocessing/step3_interpolate_eulerian.py
├── Input: step1 + raw_data/VTK/eulerian/
└── Output: data/step2_with_eulerian_features.csv (add gas context)

                    ↓

STAGE 3: Identify Injection Events
├── Script: preprocessing/step4_identify_injection.py
├── Input: step2 output
└── Output: data/step2_with_injection.csv (tag new drop births)

                    ↓

STAGE 4: Add Dimensionless Numbers
├── Script: preprocessing/step5_add_dimensionless_numbers.py
├── Input: step3 + helpers
├── Computed: Weber (We), Reynolds (Re), Ohnesorge (Oh)
└── Output: data/step3_with_injection_labels.csv

                    ↓

STAGE 5: Prepare Training Data
├── Script: preprocessing/prepare_gnn_data_simple.py
├── Input: step4 output
├── Operations: Normalize, create timestep pairs, split train/val/test
└── Output: 
    ├── train_paired.csv (1.9M rows, ~493 MB)
    ├── val_paired.csv (242k rows, ~95 MB)
    └── test_paired.csv (242k rows, ~95 MB)

                    ↓

STAGE 6: Add Cluster Labels (Preprocessing.io)
├── Script: add_clustering_to_csv.py
├── Input: Prepared CSVs
├── Operations:
│   ├── Sample 500K training records
│   ├── Fit GMM on 7D physics features
│   ├── Predict cluster assignments for all samples
│   └── Write cluster_id column to CSVs
└── Output: Updated CSVs with cluster_id

                    ↓

STAGE 7: Train SUC Model
├── Script: train_supervised_cluster_routing.py
├── Input: CSVs with cluster_id
├── Operations:
│   ├── Separate by cluster
│   ├── Train experts (separate for each cluster)
│   └── Train gating (classifier on cluster_id)
└── Output:
    ├── checkpoints/suc_best_model.pt
    ├── results/scaler_*.pkl
    └── results/gmm_*.pkl

                    ↓

DONE: Ready for inference, paper, deployment
```

### **Input Features (17D)**

Lagrangian (6):
- `d` - droplet diameter
- `U:0`, `U:1`, `U:2` - velocity components
- `T` - temperature
- `nParticle` - number of particles in parcel

Material (3):
- `rho` - droplet density
- `mu` - viscosity
- `sigma` - surface tension

Eulerian (7):
- `euler_T` - surrounding gas temperature
- `euler_U:0`, `euler_U:1`, `euler_U:2` - gas velocity
- `euler_H2O` - water content
- `euler_p` - pressure
- `euler_rho` - gas density

Derived (1):
- `mass_proxy` = d³ · n_Particle / 10¹² (scaled to avoid overflow)

### **Output Features (7D)**

Deltas to predict:
- `Δd` - diameter change
- `ΔU:0`, `ΔU:1`, `ΔU:2` - velocity change
- `ΔT` - temperature change
- `Δn_Particle` - parcel count change

Classification:
- `persists` - binary: 1 if survives to t+1, 0 if evaporates

---

## Complete Workflows

### **Workflow 1: Quick Training (5-10 min)**

```bash
cd model/suc

# Uses pre-prepared CSVs (already normalized and split)
python train_supervised_cluster_routing.py

# Result: Trained model in checkpoints/suc_best_model.pt
```

**Best for:**
- Quick model training
- Testing/debugging
- When CSVs are already prepared

### **Workflow 2: Full Preprocessing + Training (30-60 min)**

```bash
cd model/suc

# Runs all 7 preprocessing stages from scratch
bash run_full_pipeline_from_raw.sh

# Internally:
#  1. Load Lagrangian data
#  2. Interpolate Eulerian fields
#  3. Identify injection events
#  4. Add dimensionless numbers
#  5. Prepare training data
#  6. Add cluster labels
#  7. Train model

# Result: Complete pipeline output + trained model
```

**Best for:**
- Full reproducibility
- Verifying paper results
- Academic rigor

### **Workflow 3: Custom Preprocessing**

```bash
cd model/suc/preprocessing

# Run individual steps as needed
python step1_load_lagrangian.py
python step3_interpolate_eulerian.py
python step4_identify_injection.py
python step5_add_dimensionless_numbers.py
python prepare_gnn_data_simple.py

cd ..
python add_clustering_to_csv.py
python train_supervised_cluster_routing.py
```

**Best for:**
- Debugging individual stages
- Experimenting with preprocessing parameters
- Custom analysis

---

## Technical Details

### **SUC vs SelfOrgMOE: Key Differences**

| Aspect | SelfOrgMOE | SUC (This Work) |
|--------|-----------|-----------------|
| **Ground Truth** | Only regression targets (y_delta) | Explicit cluster labels |
| **Gating Output** | Soft weights (softmax) | Hard selection (argmax) |
| **Gating Loss** | Implicit (MSE on regression) | Explicit (CrossEntropyLoss on cluster_id) |
| **Expert Training** | Mixed with soft weights | Separate per-cluster data |
| **Routing** | Soft (all experts contribute) | Hard (one expert per sample) |

### **Preprocessing vs Training Features**

**Key Design Insight:**

**Clustering Input (7D physics):**
- Includes `del_nParticle` - available at preprocessing time
- Used to create ground truth cluster labels
- Captures breakup vs evaporation physics

**Training Input (17D raw):**
- Only raw particle + gas properties
- No label-based features
- All available at inference time

**This Design Allows:**
- Preprocessing: Use rich physics information to cluster (7D)
- Training: Learn gating to predict clusters from deployed features (17D)
- Inference: Only need 17D inputs, gating predicts cluster → use expert

### **Feature Normalization**

**StandardScaler Applied To:**
- All 17D input features
- All 7D output deltas
- Fitted on training data only
- Applied to val/test consistently

**Saved in:** `results/scaler_*.pkl`

### **Clustering Details**

**Algorithm:** Gaussian Mixture Model (GMM)
- **Components:** 4 clusters (physics-based)
- **Fitting:** 500K sample from training data
- **Features:** 7D physics features
- **Ground Truth:** Cluster assignments written to CSV `cluster_id` column

**Cluster Interpretation:**
- **Cluster 0:** Small droplets (d < threshold)
- **Cluster 1:** Large droplets (d > threshold)
- **Cluster 2:** Evaporating droplets
- **Cluster 3:** Coalescencing droplets

(Exact physics interpretation depends on We, Re, Oh distributions)

---

## Files Reference

### **Core Training Scripts**

| File | Purpose | Usage |
|------|---------|-------|
| `train_supervised_cluster_routing.py` | Main training entry point | `python train_supervised_cluster_routing.py` |
| `add_clustering_to_csv.py` | Add cluster labels via GMM | `python add_clustering_to_csv.py` |
| `hybrid_ruc_supervised.py` | Model class definition | (imported by training script) |
| `feature_engineering.py` | 17D feature extraction | (imported by training script) |

### **Preprocessing Scripts**

| File | Purpose |
|------|---------|
| `preprocessing/step1_load_lagrangian.py` | Load all Lagrangian CSVs |
| `preprocessing/step3_interpolate_eulerian.py` | Add Eulerian context |
| `preprocessing/step4_identify_injection.py` | Tag injection events |
| `preprocessing/step5_add_dimensionless_numbers.py` | Compute We, Re, Oh |
| `preprocessing/prepare_gnn_data_simple.py` | Normalize & split data |
| `preprocessing/calculate_dimensionless_numbers.py` | Helper for We, Re, Oh |
| `preprocessing/create_physics_clusters_gmm_subsampled.py` | GMM clustering |

### **Orchestration Scripts**

| File | Purpose |
|------|---------|
| `run_suc_workflow.sh` | Quick: clustering + training |
| `run_full_pipeline_from_raw.sh` | Complete: preprocessing + training |

### **Analysis & Documentation**

| File | Purpose |
|------|---------|
| `analysis/plot_cluster_distributions.py` | Visualize cluster properties |
| `generate_paper_figures.py` | Generate publication figures |
| `COMPLETE_GUIDE.md` | THIS FILE - all documentation |

### **Data Files**

| Location | Content | Size |
|----------|---------|------|
| `data/train_paired.csv` | Training data (1.9M rows) | 493 MB |
| `data/val_paired.csv` | Validation data (242k rows) | 95 MB |
| `data/test_paired.csv` | Test data (242k rows) | 95 MB |
| `data/metadata.pkl` | Scalers & mappings | 1 MB |
| `raw_data/VTK/lagrangian/` | Lagrangian snapshots | 5.2 GB |
| `raw_data/VTK/eulerian/` | Eulerian fields | 0.2 GB |

### **Model Outputs**

| Location | Content | Size |
|----------|---------|------|
| `checkpoints/suc_best_model.pt` | Trained model weights | 2 MB |
| `results/scaler_*.pkl` | Feature scalers (input/output) | 100 KB |
| `results/gmm_*.pkl` | GMM clustering models | Various |

---

## History of Changes

### **March 24, 2026: Initial Setup**
- ✅ Fixed feature documentation (sections 2.1.2 & 2.1.3)
- ✅ Added figure discussions
- ✅ Fixed equation overflows in LaTeX

### **March 24, 2026: Folder Cleanup**
- ✅ Archived old training variants to `../suc_archive/`
- ✅ Cleaned up test scripts
- ✅ Consolidated development files
- ✅ Created `CLEANUP_COMPLETE.md`

### **March 26, 2026: Raw Data Addition**
- ✅ Copied raw CFD data (`data/case1/VTK/` → `model/suc/raw_data/`)
- ✅ Copied preprocessing scripts to `model/suc/preprocessing/`
- ✅ Updated all hardcoded paths to relative paths
- ✅ Created `run_full_pipeline_from_raw.sh`

### **March 26, 2026: Documentation Consolidation**
- ✅ Merged all markdown files into single `COMPLETE_GUIDE.md`
- Reduced clutter while maintaining all information

---

## Size Summary

| Component | Size | Notes |
|-----------|------|-------|
| Raw CFD data | 5.4 GB | VTK Lagrangian + Eulerian files |
| Prepared training data | ~700 MB | Normalized train/val/test CSVs |
| Preprocessing scripts | 100 KB | 7 main + helpers |
| Training code | 50 KB | Model + feature engineering |
| Trained model + scalers | 2 MB | Checkpoints + artifacts |
| **Total** | **~6.1 GB** | Fully reproducible |

---

## Troubleshooting

### **Issue: "Data files not found"**
**Solution:** Ensure you're running from `model/suc/` directory
```bash
cd model/suc
python train_supervised_cluster_routing.py
```

### **Issue: "cluster_id column not found"**
**Solution:** Run preprocessing first
```bash
python add_clustering_to_csv.py
```

### **Issue: CUDA out of memory**
**Solution:** Edit script to set `device='cpu'` at top of script

### **Issue: Raw data not found for full pipeline**
**Solution:** Check that `raw_data/VTK/` exists with Lagrangian files
```bash
ls model/suc/raw_data/VTK/lagrangian/sprayCloud/Lagrangian_*.csv
```

---

## Citation & References

**If you use this work, please cite:**
```
Mishra, R., Narayanan, A., Sachdev, K. (2026). 
"A Stochastic Machine Learning Surrogate Model for Comprehensive 
Replacement of Spray Submodels." 
ILASS-Americas 36th Annual Conference.
```

**Key Papers Referenced:**
- GMM clustering approach
- Dimensionless number physics
- Mixture of Experts topology
- Spray simulation theory

---

## Contact & Support

**Author:** Rohit Mishra  
**Email:** rmishra@tamu.edu  
**Institution:** Texas A&M University  
**Department:** Mechanical Engineering

---

**Document Last Updated:** March 26, 2026  
**Status:** Complete & Verified  
**Version:** 1.0 (Consolidated from 6 separate documents)
