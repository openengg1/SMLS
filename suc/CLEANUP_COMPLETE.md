# SUC Folder Cleanup - Complete ✓

**Date:** March 24, 2026  
**Status:** Successfully cleaned and archived

---

## What Was Done

### Files Archived to `../suc_archive/`

**Old Training Scripts** (`old_training/` - 352 KB total):
- 8 training variants: `train_quick.py`, `train_suc_*.py`, `train_full_data.py`, `train_working.py`, `train_expert0_optimized.py`
- 3 test scripts: `test_data_load.py`, `test_simple.py`, `cluster_8_classification.py`
- 2 experiment scripts: `run_gmm_*.py` (5 clusters, 8 clusters variants)
- 1 tool script: `regenerate_figures_large_fonts.py`
- 3 alternative analysis: `add_clustering_full_2p5m.py`, `generate_cluster_distributions.py`, `visualize_cluster_distributions.py`

**Old Preprocessing Scripts** (`old_preprocessing/`):
- `create_physics_clusters.py` - Original GMM version
- `create_physics_clusters_kmeans.py` - KMeans variant
- `diagnose_eulerian_matching.py` - Debugging script
- `find_optimal_cluster_number.py`, `find_optimal_clusters.py` - Hyperparameter tuning
- `prepare_gnn_data_clean.py` - GNN data (not used for spray work)
- `raw_to_paired.py` - Data conversion helper

**Old Logs** (`logs/`):
- All training/clustering logs (multiple runs, experiments)

---

## Production Files Remaining

### Core Model & Training (9 files)
```
model/suc/
├── hybrid_ruc_supervised.py      (13 KB)  - Model definition & hard routing
├── feature_engineering.py         (12 KB)  - 17D/20D feature extraction
├── train_supervised_cluster_routing.py (14 KB)  - Main training script
├── add_clustering_to_csv.py       (7.3 KB) - GMM clustering preprocessing
├── generate_paper_figures.py      (6.6 KB) - Paper figure generation
├── run_suc_workflow.sh            (1.8 KB) - Full workflow runner (preprocessing + training)
├── run_training.sh                (173 B)  - Training only
├── CLEANUP_PLAN.md                (4.5 KB) - Cleanup documentation
└── README.md                      (5.4 KB) - Project documentation
```

### Preprocessing Module (9 items)
```
preprocessing/
├── create_physics_clusters_gmm_subsampled.py (11.5 KB)  - Active GMM clustering
├── run_preprocessing_pipeline.py   (4.8 KB)  - Preprocessing runner
├── __init__.py                    (1.1 KB)  - Package file
├── PHASES_DOCUMENTATION.md        - Methodology docs
├── PREPROCESSING_GUIDE.md         - User guide
├── PREPROCESSING_COMPLETE.md      - Status report
├── README.md                      - Module docs
├── SETUP_SUMMARY.md               - Setup guide
├── GMM_VS_KMEANS_COMPARISON.md    - Algorithm comparison
└── PRE_PROCESSING_CHECKLIST.md    - Checklist
```

### Data & Results (preserved automatically)
```
├── data/                          (~4.7 GB) - Training/test data
│   ├── train_paired.csv
│   ├── val_paired.csv
│   ├── test_paired.csv
│   ├── metadata.pkl
│   └── injection_events.csv
├── checkpoints/
│   └── suc_best_model.pt          - Final trained model
├── results/
│   ├── scaler_11d.pkl
│   ├── scaler_11d_3clusters.pkl
│   ├── scaler_11d_5clusters.pkl
│   ├── gmm_3clusters_500k.pkl
│   ├── gmm_5clusters_1m.pkl
│   └── gmm_8clusters.pkl
├── analysis/
│   └── plot_cluster_distributions.py  - Analysis helper
└── logs/
    └── [Empty or minimal]
```

---

## Space Summary

| Component | Size | Status |
|-----------|------|--------|
| **Production Code** | ~71 KB | ✓ Production-ready, 8 core files |
| **Preprocessing Module** | ~5.6 GB | ✓ With data and cache |
| **Training Data** | ~4.7 GB | ✓ Input data preserved |
| **Trained Model & Scalers** | ~104 KB | ✓ Final results |
| **Archived Files** | ~352 KB | → Moved to `../suc_archive` |
| **Total Production** | ~10.4 GB | ✓ Clean, organized |
| **Archive** | ~352 KB | → Available if needed |

---

## How to Use the Cleaned Repository

### Quick Start (from workspace root)
```bash
cd model/suc
python train_supervised_cluster_routing.py
```

### Full Workflow (preprocessing + training)
```bash
cd model/suc
bash run_suc_workflow.sh
```

### Generate Paper Figures
```bash
cd model/suc
python generate_paper_figures.py
```

### For Preprocessing Only
```bash
cd model/suc/preprocessing
python run_preprocessing_pipeline.py
```

---

## If You Need Archived Files

All development, test, and experimental scripts are preserved in:
```
model/suc_archive/
├── old_training/       (Testing & variant models)
├── old_preprocessing/  (Clustering experiments)
└── logs/              (Historical run logs)
```

Simply move them back to `model/suc/` if needed.

---

## Verification Checklist
- ✅ Core model class preserved
- ✅ Main training script preserved  
- ✅ Feature engineering intact
- ✅ Preprocessing pipeline complete
- ✅ Analysis tools available
- ✅ Paper figure generation working
- ✅ Trained model & scalers in place
- ✅ Training data untouched
- ✅ Old experiments safely archived
- ✅ Documentation complete

**Status: Ready for production use and paper submission** 📋
