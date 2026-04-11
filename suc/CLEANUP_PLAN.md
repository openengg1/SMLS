# SUC Folder Cleanup Plan

## Files to KEEP (Production & Results)

### Core Model & Training
- ✅ `hybrid_ruc_supervised.py` - Model definition
- ✅ `feature_engineering.py` - Feature extraction  
- ✅ `train_supervised_cluster_routing.py` - Main training script
- ✅ `add_clustering_to_csv.py` - Preprocessing helper
- ✅ `generate_paper_figures.py` - Figure generation
- ✅ `run_suc_workflow.sh` - Full workflow script
- ✅ `README.md` - Documentation

### Preprocessing Subdirectory (`preprocessing/`)
- ✅ `create_physics_clusters_gmm_subsampled.py` - GMM clustering (final version)
- ✅ `run_preprocessing_pipeline.py` - Preprocessing runner
- ✅ `__init__.py` - Package initialization
- ✅ `README.md` - Documentation
- ✅ Documentation files:
  - `PHASES_DOCUMENTATION.md`
  - `PREPROCESSING_GUIDE.md`
  - `PREPROCESSING_COMPLETE.md`

### Analysis Subdirectory (`analysis/`)
- ✅ `plot_cluster_distributions.py` - Analysis helper

### Checkpoints (`checkpoints/`)
- ✅ `suc_best_model.pt` - Final trained model

### Results (`results/`)
- ✅ `suc_best_model.pt` - Final model weights
- ✅ `gmm_*_clusters.pkl` - GMM models
- ✅ `scaler_*.pkl` - Feature scalers

### Logs (Keep minimal)
- ✅ `training_final.log` - Final successful run
- ✅ `preprocessing_full_run.log` - Final preprocessing log

---

## Files to REMOVE (Development/Testing)

### Old Training Variants (remove all)
- ❌ `train_quick.py` - Old quick test
- ❌ `train_suc_simple.py` - Old simple version
- ❌ `train_suc_v2.py` - Version 2 (replaced)
- ❌ `train_suc_sequential.py` - Sequential variant
- ❌ `train_suc_optimized.py` - Optimization attempt
- ❌ `train_full_data.py` - Full data variant
- ❌ `train_working.py` - Working version (replaced)
- ❌ `train_expert0_optimized.py` - Expert 0 only
- ❌ `train_suc_with_gmm_clustering.py` - GMM variant

### Testing Scripts
- ❌ `test_data_load.py` - Data loading test
- ❌ `test_simple.py` - Simple test

### Experiment Scripts
- ❌ `cluster_8_classification.py` - 8-cluster experiment
- ❌ `run_gmm_clustering.py` - Old GMM runner
- ❌ `run_gmm_5clusters_1m.py` - 5-cluster experiment
- ❌ `regenerate_figures_large_fonts.py` - Figure regeneration test

### Old Preprocessing (`preprocessing/`)
- ❌ `create_physics_clusters.py` - Old GMM version
- ❌ `create_physics_clusters_kmeans.py` - KMeans variant
- ❌ `diagnose_eulerian_matching.py` - Debugging script
- ❌ `find_optimal_cluster_number.py` - Hyperparameter tuning
- ❌ `find_optimal_clusters.py` - Cluster tuning variant
- ❌ `prepare_gnn_data_clean.py` - GNN data prep (not used)
- ❌ `raw_to_paired.py` - Old data conversion

### Extra Log Files
- ❌ `clustering_8clusters.log`
- ❌ `clustering_preprocessing.log`
- ❌ `training_1774212111.log`
- ❌ `training_fixed.log`
- ❌ `training_heredoc.log`
- ❌ `training_retry.log`
- ❌ `training_run.log`
- ❌ `training_shell.log`

### Cleanup for Directory Structures
If empty after file removal:
- `preprocessing/logs/` - Can be removed if empty
- `preprocessing/data/` - (check if contains only test data)
- `analysis/` - Keep with `plot_cluster_distributions.py`
- `data/` - Keep (input data)
- `results/` - Keep (output models/scalers)
- `checkpoints/` - Keep (trained model)
- `logs/` - Can be archived or removed

---

## Summary of Files to Keep
**Total: ~13 essential files + subdirectories**
- Model code: 4 files
- Data preparation: 1 file
- Figure generation: 1 file  
- Workflow: 1 file
- Preprocessing module: 5 files + __pycache__
- Analysis module: 1 file
- Results/Checkpoints: Directories with trained models

**Estimated size reduction: 60-70% of current size**

---

## How to Execute Cleanup
Option 1: Manual removal via file explorer
- Navigate to `/home/rmishra/projects/stochasticMLSpray/model/suc/`
- Delete files listed above in "Files to REMOVE"

Option 2: Command execution (once approved)
Run cleanup commands to create an archive:
```bash
cd /home/rmishra/projects/stochasticMLSpray/model/suc
mkdir -p ../suc_archive/old_training
mkdir -p ../suc_archive/old_preprocessing
mkdir -p ../suc_archive/logs

# Move old training files
mv train_*.py ../suc_archive/old_training/ 2>/dev/null
mv test_*.py ../suc_archive/old_training/ 2>/dev/null

# Move old preprocessing files  
mv preprocessing/create_physics_clusters*.py ../suc_archive/old_preprocessing/ 2>/dev/null
mv preprocessing/find_optimal*.py ../suc_archive/old_preprocessing/ 2>/dev/null

# Move old logs
mv *.log ../suc_archive/logs/ 2>/dev/null
```
