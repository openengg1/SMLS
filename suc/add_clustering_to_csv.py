#!/usr/bin/env python3
"""
Preprocessing: Add cluster assignments to training/validation/test CSVs
========================================================================

This script:
1. Fits GMM on 7D physics features from a sample of training data
2. Predicts cluster assignments for all train/val/test samples
3. Adds 'cluster_id' column to each CSV
4. Saves updated CSVs for supervised cluster routing training

This is a ONE-TIME preprocessing step. Once cluster_id is in CSV,
training scripts can load it directly without re-fitting GMM.

Cluster Features (7D physics):
- Urel_mag: Relative velocity magnitude
- We: Weber number (inertia vs surface tension)
- Oh: Ohnesorge number (viscosity vs surface tension)
- Re: Reynolds number (inertia vs viscosity)
- mass_proxy: Droplet mass indicator
- delT_T_boil: Normalized temperature difference
- del_nParticle: Breakup/coalescence indicator
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from feature_engineering import extract_clustering_features

# ============================================================================
# Configuration
# ============================================================================
data_dir = Path('data')  # Local data directory in suc folder
output_dir = data_dir  # Overwrite in place

N_CLUSTERS = 4
SAMPLE_SIZE = 500000  # 500k sample (middle ground: more robust than 50k, faster than full 1.95M)
RANDOM_STATE = 42

print("="*80)
print("PREPROCESSING: Add Cluster Assignments to CSVs (SUC Workflow)")
print("="*80)

# ============================================================================
# STEP 1: Load training data and fit clustering
# ============================================================================
print("\nSTEP 1: Load training data and fit clustering...")
train_file = data_dir / 'train_paired.csv'
print(f"  Loading {train_file.name} ({train_file.stat().st_size / 1e9:.2f} GB)...")
train_data = pd.read_csv(train_file, engine='c')
print(f"  Total samples: {len(train_data):,}")

# Filter to persistent only
train_persistent = train_data[train_data['out_persists'] == 1.0].copy()
print(f"  Persistent samples: {len(train_persistent):,}")

# Extract physics features
print(f"  Computing 7D physics features for clustering...")
X_physics, _, _ = extract_clustering_features(train_persistent, verbose=False)
print(f"  Physics features shape: {X_physics.shape}")

# Sample for GMM fitting
if len(X_physics) > SAMPLE_SIZE:
    print(f"  Sampling {SAMPLE_SIZE:,} samples for GMM fitting...")
    sample_indices = np.random.RandomState(RANDOM_STATE).choice(
        len(X_physics), size=SAMPLE_SIZE, replace=False
    )
    X_sample = X_physics[sample_indices]
else:
    X_sample = X_physics

# Fit GMM
print(f"  Fitting GMM with {N_CLUSTERS} clusters...")
gmm = GaussianMixture(
    n_components=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=5,
    max_iter=200
)
gmm.fit(X_sample)
print(f"  ✓ GMM fitted")

# Predict cluster assignments for all persistent training data
print(f"  Predicting cluster assignments for {len(X_physics):,} persistent samples...")
train_persistent['cluster_id'] = gmm.predict(X_physics)

# Merge back to full training data (non-persistent get cluster_id=-1)
train_data['cluster_id'] = -1
train_data.loc[train_data['out_persists'] == 1.0, 'cluster_id'] = train_persistent['cluster_id'].values

# Print cluster distribution
print(f"  ✓ Cluster distribution (persistent samples only):")
for cid in range(N_CLUSTERS):
    count = (train_persistent['cluster_id'] == cid).sum()
    pct = 100 * count / len(train_persistent)
    print(f"    Cluster {cid}: {count:,} ({pct:.1f}%)")

# ============================================================================
# STEP 2: Process validation data
# ============================================================================
print("\nSTEP 2: Process validation data...")
val_file = data_dir / 'val_paired.csv'
print(f"  Loading {val_file.name}...")
val_data = pd.read_csv(val_file, engine='c')
print(f"  Total samples: {len(val_data):,}")

# Get physics features for validation
val_persistent = val_data[val_data['out_persists'] == 1.0].copy()
print(f"  Persistent samples: {len(val_persistent):,}")

print(f"  Computing 7D physics features for validation...")
X_val, _, _ = extract_clustering_features(val_persistent, verbose=False)
print(f"  Predicting cluster assignments for {len(X_val):,} persistent samples...")
val_persistent['cluster_id'] = gmm.predict(X_val)

# Merge back
val_data['cluster_id'] = -1
val_data.loc[val_data['out_persists'] == 1.0, 'cluster_id'] = val_persistent['cluster_id'].values

print(f"  ✓ Cluster distribution (persistent samples only):")
for cid in range(N_CLUSTERS):
    count = (val_persistent['cluster_id'] == cid).sum()
    pct = 100 * count / len(val_persistent)
    print(f"    Cluster {cid}: {count:,} ({pct:.1f}%)")

# ============================================================================
# STEP 3: Process test data
# ============================================================================
print("\nSTEP 3: Process test data...")
test_file = data_dir / 'test_paired.csv'
print(f"  Loading {test_file.name}...")
test_data = pd.read_csv(test_file, engine='c')
print(f"  Total samples: {len(test_data):,}")

# Get physics features for test
test_persistent = test_data[test_data['out_persists'] == 1.0].copy()
print(f"  Persistent samples: {len(test_persistent):,}")

print(f"  Computing 7D physics features for test...")
X_test, _, _ = extract_clustering_features(test_persistent, verbose=False)
print(f"  Predicting cluster assignments for {len(X_test):,} persistent samples...")
test_persistent['cluster_id'] = gmm.predict(X_test)

# Merge back
test_data['cluster_id'] = -1
test_data.loc[test_data['out_persists'] == 1.0, 'cluster_id'] = test_persistent['cluster_id'].values

print(f"  ✓ Cluster distribution (persistent samples only):")
for cid in range(N_CLUSTERS):
    count = (test_persistent['cluster_id'] == cid).sum()
    pct = 100 * count / len(test_persistent)
    print(f"    Cluster {cid}: {count:,} ({pct:.1f}%)")

# ============================================================================
# STEP 4: Save updated CSVs
# ============================================================================
print("\nSTEP 4: Save updated CSVs with cluster_id column...")

print(f"  Saving train_paired.csv...")
train_data.to_csv(output_dir / 'train_paired.csv', index=False)
print(f"    ✓ {(output_dir / 'train_paired.csv').stat().st_size / 1e9:.2f} GB")

print(f"  Saving val_paired.csv...")
val_data.to_csv(output_dir / 'val_paired.csv', index=False)
print(f"    ✓ {(output_dir / 'val_paired.csv').stat().st_size / 1e9:.2f} GB")

print(f"  Saving test_paired.csv...")
test_data.to_csv(output_dir / 'test_paired.csv', index=False)
print(f"    ✓ {(output_dir / 'test_paired.csv').stat().st_size / 1e9:.2f} GB")

print("\n" + "="*80)
print("✓ PREPROCESSING COMPLETE")
print("="*80)
print("\nCluster assignments added to CSVs:")
print("  • cluster_id: 0-3 for persistent samples")
print("  • cluster_id: -1 for non-persistent samples (not used in SUC)")
print("\nNext step: Run supervised training script")
print("  python3 train_supervised_cluster_routing.py")
print("="*80)
