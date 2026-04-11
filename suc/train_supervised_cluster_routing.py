#!/usr/bin/env python3
"""
Supervised Cluster Routing Training
====================================

Training procedure:
1. Load pre-computed cluster labels from CSV (from model/suc/add_clustering_to_csv.py)
2. Separate training data by cluster assignments
3. Train each expert ONLY on its cluster data
4. Train gating network as CLASSIFIER on cluster labels (CrossEntropyLoss)
5. Evaluate: gating predicts cluster → use that expert's output

Gating Network Training:
- Input: 24D features
- Output: 4 classification logits
- Loss: CrossEntropyLoss(logits, cluster_id)  ← Cluster label from CSV
- Learns to map features directly to correct cluster assignment
"""
import sys
from pathlib import Path

# Add parent directories to path to allow imports
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd().parent))
sys.path.insert(0, str(Path.cwd().parent.parent))

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

from hybrid_ruc_supervised import HybridRUCSupervised

device = 'cpu'
print(f"Device: {device}\n")

data_dir = Path('data')  # Local data directory in suc folder

# ============================================================================
# STEP 1: Load Pre-Computed Cluster Labels from CSV
# ============================================================================
print("="*80)
print("STEP 1: LOAD PRE-COMPUTED CLUSTER LABELS")
print("="*80)

# Check if cluster_id column exists
print("Loading training data (persistent only)...")
train_data = pd.read_csv(data_dir / 'train_paired.csv',
                         usecols=['out_persists', 'cluster_id'] + 
                                 [col for col in pd.read_csv(data_dir / 'train_paired.csv', nrows=1).columns 
                                  if col.startswith('inp_') or col.startswith('out_')])

train_data_full = train_data[train_data['out_persists'] == 1.0].copy()
del train_data

if 'cluster_id' not in train_data_full.columns:
    raise ValueError(
        "❌ 'cluster_id' column not found in CSV!\n"
        "   Please run: python3 model/suc/add_clustering_to_csv.py\n"
        "   (This is a one-time preprocessing step)"
    )

print(f"✓ Loaded {len(train_data_full):,} persistent training samples")
print(f"✓ Cluster column found: cluster_id")

# Verify cluster distribution
unique_clusters = train_data_full['cluster_id'].unique()
print(f"✓ Cluster distribution:")
for cid in sorted(unique_clusters):
    if cid >= 0:  # Only show valid clusters
        count = (train_data_full['cluster_id'] == cid).sum()
        pct = 100 * count / len(train_data_full)
        print(f"  Cluster {int(cid)}: {count:,} ({pct:.1f}%)")

# Initialize model (no clustering fitting needed)
print("\nInitializing model...")
model = HybridRUCSupervised(
    gating_input_dim=17,
    clustering_input_dim=7,
    output_dim=6,
    n_clusters=4,
    expert_hidden_dim=8,
    device=device
)
print(f"✓ Model parameters: {model.count_parameters():,}")
print(f"✓ Architecture: 17D inputs → gating (predict cluster) → expert (predict outputs)")

# ============================================================================
# STEP 2: Load Full Training Data and Separate by Cluster
# ============================================================================
print("\n" + "="*80)
print("STEP 2: PREPARE TRAINING DATA")
print("="*80)

print("Loading training data and cluster labels...")
train_features_list = []
train_outputs_list = []
train_clusters_list = []

for chunk in pd.read_csv(data_dir / 'train_paired.csv', chunksize=2000):
    chunk = chunk[chunk['out_persists'] == 1].copy()
    if len(chunk) == 0:
        continue
    
    # Extract cluster labels from CSV (pre-computed)
    chunk_clusters = chunk['cluster_id'].values
    
    # Extract 17D raw features (in_* columns)
    inp_cols = [col for col in chunk.columns if col.startswith('in_')]
    train_features = chunk[inp_cols].values.astype(np.float32)
    
    # Extract output targets (out_* columns, excluding out_persists)
    train_output_cols = [col for col in chunk.columns if col.startswith('out_') and col != 'out_persists']
    train_outputs = chunk[train_output_cols].values.astype(np.float32)
    
    train_features_list.append(train_features)
    train_outputs_list.append(train_outputs)
    train_clusters_list.append(chunk_clusters)

# Combine all chunks
train_features_all = np.vstack(train_features_list)
train_outputs_all = np.vstack(train_outputs_list)
train_clusters_all = np.hstack(train_clusters_list)

print(f"✓ Training features shape: {train_features_all.shape}")
print(f"✓ Training outputs shape: {train_outputs_all.shape}")
print(f"✓ Cluster labels shape: {train_clusters_all.shape}")

# Separate data by cluster
print("\nSeparating training data by cluster...")
cluster_indices = {i: np.where(train_clusters_all == i)[0] for i in range(model.n_clusters)}
print("Cluster distribution:")
for cluster_id in range(model.n_clusters):
    count = len(cluster_indices[cluster_id])
    pct = 100 * count / len(train_clusters_all)
    print(f"  Cluster {cluster_id}: {count:,} samples ({pct:.1f}%)")

# Load val/test data with cluster labels
print("\nLoading validation data...")
val_data = pd.read_csv(data_dir / 'val_paired.csv')
val_data = val_data[val_data['out_persists'] == 1].copy()
val_clusters = val_data['cluster_id'].values

# Extract 17D features and outputs
inp_cols = [col for col in val_data.columns if col.startswith('in_')]
val_features = val_data[inp_cols].values.astype(np.float32)
val_output_cols = [col for col in val_data.columns if col.startswith('out_') and col != 'out_persists']
val_outputs = val_data[val_output_cols].values.astype(np.float32)

print("Loading test data...")
test_data = pd.read_csv(data_dir / 'test_paired.csv')
test_data = test_data[test_data['out_persists'] == 1].copy()
test_clusters = test_data['cluster_id'].values

# Extract 17D features and outputs
test_features = test_data[inp_cols].values.astype(np.float32)
test_output_cols = [col for col in test_data.columns if col.startswith('out_') and col != 'out_persists']
test_outputs = test_data[test_output_cols].values.astype(np.float32)

val_features_torch = torch.from_numpy(val_features).float()
val_outputs_torch = torch.from_numpy(val_outputs).float()
val_clusters_torch = torch.from_numpy(val_clusters).long()
val_dataset = TensorDataset(val_features_torch, val_outputs_torch, val_clusters_torch)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=False)

print(f"✓ Val samples: {len(val_data)}")
print(f"✓ Test samples: {len(test_data)}")
print(f"✓ Input dimension: 17D (raw features)")

# ============================================================================
# STEP 3: Train Experts (Each Expert on Its Cluster Data Only)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: TRAINING EXPERTS (Per-cluster supervised)")
print("="*80)

criterion = nn.MSELoss()

for cluster_id in range(model.n_clusters):
    print(f"\nTraining Expert {cluster_id}...")
    indices = cluster_indices[cluster_id]
    
    if len(indices) == 0:
        print(f"  WARNING: No samples for cluster {cluster_id}")
        continue
    
    expert_features = train_features_all[indices]
    expert_outputs = train_outputs_all[indices]
    
    expert_features_torch = torch.from_numpy(expert_features).float()
    expert_outputs_torch = torch.from_numpy(expert_outputs).float()
    expert_dataset = TensorDataset(expert_features_torch, expert_outputs_torch)
    expert_loader = DataLoader(expert_dataset, batch_size=128, shuffle=True, pin_memory=False)
    
    # Train this expert
    optimizer = optim.Adam(model.experts[cluster_id].parameters(), lr=1e-3)
    
    for epoch in range(1, 21):
        model.experts[cluster_id].train()
        loss = 0.0
        n_batches = 0
        
        for x_batch, y_batch in expert_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model.experts[cluster_id](x_batch)
            batch_loss = criterion(pred, y_batch)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            loss += batch_loss.item()
            n_batches += 1
        
        if n_batches > 0:
            loss /= n_batches
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d} | Loss: {loss:.4f}")

# ============================================================================
# STEP 4: Train Gating Network (Classifier on Cluster Labels)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: TRAINING GATING NETWORK (Cluster classifier)")
print("="*80)

train_features_torch = torch.from_numpy(train_features_all).float()
train_clusters_torch = torch.from_numpy(train_clusters_all).long()
train_gating_dataset = TensorDataset(train_features_torch, train_clusters_torch)

gating_optimizer = optim.Adam(model.gating_net.parameters(), lr=1e-3)
gating_criterion = nn.CrossEntropyLoss()
gating_scheduler = optim.lr_scheduler.ReduceLROnPlateau(gating_optimizer, mode='min', factor=0.5, patience=3, verbose=False)

print(f"\n{'Epoch':<6} | {'Train Acc':<10} | {'Val Acc':<10} | Status")
print("-" * 60)

best_val_acc = 0.0
best_epoch = 0
epochs_without_improvement = 0
early_stopping_patience = 15

for epoch in range(1, 51):
    model.gating_net.train()
    train_acc = 0.0
    n_batches = 0
    
    train_loader = DataLoader(train_gating_dataset, batch_size=128, shuffle=True, pin_memory=False)
    
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model.gating_net(x_batch)
        loss = gating_criterion(logits, y_batch)
        
        gating_optimizer.zero_grad()
        loss.backward()
        gating_optimizer.step()
        
        preds = torch.argmax(logits, dim=1)
        train_acc += (preds == y_batch).float().mean().item()
        n_batches += 1
    
    if n_batches > 0:
        train_acc /= n_batches
    
    # Validation
    model.gating_net.eval()
    val_acc = 0.0
    with torch.no_grad():
        for x_batch, _, y_batch_clusters in val_loader:
            x_batch, y_batch_clusters = x_batch.to(device), y_batch_clusters.to(device)
            logits = model.gating_net(x_batch)
            preds = torch.argmax(logits, dim=1)
            val_acc += (preds == y_batch_clusters).float().mean().item()
    
    val_acc /= len(val_loader)
    gating_scheduler.step(1 - val_acc)  # Minimize error
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'checkpoints/suc_best_model.pt')
        print(f"        Saved checkpoint: checkpoints/suc_best_model.pt")
        status = "✅ BEST"
    else:
        epochs_without_improvement += 1
        status = ""
    
    print(f"{epoch:<6} | {train_acc:<10.4f} | {val_acc:<10.4f} | {status}")
    
    if epochs_without_improvement >= early_stopping_patience:
        print("-" * 60)
        print(f"⚠️  Early stopping at epoch {epoch}")
        break

# ============================================================================
# STEP 5: Evaluate on Test Set
# ============================================================================
print("\n" + "="*80)
print("TEST SET EVALUATION")
print("="*80)

model.eval()
test_features_torch = torch.from_numpy(test_features).float()
test_outputs_torch = torch.from_numpy(test_outputs).float()
test_clusters_torch = torch.from_numpy(test_clusters).long()

with torch.no_grad():
    # Gating predicts cluster
    test_logits = model.gating_net(test_features_torch)
    predicted_clusters = torch.argmax(test_logits, dim=1).numpy()
    
    # Use predicted cluster expert for each sample
    predictions = np.zeros_like(test_outputs)
    for i in range(len(test_features_torch)):
        cluster_id = predicted_clusters[i]
        expert_pred = model.experts[cluster_id](test_features_torch[i:i+1]).numpy()
        predictions[i] = expert_pred[0]

# Evaluate gating accuracy
gating_accuracy = accuracy_score(test_clusters, predicted_clusters)
print(f"\nGating Network Accuracy: {gating_accuracy:.4f}")
print(f"  Correctly predicted clusters: {np.sum(predicted_clusters == test_clusters):,} / {len(test_clusters):,}")

# Per-feature R²
features = ["Δd", "ΔU₀", "ΔU₁", "ΔU₂", "ΔT", "ΔnParticle"]
print(f"\nPer-Feature R² (n={len(test_data):,} samples):")
print("-" * 90)
print(f"{'Feature':<15} | {'R²':<10} | {'RMSE':<10} | Status")
print("-" * 90)

r2_scores = []
for i, feature in enumerate(features):
    r2 = r2_score(test_outputs[:, i], predictions[:, i])
    rmse = np.sqrt(mean_squared_error(test_outputs[:, i], predictions[:, i]))
    r2_scores.append(r2)
    status = "✅ Excellent" if r2 > 0.9 else "✔️ Good" if r2 > 0.8 else "⚠️ Fair" if r2 > 0.6 else "❌ Poor"
    print(f"{feature:<15} | {r2:<10.4f} | {rmse:<10.4f} | {status}")

overall_r2 = np.mean(r2_scores)
print("-" * 90)
print(f"{'OVERALL':<15} | {overall_r2:<10.4f} |")
print("=" * 90)

print(f"\nTraining Summary:")
print(f"  Best gating epoch: {best_epoch}")
print(f"  Best gating val accuracy: {best_val_acc:.4f}")
print(f"  Test gating accuracy: {gating_accuracy:.4f}")
print(f"  Test R² (mean): {overall_r2:.4f}")
print(f"\nComparison:")
print(f"  MLP baseline (all data):        R² = 0.9367")
print(f"  Physics-Informed v1 (all data): R² = 0.6248")
print(f"  SelfOrgMOE (current):           R² = TBD")
print(f"  Supervised (now):               R² = {overall_r2:.4f}")
print("=" * 90)
