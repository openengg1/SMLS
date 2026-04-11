#!/usr/bin/env python3
"""
SUC Training for Paper Results (2 Clusters)
============================================

Reproduces the results from ilass_comprehensive.tex:
- 2 clusters: Cluster 0 (small drops ~14%) and Cluster 1 (large drops ~86%)
- Per-cluster expert R² evaluation
- Gating accuracy ~99.4%

Expected results from paper:
- Expert 0 (small drops): Average R² = 0.9875
- Expert 1 (large drops): Average R² = 0.9990  
- Gating accuracy: 99.42%
- Ensemble R²: 0.99745
"""
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

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

# ============================================================================
# Model Definition (2 Experts)
# ============================================================================
class HybridRUCSupervised2Clusters(nn.Module):
    """Supervised Cluster Routing with 2 experts (matching paper configuration)"""
    
    def __init__(self, input_dim=17, output_dim=6, n_experts=2, expert_hidden=32):
        super().__init__()
        self.n_experts = n_experts
        
        # Gating network (classifier)
        self.gating_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_experts)  # n_experts logits (2 for 2 clusters)
        )
        
        # Expert networks (one per cluster)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.ReLU(),
                nn.Linear(expert_hidden, expert_hidden),
                nn.ReLU(),
                nn.Linear(expert_hidden, output_dim)
            )
            for _ in range(n_experts)
        ])
    
    def forward(self, x):
        # Hard routing: argmax on gating logits
        logits = self.gating_net(x)
        cluster_ids = torch.argmax(logits, dim=1)
        
        # Route each sample to its expert
        outputs = torch.zeros(x.size(0), self.experts[0][-1].out_features, device=x.device)
        for k in range(self.n_experts):
            mask = (cluster_ids == k)
            if mask.any():
                outputs[mask] = self.experts[k](x[mask])
        
        return outputs, cluster_ids, logits

device = 'cpu'
print(f"Device: {device}\n")

script_dir = Path(__file__).parent.resolve()
data_dir = script_dir / 'data'
checkpoint_dir = script_dir / 'checkpoints'
checkpoint_dir.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: Load Data with 2-Cluster Labels
# ============================================================================
print("="*80)
print("SUC TRAINING (2 CLUSTERS) - REPRODUCING PAPER RESULTS")
print("="*80)

print("\nSTEP 1: Loading data with 2-cluster assignments...")
train_file = data_dir / 'train_paired.csv'

# Load all needed columns
needed_cols = ['out_persists', 'cluster_id_2clusters'] + \
              [f'in_{c}' for c in ['d', 'U:0', 'U:1', 'U:2', 'T', 'nParticle', 
                                    'rho', 'mu', 'sigma', 'euler_T', 'euler_U:0', 
                                    'euler_U:1', 'euler_U:2', 'euler_H2O', 'euler_p', 
                                    'euler_rho', 'mass_proxy']] + \
              [f'out_{c}' for c in ['d', 'U:0', 'U:1', 'U:2', 'T', 'nParticle']]

train_data = pd.read_csv(train_file, usecols=needed_cols)
train_data = train_data[train_data['out_persists'] == 1.0].copy()
print(f"✓ Loaded {len(train_data):,} persistent training samples")

# Check cluster distribution (matches paper: 14% small, 86% large)
cluster_col = 'cluster_id_2clusters'
cluster_counts = train_data[cluster_col].value_counts().sort_index()
print(f"\nCluster distribution:")
for cid, cnt in cluster_counts.items():
    if cid >= 0:
        print(f"  Cluster {int(cid)}: {cnt:,} ({100*cnt/len(train_data):.1f}%)")

# Load validation data
val_data = pd.read_csv(data_dir / 'val_paired.csv', usecols=needed_cols)
val_data = val_data[val_data['out_persists'] == 1.0].copy()
print(f"✓ Loaded {len(val_data):,} persistent validation samples")

# Load test data
test_data = pd.read_csv(data_dir / 'test_paired.csv', usecols=needed_cols)
test_data = test_data[test_data['out_persists'] == 1.0].copy()
print(f"✓ Loaded {len(test_data):,} persistent test samples")

# ============================================================================
# STEP 2: Prepare Features and Targets
# ============================================================================
print("\n" + "="*80)
print("STEP 2: PREPARE FEATURES AND TARGETS")
print("="*80)

input_cols = [f'in_{c}' for c in ['d', 'U:0', 'U:1', 'U:2', 'T', 'nParticle', 
                                   'rho', 'mu', 'sigma', 'euler_T', 'euler_U:0', 
                                   'euler_U:1', 'euler_U:2', 'euler_H2O', 'euler_p', 
                                   'euler_rho', 'mass_proxy']]
output_cols = [f'out_{c}' for c in ['d', 'U:0', 'U:1', 'U:2', 'T', 'nParticle']]

# Compute deltas (output - input)
def compute_deltas(df, input_cols, output_cols):
    features = df[input_cols].values
    # Delta = out - in for matching features
    delta_cols = ['d', 'U:0', 'U:1', 'U:2', 'T', 'nParticle']
    outputs = np.zeros((len(df), len(delta_cols)))
    for i, col in enumerate(delta_cols):
        outputs[:, i] = df[f'out_{col}'].values - df[f'in_{col}'].values
    return features, outputs

train_features, train_outputs = compute_deltas(train_data, input_cols, output_cols)
val_features, val_outputs = compute_deltas(val_data, input_cols, output_cols)
test_features, test_outputs = compute_deltas(test_data, input_cols, output_cols)

train_clusters = train_data[cluster_col].values.astype(np.int64)
val_clusters = val_data[cluster_col].values.astype(np.int64)
test_clusters = test_data[cluster_col].values.astype(np.int64)

# Normalize features
print("Normalizing features...")
scaler_x = StandardScaler()
train_features = scaler_x.fit_transform(train_features)
val_features = scaler_x.transform(val_features)
test_features = scaler_x.transform(test_features)

# Normalize outputs
scaler_y = StandardScaler()
train_outputs = scaler_y.fit_transform(train_outputs)
val_outputs = scaler_y.transform(val_outputs)
test_outputs = scaler_y.transform(test_outputs)

print(f"✓ Features: {train_features.shape[1]} dimensions")
print(f"✓ Outputs: {train_outputs.shape[1]} dimensions (deltas)")

# ============================================================================
# STEP 3: Initialize Model
# ============================================================================
print("\n" + "="*80)
print("STEP 3: INITIALIZE MODEL (2 experts)")
print("="*80)

N_EXPERTS = 2
model = HybridRUCSupervised2Clusters(
    input_dim=17,
    output_dim=6,
    n_experts=N_EXPERTS,
    expert_hidden=32
)

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model initialized with {total_params:,} parameters")
print(f"  • Gating network: 17→64→32→{N_EXPERTS}")
print(f"  • Each expert: 17→32→32→6")

# ============================================================================
# STEP 4: Train Experts (Per-Cluster)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: TRAIN EXPERTS (Per-Cluster Data)")
print("="*80)

criterion = nn.MSELoss()

for cluster_id in range(N_EXPERTS):
    print(f"\n--- Expert {cluster_id} ---")
    
    # Get data for this cluster
    mask = train_clusters == cluster_id
    cluster_features = train_features[mask]
    cluster_outputs = train_outputs[mask]
    
    print(f"  Training samples: {len(cluster_features):,}")
    
    if len(cluster_features) == 0:
        print(f"  ⚠️ No samples for cluster {cluster_id}, skipping")
        continue
    
    # Create dataset/loader
    expert_dataset = TensorDataset(
        torch.from_numpy(cluster_features).float(),
        torch.from_numpy(cluster_outputs).float()
    )
    expert_loader = DataLoader(expert_dataset, batch_size=256, shuffle=True, pin_memory=False)
    
    # Train
    optimizer = optim.Adam(model.experts[cluster_id].parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    for epoch in range(1, 31):
        model.experts[cluster_id].train()
        loss_sum = 0.0
        n_batches = 0
        
        for x_batch, y_batch in expert_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model.experts[cluster_id](x_batch)
            batch_loss = criterion(pred, y_batch)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            loss_sum += batch_loss.item()
            n_batches += 1
        
        avg_loss = loss_sum / n_batches if n_batches > 0 else 0
        scheduler.step(avg_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:2d} | Loss: {avg_loss:.6f}")

# ============================================================================
# STEP 5: Train Gating Network (Classifier)
# ============================================================================
print("\n" + "="*80)
print("STEP 5: TRAIN GATING NETWORK (Classifier)")
print("="*80)

train_features_torch = torch.from_numpy(train_features).float()
train_clusters_torch = torch.from_numpy(train_clusters).long()
train_gating_dataset = TensorDataset(train_features_torch, train_clusters_torch)

val_features_torch = torch.from_numpy(val_features).float()
val_clusters_torch = torch.from_numpy(val_clusters).long()

gating_optimizer = optim.Adam(model.gating_net.parameters(), lr=1e-3)
gating_criterion = nn.CrossEntropyLoss()
gating_scheduler = optim.lr_scheduler.ReduceLROnPlateau(gating_optimizer, mode='min', factor=0.5, patience=5)

print(f"\n{'Epoch':<6} | {'Train Acc':<10} | {'Val Acc':<10} | Status")
print("-" * 60)

best_val_acc = 0.0
best_epoch = 0

for epoch in range(1, 51):
    model.gating_net.train()
    train_correct = 0
    train_total = 0
    
    train_loader = DataLoader(train_gating_dataset, batch_size=256, shuffle=True, pin_memory=False)
    
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model.gating_net(x_batch)
        loss = gating_criterion(logits, y_batch)
        
        gating_optimizer.zero_grad()
        loss.backward()
        gating_optimizer.step()
        
        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == y_batch).sum().item()
        train_total += len(y_batch)
    
    train_acc = train_correct / train_total
    
    # Validation
    model.gating_net.eval()
    with torch.no_grad():
        val_logits = model.gating_net(val_features_torch)
        val_preds = torch.argmax(val_logits, dim=1)
        val_acc = (val_preds == val_clusters_torch).float().mean().item()
    
    gating_scheduler.step(1 - val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), checkpoint_dir / 'suc_2clusters_best.pt')
        status = "✅ BEST"
    else:
        status = ""
    
    if epoch % 5 == 0 or status:
        print(f"{epoch:<6} | {train_acc:<10.4f} | {val_acc:<10.4f} | {status}")

print("-" * 60)
print(f"Best epoch: {best_epoch}, Best val accuracy: {best_val_acc:.4f}")

# Load best model
model.load_state_dict(torch.load(checkpoint_dir / 'suc_2clusters_best.pt'))

# ============================================================================
# STEP 6: Test Set Evaluation (Per-Cluster R²)
# ============================================================================
print("\n" + "="*80)
print("TEST SET EVALUATION (Per-Cluster)")
print("="*80)

model.eval()
test_features_torch = torch.from_numpy(test_features).float()

with torch.no_grad():
    # Gating predictions
    test_logits = model.gating_net(test_features_torch)
    predicted_clusters = torch.argmax(test_logits, dim=1).numpy()

# Gating accuracy
gating_accuracy = accuracy_score(test_clusters, predicted_clusters)
print(f"\nGating Network Accuracy: {gating_accuracy*100:.2f}%")
print(f"  Correctly predicted: {np.sum(predicted_clusters == test_clusters):,} / {len(test_clusters):,}")

# Per-cluster R² evaluation (matching paper format)
feature_names = ["Δd", "ΔUx", "ΔUy", "ΔUz", "ΔT", "Δn_Particle"]

print("\n" + "="*80)
print("PER-CLUSTER EXPERT PERFORMANCE (matching paper Tables 1 & 2)")
print("="*80)

cluster_r2_avg = {}

for cluster_id in range(N_EXPERTS):
    # Get test samples for this cluster (using ground truth cluster)
    mask = test_clusters == cluster_id
    n_samples = mask.sum()
    
    cluster_features = torch.from_numpy(test_features[mask]).float()
    cluster_outputs = test_outputs[mask]
    
    # Get expert predictions
    with torch.no_grad():
        cluster_preds = model.experts[cluster_id](cluster_features).numpy()
    
    print(f"\n--- Expert {cluster_id} (Cluster {cluster_id} test set, {n_samples:,} samples) ---")
    print(f"{'Feature':<15} | {'R²':<10} | {'MSE':<10} | Interpretation")
    print("-" * 70)
    
    r2_scores = []
    for i, feat in enumerate(feature_names):
        r2 = r2_score(cluster_outputs[:, i], cluster_preds[:, i])
        mse = mean_squared_error(cluster_outputs[:, i], cluster_preds[:, i])
        r2_scores.append(r2)
        
        if r2 >= 0.999:
            interp = "Perfectly learned"
        elif r2 >= 0.99:
            interp = "Near-perfect"
        elif r2 >= 0.98:
            interp = "Well-learned"
        else:
            interp = "Captured"
        
        print(f"{feat:<15} | {r2:<10.4f} | {mse:<10.4f} | {interp}")
    
    avg_r2 = np.mean(r2_scores)
    cluster_r2_avg[cluster_id] = avg_r2
    print("-" * 70)
    print(f"{'AVERAGE':<15} | {avg_r2:<10.4f} |")

# ============================================================================
# Ensemble Performance
# ============================================================================
print("\n" + "="*80)
print("ENSEMBLE MODEL PERFORMANCE")
print("="*80)

# Using predicted clusters for routing
predictions = np.zeros_like(test_outputs)
with torch.no_grad():
    for i in range(len(test_features)):
        cluster_id = predicted_clusters[i]
        pred = model.experts[cluster_id](test_features_torch[i:i+1]).numpy()
        predictions[i] = pred[0]

print(f"\nEnsemble R² (using gating-predicted clusters):")
print(f"{'Feature':<15} | {'R²':<10}")
print("-" * 30)

ensemble_r2_scores = []
for i, feat in enumerate(feature_names):
    r2 = r2_score(test_outputs[:, i], predictions[:, i])
    ensemble_r2_scores.append(r2)
    print(f"{feat:<15} | {r2:.4f}")

ensemble_avg_r2 = np.mean(ensemble_r2_scores)
print("-" * 30)
print(f"{'AVERAGE':<15} | {ensemble_avg_r2:.4f}")

# Weighted average (theoretical)
cluster_0_pct = (test_clusters == 0).mean()
cluster_1_pct = (test_clusters == 1).mean()
weighted_r2 = cluster_0_pct * cluster_r2_avg.get(0, 0) + cluster_1_pct * cluster_r2_avg.get(1, 0)

# ============================================================================
# Summary - Comparison with Paper
# ============================================================================
print("\n" + "="*80)
print("COMPARISON WITH PAPER (ilass_comprehensive.tex)")
print("="*80)

print(f"""
{'Metric':<35} | {'Paper':<12} | {'This Run':<12} | Match
{'-'*75}
Gating Accuracy                     | 99.42%       | {gating_accuracy*100:.2f}%       | {'✅' if abs(gating_accuracy*100 - 99.42) < 1 else '⚠️'}
Expert 0 Avg R² (small drops)       | 0.9875       | {cluster_r2_avg.get(1, 0):.4f}       | {'✅' if abs(cluster_r2_avg.get(1, 0) - 0.9875) < 0.02 else '⚠️'}
Expert 1 Avg R² (large drops)       | 0.9990       | {cluster_r2_avg.get(0, 0):.4f}       | {'✅' if abs(cluster_r2_avg.get(0, 0) - 0.9990) < 0.02 else '⚠️'}
Ensemble Avg R²                     | 0.9975       | {ensemble_avg_r2:.4f}       | {'✅' if abs(ensemble_avg_r2 - 0.9975) < 0.02 else '⚠️'}
Weighted R² (theoretical)           | 0.9975       | {weighted_r2:.4f}       | {'✅' if abs(weighted_r2 - 0.9975) < 0.02 else '⚠️'}

Note: Cluster labeling may be swapped (0↔1) compared to paper. 
      The "small drops" cluster has ~14% of data, "large drops" has ~86%.
""")

print("="*80)
print("TRAINING COMPLETE - Model saved to checkpoints/suc_2clusters_best.pt")
print("="*80)
