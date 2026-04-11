#!/usr/bin/env python3
"""
Generate all paper figures: PDF plots of physics features by cluster, 
distribution overlays, and analysis plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

# Use relative paths for self-containment
script_dir = Path(__file__).parent.resolve()
data_dir = script_dir / 'preprocessing' / 'data'
train_df = pd.read_csv(data_dir / 'train_paired_gmm.csv')

# Extract features
features_to_plot = {
    'We': 'Weber Number',
    'Re': 'Reynolds Number', 
    'delta_d': 'Δd (Diameter Change)',
    'delta_T': 'ΔT (Temperature Change)',
    'delta_nParticle': 'Δn_particle (Count Change)',
    'delta_Urel_mag': 'Δ|U_rel| (Velocity Change)'
}

# Create output directory (self-contained in suc/figures/)
fig_dir = script_dir / 'figures'
fig_dir.mkdir(exist_ok=True)

# Color map for clusters
colors = {0: '#1f77b4', 1: '#ff7f0e'}  # Blue for cluster 0, Orange for cluster 1

# ============================================================================
# 1. Individual PDF plots for each feature (overlaid by cluster)
# ============================================================================
print("Creating individual feature PDF plots...")

for feature, title in features_to_plot.items():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for cluster_id in [0, 1]:
        data_c = train_df[train_df['cluster_id_physics'] == cluster_id][feature]
        data_c = data_c[np.isfinite(data_c)]  # Remove NaN/inf
        
        # Handle negative values for log scales
        if feature in ['We', 'Re']:
            data_c = np.abs(data_c)
            data_c = data_c[data_c > 0]
            ax.hist(data_c, bins=100, alpha=0.6, label=f'Cluster {cluster_id}', 
                   color=colors[cluster_id], density=True, log=False)
            ax.set_xscale('log')
        else:
            ax.hist(data_c, bins=100, alpha=0.6, label=f'Cluster {cluster_id}',
                   color=colors[cluster_id], density=True)
    
    ax.set_xlabel(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax.set_title(f'Distribution: {title} by Cluster', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fname = f"{feature}_pdf_by_cluster.png"
    plt.savefig(fig_dir / fname, dpi=300, bbox_inches='tight')
    print(f"  ✓ {fname}")
    plt.close()

# ============================================================================
# 2. Composite 2x3 figure with all PDFs
# ============================================================================
print("\nCreating composite 2x3 PDF figure...")

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for idx, (feature, title) in enumerate(features_to_plot.items()):
    ax = axes[idx]
    
    for cluster_id in [0, 1]:
        data_c = train_df[train_df['cluster_id_physics'] == cluster_id][feature]
        data_c = data_c[np.isfinite(data_c)]
        
        if feature in ['We', 'Re']:
            data_c = np.abs(data_c)
            data_c = data_c[data_c > 0]
            ax.hist(data_c, bins=80, alpha=0.6, label=f'Cluster {cluster_id}',
                   color=colors[cluster_id], density=True, log=False)
            ax.set_xscale('log')
        else:
            ax.hist(data_c, bins=80, alpha=0.6, label=f'Cluster {cluster_id}',
                   color=colors[cluster_id], density=True)
    
    ax.set_xlabel(title, fontsize=10, fontweight='bold')
    ax.set_ylabel('PDF', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / 'all_features_pdf_composite.png', dpi=300, bbox_inches='tight')
print("  ✓ all_features_pdf_composite.png")
plt.close()

# ============================================================================
# 3. Cluster statistics table (text-based)
# ============================================================================
print("\nGenerating cluster statistics...")

stats = {}
for cluster_id in [0, 1]:
    cluster_data = train_df[train_df['cluster_id_physics'] == cluster_id]
    stats[cluster_id] = {
        'samples': len(cluster_data),
        'pct': 100 * len(cluster_data) / len(train_df),
        'We_mean': cluster_data['We'].mean(),
        'Re_mean': cluster_data['Re'].mean(),
        'delta_d_mean': cluster_data['delta_d'].mean(),
        'delta_T_mean': cluster_data['delta_T'].mean(),
        'delta_nParticle_mean': cluster_data['delta_nParticle'].mean(),
        'delta_Urel_mag_mean': cluster_data['delta_Urel_mag'].mean(),
    }

with open(fig_dir / 'cluster_statistics.txt', 'w') as f:
    f.write("=== CLUSTER STATISTICS ===\n\n")
    for cluster_id in [0, 1]:
        s = stats[cluster_id]
        f.write(f"Cluster {cluster_id}:\n")
        f.write(f"  Samples: {s['samples']:,} ({s['pct']:.1f}%)\n")
        f.write(f"  We (mean): {s['We_mean']:.2f}\n")
        f.write(f"  Re (mean): {s['Re_mean']:.2f}\n")
        f.write(f"  Δd (mean): {s['delta_d_mean']:.6f}\n")
        f.write(f"  ΔT (mean): {s['delta_T_mean']:.2f}\n")
        f.write(f"  Δn_particle (mean): {s['delta_nParticle_mean']:.2f}\n")
        f.write(f"  Δ|U_rel| (mean): {s['delta_Urel_mag_mean']:.4f}\n\n")

print("  ✓ cluster_statistics.txt")

# ============================================================================
# 4. Scatter plot: We vs Re, colored by cluster
# ============================================================================
print("\nCreating We vs Re scatter plot...")

fig, ax = plt.subplots(figsize=(10, 7))

for cluster_id in [0, 1]:
    mask = train_df['cluster_id_physics'] == cluster_id
    data = train_df[mask]
    
    We = np.abs(data['We'])
    Re = np.abs(data['Re'])
    
    We = We[We > 0]
    Re = Re[Re > 0]
    
    ax.scatter(We, Re, alpha=0.3, s=5, label=f'Cluster {cluster_id}',
              color=colors[cluster_id], rasterized=True)

ax.set_xlabel('Weber Number (We)', fontsize=12, fontweight='bold')
ax.set_ylabel('Reynolds Number (Re)', fontsize=12, fontweight='bold')
ax.set_title('Spray Regimes: Weber vs Reynolds Number', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, which='both')

plt.savefig(fig_dir / 'we_vs_re_clustering.png', dpi=300, bbox_inches='tight')
print("  ✓ we_vs_re_clustering.png")
plt.close()

print("\n✓ All paper figures generated successfully!")
print(f"  Saved to: {fig_dir}")
