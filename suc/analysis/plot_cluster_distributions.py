#!/usr/bin/env python3
"""
Generate cluster distribution PDF plots for paper.
Shows how the 2 clusters represent different physical regimes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Configure style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Use relative paths for self-containment
script_dir = Path(__file__).parent.resolve()
data_dir = script_dir.parent / 'preprocessing' / 'data'
train_df = pd.read_csv(data_dir / 'train_paired_gmm.csv')

# Extract features for plotting
features = {
    'We': 'Weber Number',
    'Re': 'Reynolds Number',
    'delta_d': 'Diameter Change (mm)',
    'delta_T': 'Temperature Change (K)',
    'delta_nParticle': 'Particle Count Change',
    'delta_Urel_mag': 'Relative Velocity Change (m/s)'
}

clusters = train_df['cluster_id_physics'].unique()
colors = {0: '#1f77b4', 1: '#ff7f0e'}  # Blue and Orange
cluster_labels = {0: 'Cluster 0 (13.7%)', 1: 'Cluster 1 (86.3%)'}

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Probability Density Functions: Two Distinct Physical Regimes', 
             fontsize=16, fontweight='bold', y=0.995)

axes = axes.flatten()

for idx, (feature, title) in enumerate(features.items()):
    ax = axes[idx]
    
    # Plot PDF for each cluster
    for cluster_id in sorted(clusters):
        cluster_data = train_df[train_df['cluster_id_physics'] == cluster_id][feature]
        
        # Remove infinities and extreme outliers for better visualization
        cluster_data = cluster_data[np.isfinite(cluster_data)]
        q1, q99 = cluster_data.quantile([0.01, 0.99])
        cluster_data = cluster_data[(cluster_data >= q1) & (cluster_data <= q99)]
        
        ax.hist(cluster_data, bins=50, density=True, alpha=0.6, 
               color=colors[cluster_id], label=cluster_labels[cluster_id], 
               edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_dir = script_dir.parent / 'figures'
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / 'cluster_distributions_pdf.png', 
            dpi=300, bbox_inches='tight')
print("✓ Cluster distribution PDF plot saved: figures/cluster_distributions_pdf.png")

# Generate summary statistics
print("\n" + "="*80)
print("CLUSTER STATISTICS")
print("="*80)

for cluster_id in sorted(clusters):
    cluster_data = train_df[train_df['cluster_id_physics'] == cluster_id]
    print(f"\n{cluster_labels[cluster_id]}:")
    print(f"  Samples: {len(cluster_data):,}")
    
    for feature, title in features.items():
        vals = cluster_data[feature][np.isfinite(cluster_data[feature])]
        print(f"  {feature:20s}: mean={vals.mean():10.4f}, std={vals.std():10.4f}, " +
              f"min={vals.min():10.4f}, max={vals.max():10.4f}")
