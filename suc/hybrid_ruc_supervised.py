"""
Supervised Cluster Routing (SUC) - Physics-Informed Mixture of Experts
=======================================================================

Architecture combining supervised cluster routing with physics-informed experts:
1. 11D physics-based clustering (unsupervised GMM on aerodynamic, kinematic & thermal features)
2. Per-cluster expert networks trained SEPARATELY on cluster-specific data
3. Gating network trained as CLASSIFIER on explicit cluster labels (CrossEntropyLoss)
4. Hard cluster selection: Each sample routed to ONE dominant cluster

Key differences from SelfOrgMOE:
- Experts: Trained per-cluster on separate subsets (not mixed with soft weights)
- Gating: Trained as classifier with CrossEntropyLoss (not regression loss)
- Routing: Hard selection of best cluster (not soft weighted combination)
- Ground truth: Explicit cluster labels from GMM (not implicit from regression targets)

Physics features for clustering (11D):
- We: Weber number (inertia vs surface tension)
- Oh: Ohnesorge number (viscosity vs surface tension)
- Re: Reynolds number (inertia vs viscosity)
- Urel_mag: Relative velocity magnitude
- mass_proxy: Droplet mass indicator
- delT_T_boil: Normalized temperature difference
- del_nParticle: Breakup/coalescence indicator
- delUx: Change in x-velocity component
- delUy: Change in y-velocity component
- delUz: Change in z-velocity component
- deld: Change in diameter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import numpy as np
from typing import Tuple, List, Dict, Optional


class ExpertNetwork(nn.Module):
    """Expert network for cluster-specific regression.
    
    Trained on samples belonging to a specific cluster only.
    Architecture: 17 → 8 → 8 → 6 (lightweight for cluster-specific learning)
    
    Input: 17D raw features only
    Output: 6D predictions (Δd, ΔU₀, ΔU₁, ΔU₂, ΔT, ΔnParticle)
    """
    
    def __init__(self, input_dim: int = 17, output_dim: int = 6, hidden_dim: int = 8):
        """
        Args:
            input_dim: Input feature dimension (17 raw features)
            output_dim: Output dimension (6 regression targets)
            hidden_dim: Hidden layer dimension (8 for efficiency)
        """
        super().__init__()
        
        # 2-layer architecture: 17 → 8 → 8 → 6
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 17] - raw input features
            
        Returns:
            predictions: [batch_size, output_dim]
        """
        return self.net(x)


class GatingNetworkSupervised(nn.Module):
    """Gating network trained as classifier on cluster labels.
    
    Architecture: 2 hidden layers
    17D input → 64 → 32 → 4 logits (for CrossEntropyLoss classification)
    
    Unlike SelfOrgMOE which learns soft weights via MSE,
    this learns hard cluster selection via CrossEntropyLoss.
    """
    
    def __init__(self, input_dim: int = 17, n_clusters: int = 4):
        """
        Args:
            input_dim: Input feature dimension (17 raw features)
            n_clusters: Number of clusters (4 for persistent-only data)
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_clusters),  # Output logits for classification
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 17] - raw input features
            
        Returns:
            logits: [batch_size, n_clusters] - prediction scores for classification
        """
        logits = self.net(x)
        return logits


class HybridRUCSupervised(nn.Module):
    """Supervised Cluster Routing - Physics-informed mixture of experts.
    
    Key features:
    1. Unsupervised clustering: GMM on 11D physics features
    2. Supervised expert training: Each expert trained on its cluster's data only
    3. Supervised gating: Classifier on explicit cluster labels (CrossEntropyLoss)
    4. Hard routing: Each sample routed to ONE dominant cluster
    
    This differs from SelfOrgMOE where:
    - SelfOrgMOE: Soft weights, gating learns from regression MSE, experts share data
    - Supervised: Hard selection, gating learns from cluster classification, experts separate
    """
    
    def __init__(self,
                 gating_input_dim: int = 17,
                 clustering_input_dim: int = 11,
                 output_dim: int = 6,
                 n_clusters: int = 4,
                 expert_hidden_dim: int = 8,
                 device: str = 'cpu'):
        """
        Args:
            gating_input_dim: Input dimension for gating (17 raw features)
            clustering_input_dim: Physics-only dimension for GMM clustering (11)
            output_dim: Output dimension (6 regression targets: Δd, ΔU₀, ΔU₁, ΔU₂, ΔT, ΔnParticle)
            n_clusters: Number of clusters (4 for persistent-only data)
            expert_hidden_dim: Hidden dimension for expert networks (8)
            device: Device to use ('cpu' or 'cuda')
        """
        super().__init__()
        
        self.gating_input_dim = gating_input_dim
        self.clustering_input_dim = clustering_input_dim
        self.output_dim = output_dim
        self.n_clusters = n_clusters
        self.device = torch.device(device)
        
        # Clustering (unsupervised GMM on physics features only)
        self.gmm = None
        self.cluster_centers = None
        self.cluster_labels = None  # Will store hard cluster assignments
        
        # Expert networks: n_clusters experts (trained separately per cluster)
        self.experts = nn.ModuleList([
            ExpertNetwork(gating_input_dim, output_dim, expert_hidden_dim)
            for _ in range(self.n_clusters)
        ])
        
        # Gating network: trained as CLASSIFIER on cluster labels (not regression)
        self.gating_net = GatingNetworkSupervised(gating_input_dim, self.n_clusters)
        
        self.to(device)
    
    def fit_clustering(self, X_physics: np.ndarray, persistence_flags: np.ndarray = None,
                       random_state: int = 42) -> np.ndarray:
        """
        Fit Gaussian Mixture Model (GMM) on 11D physics features.
        
        Returns hard cluster assignments that will be used as ground truth
        for training the gating network as a classifier.
        
        Physics-informed clustering on:
        - We: Weber number
        - Oh: Ohnesorge number  
        - Re: Reynolds number
        - Urel_mag: Relative velocity (aerodynamic regime)
        - mass_proxy: Droplet mass
        - delT_T_boil: Thermal stress
        - del_nParticle: Breakup indicator
        - delUx: Change in x-velocity component
        - delUy: Change in y-velocity component
        - delUz: Change in z-velocity component
        - deld: Change in diameter
        
        Args:
            X_physics: [N, 11] physics-only features (normalized)
            persistence_flags: [N] binary flags (optional, info only)
            random_state: Random seed for reproducibility
            
        Returns:
            cluster_labels: [N] hard cluster assignments for each sample
        """
        print(f"\nFitting GMM clustering on {self.clustering_input_dim}D physics features...")
        print(f"  Total samples: {X_physics.shape[0]:,}")
        if persistence_flags is not None:
            print(f"  Non-persistent samples: {(persistence_flags==0).sum():,} ({100*(persistence_flags==0).sum()/len(persistence_flags):.2f}%)")
        
        # Use representative sample for GMM fitting
        sample_size = min(50000, max(10000, len(X_physics) // 40))
        if len(X_physics) > sample_size:
            print(f"  Sampling {sample_size:,} representative samples for GMM fitting...")
            sample_indices = np.random.RandomState(random_state).choice(
                len(X_physics), size=sample_size, replace=False
            )
            X_sample = X_physics[sample_indices]
        else:
            X_sample = X_physics
        
        # Fit Gaussian Mixture Model
        self.gmm = GaussianMixture(n_components=self.n_clusters, 
                                   random_state=random_state,
                                   n_init=5,
                                   max_iter=200)
        self.gmm.fit(X_sample)
        print(f"  ✓ GMM fitted with {self.n_clusters} clusters")
        
        self.cluster_centers = torch.tensor(
            self.gmm.means_,
            dtype=torch.float32,
            device=self.device
        )
        
        # Get hard cluster assignments for ALL data
        cluster_labels = self.gmm.predict(X_physics)
        self.cluster_labels = cluster_labels
        
        # Print cluster distribution
        unique, counts = np.unique(cluster_labels, return_counts=True)
        print(f"  ✓ Final cluster distribution:")
        for cluster_id in range(self.n_clusters):
            if cluster_id in unique:
                count = counts[np.where(unique == cluster_id)[0][0]]
                pct = 100 * count / len(X_physics)
                print(f"    Cluster {cluster_id}: {count:,} samples ({pct:.1f}%)")
        
        return cluster_labels
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with HARD cluster selection via gating classifier.
        
        Unlike SelfOrgMOE which uses soft weights, this uses the gating network
        to select a single best cluster for each sample (hard routing).
        
        Args:
            x: [batch_size, 17] - raw input features only
            
        Returns:
            predictions: [batch_size, output_dim] predictions from selected expert
            gating_logits: [batch_size, n_clusters] classification logits (for CrossEntropyLoss)
        """
        # Get gating network output (classification logits, not regression weights)
        gating_logits = self.gating_net(x)  # [batch, n_clusters]
        
        # Hard selection: choose cluster with highest logit
        selected_clusters = torch.argmax(gating_logits, dim=1)  # [batch] - cluster indices
        
        # Get predictions from selected experts
        batch_size = x.shape[0]
        expert_outputs_all = []
        for expert in self.experts:
            expert_outputs_all.append(expert(x))  # [batch, output_dim]
        
        # Stack: [batch, n_clusters, output_dim]
        expert_outputs_stacked = torch.stack(expert_outputs_all, dim=1)
        
        # Select output from chosen expert for each sample
        # Use advanced indexing to select per-sample
        batch_indices = torch.arange(batch_size, device=self.device)
        predictions = expert_outputs_stacked[batch_indices, selected_clusters, :]  # [batch, output_dim]
        
        return predictions, gating_logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model instantiation and parameter count
    model = HybridRUCSupervised(
        gating_input_dim=17,
        clustering_input_dim=11,
        output_dim=6,
        n_clusters=4,
        expert_hidden_dim=8,
        device='cpu'
    )
    
    print("\n" + "="*70)
    print("Supervised Cluster Routing (SUC) - Physics-Informed MoE")
    print("="*70)
    print(f"Architecture: Supervised routing with per-cluster experts")
    print(f"Gating input dimension: {model.gating_input_dim}")
    print(f"Clustering dimension: {model.clustering_input_dim}")
    print(f"Output dimension: {model.output_dim}")
    print(f"Number of clusters: {model.n_clusters}")
    print(f"Routing type: HARD selection via gating classifier (CrossEntropyLoss)")
    print(f"Expert training: Separate per-cluster (not mixed)")
    print(f"\nExperts:")
    for i, expert in enumerate(model.experts):
        params = sum(p.numel() for p in expert.parameters())
        print(f"  Expert {i} (cluster-specific): {params:,} parameters")
    gating_params = sum(p.numel() for p in model.gating_net.parameters())
    print(f"\nGating network (classifier): {gating_params:,} parameters")
    print(f"Total trainable parameters: {model.count_parameters():,}")
    print("="*70)
    print("\nKey differences from SelfOrgMOE:")
    print("  • SelfOrgMOE: Soft weights + regression MSE → implicit learning from targets")
    print("  • SUC: Hard selection + CrossEntropyLoss → explicit learning from labels")
    print("="*70)

