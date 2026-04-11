#!/usr/bin/env python3
"""
Physics-Informed Clustering using GMM (Subsampled)
===================================================

Creates 3-cluster assignments using Gaussian Mixture Model on subsampled data:
  1. Load 1.9M training samples
  2. Subsample to 500k (or configurable size)
  3. Fit GMM on subsample (fast & reliable)
  4. Predict on full train/val/test sets (preserves probabilistic nature)

This is the best of both worlds:
  - GMM's probabilistic framework (vs KMeans hard clustering)
  - Computational feasibility (vs fitting on 2M directly)
  - Full dataset coverage (predict on all 2.4M samples)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pickle
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class PhysicsFeatureClustererGMMSubsampled:
    """GMM clustering on physics features using subsampled training."""
    
    def __init__(self, 
                 train_csv: str = './data/train_paired.csv',
                 val_csv: str = './data/val_paired.csv',
                 test_csv: str = './data/test_paired.csv',
                 output_dir: str = './data',
                 n_clusters: int = 3,
                 subsample_size: int = 500000,
                 random_seed: int = 42):
        """
        Initialize clusterer.
        
        Args:
            subsample_size: Number of training samples to use for GMM fitting
                           (pred icted on full dataset)
        """
        self.train_csv = Path(train_csv)
        self.val_csv = Path(val_csv)
        self.test_csv = Path(test_csv)
        self.output_dir = Path(output_dir)
        self.n_clusters = n_clusters
        self.subsample_size = subsample_size
        self.random_seed = random_seed
        self.gmm = None
        self.scaler = None
    
    def load_data(self):
        """Load train/val/test CSVs."""
        print("\nLoading data...")
        train_df = pd.read_csv(self.train_csv)
        val_df = pd.read_csv(self.val_csv)
        test_df = pd.read_csv(self.test_csv)
        
        print(f"  Train: {len(train_df):,} samples, {train_df.shape[1]} columns")
        print(f"  Val:   {len(val_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")
        print(f"  Total: {len(train_df) + len(val_df) + len(test_df):,} samples")
        
        return train_df, val_df, test_df
    
    def compute_physics_features(self, 
                                train_df: pd.DataFrame,
                                val_df: pd.DataFrame,
                                test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Compute physics features from absolute columns."""
        print("\nComputing physics features (using absolute values from CSV)...")
        
        feature_names = ['Re', 'Urel_mag', 'We', 'delta_T', 'delta_Urel_mag', 'delta_d', 'delta_nParticle']
        
        def compute_features_for_df(df):
            """Helper to compute features for a dataframe."""
            features = {}
            
            # Reynolds number: Re = rho * U * d / mu
            u_mag = np.sqrt(df['in_U:0_abs']**2 + df['in_U:1_abs']**2 + df['in_U:2_abs']**2)
            features['Re'] = df['in_rho_abs'] * u_mag * df['in_d_abs'] / (df['in_mu_abs'] + 1e-12)
            
            # Relative velocity magnitude
            features['Urel_mag'] = u_mag
            
            # Weber number: We = rho * U^2 * d / sigma
            features['We'] = df['in_rho_abs'] * u_mag**2 * df['in_d_abs'] / (df['in_sigma_abs'] + 1e-12)
            
            # Temperature change
            features['delta_T'] = df['out_T_abs'] - df['in_T_abs']
            
            # Velocity change
            u_mag_out = np.sqrt(df['out_U:0_abs']**2 + df['out_U:1_abs']**2 + df['out_U:2_abs']**2)
            features['delta_Urel_mag'] = u_mag_out - u_mag
            
            # Diameter change
            features['delta_d'] = df['out_d_abs'] - df['in_d_abs']
            
            # Particle count change
            features['delta_nParticle'] = df['out_nParticle_abs'] - df['in_nParticle_abs']
            
            # Convert to numpy array
            X = np.column_stack([features[name] for name in feature_names])
            
            # Handle infinities and NaNs
            X = np.nan_to_num(X, nan=0, posinf=1e6, neginf=-1e6)
            
            return X
        
        X_train = compute_features_for_df(train_df)
        X_val = compute_features_for_df(val_df)
        X_test = compute_features_for_df(test_df)
        
        print(f"  Train shape: {X_train.shape}")
        print(f"  Val shape:   {X_val.shape}")
        print(f"  Test shape:  {X_test.shape}")
        print(f"  Features: {feature_names}")
        
        return X_train, X_val, X_test, feature_names
    
    def fit_gmm(self, X_train_full: np.ndarray):
        """
        Fit GMM on subsampled training data.
        
        Args:
            X_train_full: Full training features (will be subsampled)
        """
        # Subsample
        subsample_indices = np.random.RandomState(self.random_seed).choice(
            len(X_train_full), 
            size=min(self.subsample_size, len(X_train_full)),
            replace=False
        )
        X_subsample = X_train_full[subsample_indices]
        
        print(f"\nFitting {self.n_clusters}-component GMM on subsample...")
        print(f"  Subsample size: {len(X_subsample):,} / {len(X_train_full):,} training samples")
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_subsample)
        
        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_clusters, 
            random_state=self.random_seed,
            n_init=10,
            max_iter=200,
            verbose=1
        )
        self.gmm.fit(X_scaled)
        
        print(f"\n  BIC: {self.gmm.bic(X_scaled):.2f}")
        print(f"  AIC: {self.gmm.aic(X_scaled):.2f}")
        print(f"  Log-likelihood: {self.gmm.score(X_scaled):.4f}")
        
        # Check cluster distribution on subsample
        labels_subsample = self.gmm.predict(X_scaled)
        unique, counts = np.unique(labels_subsample, return_counts=True)
        print(f"\n  Cluster distribution on subsample:")
        for cluster_id, count in zip(unique, counts):
            pct = 100 * count / len(labels_subsample)
            print(f"    Cluster {cluster_id}: {count:,} ({pct:.1f}%)")
    
    def assign_clusters(self, 
                       train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       X_train: np.ndarray,
                       X_val: np.ndarray,
                       X_test: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Assign cluster IDs to all datasets using fitted GMM."""
        print("\nAssigning clusters to full dataset...")
        
        # Scale and predict
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        train_clusters = self.gmm.predict(X_train_scaled)
        val_clusters = self.gmm.predict(X_val_scaled)
        test_clusters = self.gmm.predict(X_test_scaled)
        
        # Also get soft cluster assignments (responsibilities)
        train_probs = self.gmm.predict_proba(X_train_scaled)
        val_probs = self.gmm.predict_proba(X_val_scaled)
        test_probs = self.gmm.predict_proba(X_test_scaled)
        
        train_df['cluster_id_physics'] = train_clusters
        val_df['cluster_id_physics'] = val_clusters
        test_df['cluster_id_physics'] = test_clusters
        
        # Store probabilities for each cluster
        for i in range(self.n_clusters):
            train_df[f'cluster_{i}_prob'] = train_probs[:, i]
            val_df[f'cluster_{i}_prob'] = val_probs[:, i]
            test_df[f'cluster_{i}_prob'] = test_probs[:, i]
        
        print(f"  Train: {len(train_df):,} samples assigned")
        print(f"  Val:   {len(val_df):,} samples assigned")
        print(f"  Test:  {len(test_df):,} samples assigned")
        
        # Show distribution on full dataset
        unique, counts = np.unique(train_clusters, return_counts=True)
        print(f"\n  Cluster distribution on full training data:")
        for cluster_id, count in zip(unique, counts):
            pct = 100 * count / len(train_clusters)
            print(f"    Cluster {cluster_id}: {count:,} ({pct:.1f}%)")
        
        return train_df, val_df, test_df
    
    def cluster(self):
        """Execute full clustering pipeline."""
        print("=" * 80)
        print("PHYSICS-INFORMED CLUSTERING (GMM - Subsampled)")
        print("=" * 80)
        
        # Load
        train_df, val_df, test_df = self.load_data()
        
        # Compute features
        X_train, X_val, X_test, feature_names = self.compute_physics_features(
            train_df, val_df, test_df
        )
        
        # Fit GMM on subsample
        self.fit_gmm(X_train)
        
        # Assign clusters to full dataset
        train_df, val_df, test_df = self.assign_clusters(
            train_df, val_df, test_df, X_train, X_val, X_test
        )
        
        # Save
        print("\nSaving clustered data...")
        
        train_path = self.output_dir / 'train_paired_gmm.csv'
        val_path = self.output_dir / 'val_paired_gmm.csv'
        test_path = self.output_dir / 'test_paired_gmm.csv'
        
        train_df.to_csv(train_path, index=False)
        print(f"  ✓ {train_path}")
        
        val_df.to_csv(val_path, index=False)
        print(f"  ✓ {val_path}")
        
        test_df.to_csv(test_path, index=False)
        print(f"  ✓ {test_path}")
        
        # Save GMM model
        model_path = self.output_dir / 'gmm_clustering_model_subsampled.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'gmm': self.gmm,
                'scaler': self.scaler,
                'feature_names': feature_names,
                'n_clusters': self.n_clusters,
                'subsample_size': self.subsample_size
            }, f)
        print(f"  ✓ {model_path}")
        
        print("\n" + "=" * 80)
        print("✓ GMM CLUSTERING COMPLETE")
        print("=" * 80)
        print(f"\nClustered datasets saved (with cluster probabilities):")
        print(f"  • {train_path} ({len(train_df):,} samples)")
        print(f"  • {val_path} ({len(val_df):,} samples)")
        print(f"  • {test_path} ({len(test_df):,} samples)")
        print(f"\nAdditional columns added:")
        print(f"  • cluster_id_physics (hard assignment)")
        probs = ', '.join([f"cluster_{i}_prob" for i in range(self.n_clusters)])
        print(f"  • {probs} (soft assignments)")
        print()
        
        return train_df, val_df, test_df


if __name__ == '__main__':
    clusterer = PhysicsFeatureClustererGMMSubsampled(
        train_csv='./data/train_paired.csv',
        val_csv='./data/val_paired.csv',
        test_csv='./data/test_paired.csv',
        output_dir='./data',
        n_clusters=2,
        subsample_size=500000  # Fit on 500k, predict on all
    )
    
    train_df, val_df, test_df = clusterer.cluster()
