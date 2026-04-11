"""
Data Preparation for Simplified GNN Training
=============================================

Creates paired timestep data for property change prediction:
  Input: particle properties at time t + context (Eulerian) + physics-engineered features
  Output: Δproperties = properties(t+1) - properties(t)

Handles:
  1. Injection events: identified and saved separately
  2. Training data: includes injection events as normal drops
  3. Normalization: StandardScaler for stable training
  4. Splits: train/val/test by particle to avoid temporal leakage
  
Features (17 total - Phase 3 physics-aware approach):
  - Lagrangian (6): d, U:0-2, T, nParticle
  - Material (3): rho, mu, sigma
  - Eulerian context (7): euler_T, euler_U:0-2, euler_H2O, euler_p, euler_rho
  - Computed (1): mass_proxy
  
Phase 2 achieved R²=0.9801 on d/U/T but struggled with nParticle (68.5% loss).
Phase 3 adds mass context via d³·nP feature to disambiguate breakup vs evaporation phases.
Key insight: Let the GNN learn physics relationships, not pre-engineered numbers.
TAU filtering captures interaction physics better than engineered features.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class SimpleGNNDataPreparator:
    """Prepares paired timestep data for GNN training."""
    
    # Input features: Lagrangian (droplet) + Eulerian (gas environment)
    # 17 features: 6 Lagrangian + 3 Material + 7 Eulerian + 1 mass_proxy
    INPUT_FEATURES = [
        # Lagrangian: droplet properties
        'd', 'U:0', 'U:1', 'U:2', 'T', 'nParticle',
        # Material properties (temperature/evaporation dependent)
        'rho', 'mu', 'sigma',
        # Eulerian: surrounding gas environment (critical for predicting changes)
        'euler_T', 'euler_U:0', 'euler_U:1', 'euler_U:2', 'euler_H2O', 'euler_p', 'euler_rho',
        # Mass proxy: represents parcel mass (m ∝ d³·nParticle)
        'mass_proxy',  # = d³ * nParticle, scaled to avoid numerical overflow
    ]
    
    # Output features: deltas to predict (droplet properties + persistence)
    # 7 outputs: 6 property deltas + 1 persistence flag
    # Note: rho, mu, sigma are NOT predicted here - they're 99%+ correlated with T
    # A separate MLP can reconstruct them from T post-hoc
    OUTPUT_FEATURES = [
        'd', 'U:0', 'U:1', 'U:2', 'T', 'nParticle',  # Physical property changes
        'persists'  # Binary: 1 if drop survives to next timestep, 0 if evaporated
    ]
    
    # Position features (used for TAU graph construction, not normalized)
    POSITION_FEATURES = ['Points:0', 'Points:1', 'Points:2']
    
    # Velocity features (used for TAU filtering, not normalized)
    VELOCITY_FEATURES = ['U:0', 'U:1', 'U:2']
    
    def __init__(self, 
                 input_csv: str = '../data/step3_with_injection_labels.csv',
                 output_dir: str = '../data/',
                 k_neighbors: int = 10):
        """
        Args:
            input_csv: path to main dataset with injection labels
            output_dir: output directory for prepared data
            k_neighbors: number of nearest neighbors for TAU filtering
        """
        self.input_csv = input_csv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.k_neighbors = k_neighbors
        
        self.scaler_input = StandardScaler()
        self.scaler_output = StandardScaler()
        self.injection_idxs = []  # Track which samples are injection events
    
    def load_data(self) -> pd.DataFrame:
        """Load main dataset."""
        print("Loading main dataset...")
        df = pd.read_csv(self.input_csv)
        print(f"  Rows: {len(df):,}")
        print(f"  Timesteps: {df['timestep'].min():.0f} to {df['timestep'].max():.0f}")
        print(f"  Unique particles: {df['origId'].nunique():,}")
        
        # Check for injection labels
        if 'is_injection_event' in df.columns:
            n_inj = (df['is_injection_event'] == 1).sum()
            print(f"  Injection events: {n_inj} ({100*n_inj/len(df):.2f}%)")
        
        return df
    
    def identify_injection_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify injection events (first appearance of new origIDs).
        Save to separate CSV for reference.
        """
        print("\nIdentifying injection events...")
        
        injection_list = []
        seen_orig_ids = set()
        
        for ts in sorted(df['timestep'].unique()):
            ts_data = df[df['timestep'] == ts]
            curr_orig_ids = set(ts_data['origId'].unique())
            new_orig_ids = curr_orig_ids - seen_orig_ids
            
            if new_orig_ids:
                inj_data = ts_data[ts_data['origId'].isin(new_orig_ids)].copy()
                inj_data['injection_timestep'] = ts
                injection_list.append(inj_data)
            
            seen_orig_ids = curr_orig_ids
        
        if injection_list:
            injection_df = pd.concat(injection_list, ignore_index=True)
            n_inj = len(injection_df)
            n_unique = injection_df['origId'].nunique()
            print(f"  Found {n_inj} injection events from {n_unique} unique origIDs")
            
            # Save injection events
            inj_path = self.output_dir / 'injection_events.csv'
            injection_df.to_csv(inj_path, index=False)
            print(f"  Saved: {inj_path}")
            
            return injection_df
        else:
            print("  No injection events found")
            return pd.DataFrame()
    
    def create_paired_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[bool]]:
        """
        Create paired timestep data using chunked processing to save memory.
        For each particle at t, find same particle at t+1 and compute deltas.
        
        Returns:
            paired_df: DataFrame with paired data
            is_injection: list of booleans marking injection events
        """
        print("\nCreating paired timesteps...")
        
        # Sort by origId and timestep
        df_sorted = df.sort_values(['origId', 'timestep']).reset_index(drop=True)
        del df  # Free original dataframe
        
        # Compute mass proxy feature
        print("\n  Computing mass proxy feature...")
        if 'd' in df_sorted.columns and 'nParticle' in df_sorted.columns:
            d_val = df_sorted['d'].values
            nP_val = df_sorted['nParticle'].values
            df_sorted['mass_proxy'] = (d_val**3 * nP_val) / 1e12  # Scale to prevent overflow
            mp_min = df_sorted['mass_proxy'].min()
            mp_max = df_sorted['mass_proxy'].max()
            print(f"    ✓ mass_proxy = d³·nP / 1e12 (range: {mp_min:.6f} to {mp_max:.6f})")
        else:
            df_sorted['mass_proxy'] = 0.0
            print(f"    ✗ mass_proxy (missing d or nParticle, set to 0)")
        
        # Create masks for valid consecutive pairs
        next_orig_id = df_sorted['origId'].shift(-1)
        next_timestep = df_sorted['timestep'].shift(-1)
        
        orig_id_match = df_sorted['origId'] == next_orig_id
        timestep_consecutive = (next_timestep - df_sorted['timestep']) == 1
        valid_pairs = orig_id_match & timestep_consecutive & (~next_timestep.isna())
        
        # Identify disappearing particles (in current t, but NOT in t+1)
        # These are particles at their final timestep where the next row has a different origId
        disappearing_mask = ~orig_id_match & (~next_orig_id.isna())
        
        print(f"  Total rows: {len(df_sorted):,}")
        print(f"  Valid pairs (persisting): {valid_pairs.sum():,}")
        print(f"  Disappearing particles: {disappearing_mask.sum():,}")
        
        # ============================================================================
        # PART 1: Create normal pairs for particles that persist to next timestep
        # ============================================================================
        valid_idx = np.where(valid_pairs)[0]
        next_idx = valid_idx + 1
        
        # Build paired data in chunks to manage memory
        chunk_size = 100000
        pair_dicts = []
        
        for chunk_start in range(0, len(valid_idx), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(valid_idx))
            chunk_valid_idx = valid_idx[chunk_start:chunk_end]
            chunk_next_idx = next_idx[chunk_start:chunk_end]
            
            chunk_data = {}
            curr_origids = df_sorted.iloc[chunk_valid_idx]['origId'].values
            next_origids = df_sorted.iloc[chunk_next_idx]['origId'].values
            
            chunk_data['origId'] = curr_origids
            chunk_data['timestep'] = df_sorted.iloc[chunk_valid_idx]['timestep'].values.astype(int)
            chunk_data['timestep_next'] = df_sorted.iloc[chunk_next_idx]['timestep'].values.astype(int)
            chunk_data['is_disappearance'] = 0  # Normal pair, not disappearance event
            
            # Input features: current state
            for feat in self.INPUT_FEATURES:
                if feat in df_sorted.columns:
                    chunk_data[f'in_{feat}'] = df_sorted.iloc[chunk_valid_idx][feat].values
            
            # Output features: deltas
            for feat in self.OUTPUT_FEATURES:
                if feat == 'persists':
                    # Binary flag: 1 if origId persists to next timestep, 0 if particle disappears
                    chunk_data[f'out_{feat}'] = (curr_origids == next_origids).astype(float)
                elif feat in df_sorted.columns:
                    curr_vals = df_sorted.iloc[chunk_valid_idx][feat].values
                    next_vals = df_sorted.iloc[chunk_next_idx][feat].values
                    chunk_data[f'out_{feat}'] = (next_vals - curr_vals)
            
            # Positions (for VOI graph construction)
            for pos_feat in self.POSITION_FEATURES:
                if pos_feat in df_sorted.columns:
                    chunk_data[f'pos_{pos_feat}'] = df_sorted.iloc[chunk_valid_idx][pos_feat].values
            
            pair_dicts.append(pd.DataFrame(chunk_data))
        
        # Concatenate all chunks
        paired_df = pd.concat(pair_dicts, ignore_index=True)
        print(f"  Created {len(paired_df):,} paired samples (persisting particles)")
        
        # ============================================================================
        # PART 2: Create special samples for disappearing particles
        # These have persists=0 and all deltas=0 (since no t+1 data exists)
        # ============================================================================
        disappearing_idx = np.where(disappearing_mask)[0]
        n_disappearing = len(disappearing_idx)
        
        if n_disappearing > 0:
            print(f"  Creating {n_disappearing:,} disappearance samples...")
            
            # Process disappearing particles in chunks
            disappear_dicts = []
            for chunk_start in range(0, n_disappearing, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_disappearing)
                chunk_disappear_idx = disappearing_idx[chunk_start:chunk_end]
                
                chunk_data = {}
                
                chunk_data['origId'] = df_sorted.iloc[chunk_disappear_idx]['origId'].values
                chunk_data['timestep'] = df_sorted.iloc[chunk_disappear_idx]['timestep'].values.astype(int)
                chunk_data['timestep_next'] = df_sorted.iloc[chunk_disappear_idx]['timestep'].values.astype(int)  # Same as current
                chunk_data['is_disappearance'] = 1  # Mark as disappearance event
                
                # Input features: current state
                for feat in self.INPUT_FEATURES:
                    if feat in df_sorted.columns:
                        chunk_data[f'in_{feat}'] = df_sorted.iloc[chunk_disappear_idx][feat].values
                
                # Output features: all set to 0 (no change because no next state)
                for feat in self.OUTPUT_FEATURES:
                    if feat == 'persists':
                        # 0 = particle disappears
                        chunk_data[f'out_{feat}'] = np.zeros(len(chunk_disappear_idx), dtype=float)
                    else:
                        # All deltas = 0 (unknown, particle evaporates)
                        chunk_data[f'out_{feat}'] = np.zeros(len(chunk_disappear_idx), dtype=float)
                
                # Positions (for VOI graph construction)
                for pos_feat in self.POSITION_FEATURES:
                    if pos_feat in df_sorted.columns:
                        chunk_data[f'pos_{pos_feat}'] = df_sorted.iloc[chunk_disappear_idx][pos_feat].values
                
                disappear_dicts.append(pd.DataFrame(chunk_data))
            
            # Append disappearing samples
            if disappear_dicts:
                disappear_df = pd.concat(disappear_dicts, ignore_index=True)
                paired_df = pd.concat([paired_df, disappear_df], ignore_index=True)
                print(f"  Added {len(disappear_df):,} disappearance samples")
        
        # CRITICAL: Shuffle the data to interleave disappearing samples throughout training
        # This prevents the model from learning "always predict persists=1" in early epochs
        paired_df = paired_df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"  Shuffled {len(paired_df):,} samples to distribute disappearing events")
        
        print(f"  Total paired samples: {len(paired_df):,}")
        
        # Mark injection events
        if 'is_injection_event' in df_sorted.columns:
            inj_orig_ids = set(df_sorted[df_sorted['is_injection_event'] == 1]['origId'].unique())
            is_injection = [orig_id in inj_orig_ids for orig_id in paired_df['origId']]
            n_inj = sum(is_injection)
            print(f"  Injection event samples: {n_inj} ({100*n_inj/len(paired_df):.2f}%)")
        else:
            is_injection = [False] * len(paired_df)
        
        del df_sorted  # Free sorted dataframe
        return paired_df, is_injection
    
    def normalize_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Normalize input and output features using StandardScaler.
        
        CRITICAL: 
        - Scalers are fitted ONLY on training data, then applied to val/test (prevents data leakage)
        - Persistence flag is NOT normalized (binary 0/1, not normalized)
        - Positions are NOT normalized (kept in physical units for VOI)
        
        Returns:
            Tuple of (train_norm, val_norm, test_norm, feature_cols_dict)
        """
        print("\nNormalizing features...")
        
        # Feature columns - EXCLUDE persistence from normalization
        input_cols = [f'in_{feat}' for feat in self.INPUT_FEATURES 
                      if f'in_{feat}' in train_df.columns]
        output_cols = [f'out_{feat}' for feat in self.OUTPUT_FEATURES 
                       if f'out_{feat}' in train_df.columns and feat != 'persists']
        persistence_col = 'out_persists'  # Keep separate, don't normalize
        
        print(f"  Input features: {len(input_cols)}")
        print(f"  Output features (normalized): {len(output_cols)}")
        print(f"  Persistence flag (NOT normalized): {persistence_col}")
        
        # FIX: Fit scalers ONLY on training data to prevent data leakage
        print("\n  Fitting scalers on TRAINING data only...")
        self.scaler_input.fit(train_df[input_cols])
        self.scaler_output.fit(train_df[output_cols])
        
        # Transform all splits using the training scaler
        train_norm = train_df.copy()
        val_norm = val_df.copy()
        test_norm = test_df.copy()
        
        train_norm[input_cols] = self.scaler_input.transform(train_df[input_cols])
        train_norm[output_cols] = self.scaler_output.transform(train_df[output_cols])
        # Persistence stays as raw 0/1
        
        val_norm[input_cols] = self.scaler_input.transform(val_df[input_cols])
        val_norm[output_cols] = self.scaler_output.transform(val_df[output_cols])
        # Persistence stays as raw 0/1
        
        test_norm[input_cols] = self.scaler_input.transform(test_df[input_cols])
        test_norm[output_cols] = self.scaler_output.transform(test_df[output_cols])
        # Persistence stays as raw 0/1
        
        # Verify normalization is correct
        print(f"\n  Train stats (after scaler fit on train):")
        print(f"    Input: mean={train_norm[input_cols].mean().mean():.4f}, std={train_norm[input_cols].std().mean():.4f}")
        print(f"    Output (excluding persists): mean={train_norm[output_cols].mean().mean():.4f}, std={train_norm[output_cols].std().mean():.4f}")
        if persistence_col in train_norm.columns:
            print(f"    Persistence: unique={train_norm[persistence_col].unique()}, mean={train_norm[persistence_col].mean():.4f}")
        
        print(f"\n  Val stats (applied train scaler):")
        print(f"    Input: mean={val_norm[input_cols].mean().mean():.4f}, std={val_norm[input_cols].std().mean():.4f}")
        print(f"    Output (excluding persists): mean={val_norm[output_cols].mean().mean():.4f}, std={val_norm[output_cols].std().mean():.4f}")
        if persistence_col in val_norm.columns:
            print(f"    Persistence: unique={val_norm[persistence_col].unique()}, mean={val_norm[persistence_col].mean():.4f}")
        
        print(f"\n  Test stats (applied train scaler):")
        print(f"    Input: mean={test_norm[input_cols].mean().mean():.4f}, std={test_norm[input_cols].std().mean():.4f}")
        print(f"    Output (excluding persists): mean={test_norm[output_cols].mean().mean():.4f}, std={test_norm[output_cols].std().mean():.4f}")
        
        # Include persistence in output_cols for model (it's separate, not normalized)
        all_output_cols = output_cols + [persistence_col]
        
        feature_cols = {
            'input_cols': input_cols,
            'output_cols': all_output_cols,  # Includes persistence (raw 0/1)
            'persistence_col': persistence_col,  # Mark which one is persistence
            'position_cols': [f'pos_{feat}' for feat in self.POSITION_FEATURES 
                            if f'pos_{feat}' in train_df.columns],
            'velocity_cols': [f'in_{feat}' for feat in self.VELOCITY_FEATURES
                            if f'in_{feat}' in train_df.columns]
        }
        
        return train_norm, val_norm, test_norm, feature_cols
    
    def create_splits(self, paired_df: pd.DataFrame, 
                     train_frac: float = 0.7,
                     val_frac: float = 0.15,
                     test_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/val/test splits by PARTICLE groups.
        
        Since each pair is tied (t, t+1), we split by ORIGID (particle) to ensure:
        - All consecutive timesteps of same particle stay together
        - No temporal leakage from adjacent timesteps in different splits
        - Each split gets diverse particles from all time ranges
        """
        print("\nCreating train/val/test splits (by particle group)...")
        
        # Group by particle ID
        unique_particles = paired_df['origId'].unique()
        n_particles = len(unique_particles)
        
        # Shuffle particles (not individual samples)
        shuffled_particles = np.random.RandomState(42).permutation(unique_particles)
        
        train_split_idx = int(n_particles * train_frac)
        val_split_idx = int(n_particles * (train_frac + val_frac))
        
        train_particles = set(shuffled_particles[:train_split_idx])
        val_particles = set(shuffled_particles[train_split_idx:val_split_idx])
        test_particles = set(shuffled_particles[val_split_idx:])
        
        train_df = paired_df[paired_df['origId'].isin(train_particles)].copy()
        val_df = paired_df[paired_df['origId'].isin(val_particles)].copy()
        test_df = paired_df[paired_df['origId'].isin(test_particles)].copy()
        
        print(f"  Train: {len(train_df):,} samples from {len(train_particles):,} particles")
        print(f"  Val:   {len(val_df):,} samples from {len(val_particles):,} particles")
        print(f"  Test:  {len(test_df):,} samples from {len(test_particles):,} particles")
        
        # Verify distribution across timesteps
        print(f"\n  Train timestep coverage: {train_df['timestep'].min():.0f}-{train_df['timestep'].max():.0f}")
        print(f"  Val timestep coverage:   {val_df['timestep'].min():.0f}-{val_df['timestep'].max():.0f}")
        print(f"  Test timestep coverage:  {test_df['timestep'].min():.0f}-{test_df['timestep'].max():.0f}")
        
        return train_df, val_df, test_df
    
    def prepare(self):
        """Execute full data preparation pipeline."""
        print("="*80)
        print("GNN DATA PREPARATION (SIMPLIFIED)")
        print("="*80)
        
        # Load
        df = self.load_data()
        
        # Identify injection events
        inj_df = self.identify_injection_events(df)
        
        # Create paired data
        paired_df, is_injection = self.create_paired_data(df)
        paired_df['is_injection'] = is_injection
        
        # CREATE SPLITS FIRST (on unnormalized data to avoid data leakage)
        print("\nCreating train/val/test splits (on unnormalized data)...")
        train_df_raw, val_df_raw, test_df_raw = self.create_splits(paired_df)
        
        # THEN normalize using only training data statistics
        train_df, val_df, test_df, feature_cols = self.normalize_features(
            train_df_raw, val_df_raw, test_df_raw
        )
        
        # SHUFFLE within each split to ensure random row ordering
        # (critical for random batch sampling in training)
        print("\nShuffling rows within each split...")
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=43).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=44).reset_index(drop=True)
        print("  ✓ Train shuffled")
        print("  ✓ Val shuffled")
        print("  ✓ Test shuffled")
        
        # Save
        print("\nSaving prepared data...")
        train_df.to_csv(self.output_dir / 'train_paired.csv', index=False)
        val_df.to_csv(self.output_dir / 'val_paired.csv', index=False)
        test_df.to_csv(self.output_dir / 'test_paired.csv', index=False)
        
        # Save metadata
        metadata = {
            'scaler_input': self.scaler_input,
            'scaler_output': self.scaler_output,
            'feature_cols': feature_cols,
            'k_neighbors': self.k_neighbors,
            'n_samples': {
                'train': len(train_df),
                'val': len(val_df),
                'test': len(test_df),
            }
        }
        
        with open(self.output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"  Train: {self.output_dir / 'train_paired.csv'}")
        print(f"  Val:   {self.output_dir / 'val_paired.csv'}")
        print(f"  Test:  {self.output_dir / 'test_paired.csv'}")
        print(f"  Metadata: {self.output_dir / 'metadata.pkl'}")
        
        print("\n" + "="*80)
        print("✓ Data preparation complete!")
        print("="*80)
        
        return train_df, val_df, test_df, metadata


if __name__ == '__main__':
    preparator = SimpleGNNDataPreparator()
    preparator.prepare()
