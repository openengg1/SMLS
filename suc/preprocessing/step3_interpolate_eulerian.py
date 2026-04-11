#!/usr/bin/env python3
"""
Step 3: Interpolate Eulerian fields to Lagrangian particle locations
Input: step1_lagrangian_base.csv, raw Eulerian/*.csv files
Output: preprocess/processed_data/step3_lagrangian_with_eulerian.csv

Processes per timestep with on-demand Eulerian loading (avoids combining huge file)
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_eulerian_at_timestep(euler_dir: str, timestep: int) -> pd.DataFrame:
    """Load Eulerian data for a specific timestep"""
    euler_path = Path(euler_dir) / f'eulerian1_{timestep}.csv'
    if euler_path.exists():
        return pd.read_csv(euler_path)
    else:
        logger.warning(f"  Eulerian file not found for timestep {timestep}")
        return None


def interpolate_eulerian_to_lagrangian(lag_subset: pd.DataFrame, euler_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate Eulerian fields to Lagrangian particle locations for a single timestep
    Uses nearest-neighbor (NN) interpolation via cKDTree:
    - For each droplet at position (x,y,z), finds the nearest Eulerian grid cell
    - Assigns all Eulerian field values from that nearest grid point to the droplet
    
    Eulerian fields interpolated: p, H2O, rho, T, U:0, U:1, U:2 (velocity components)
    """
    
    if euler_df is None or len(euler_df) == 0:
        # Add columns with NaN if no Eulerian data
        for field in ['p', 'H2O', 'rho', 'T', 'U:0', 'U:1', 'U:2']:
            lag_subset[f'euler_{field}'] = np.nan
        return lag_subset
    
    lag_t = lag_subset.copy()
    
    # Extract positions
    # Try to find position columns in Eulerian data
    euler_cols = euler_df.columns.tolist()
    euler_pos_cols = [c for c in euler_cols if 'Points' in c]
    
    if len(euler_pos_cols) < 3:
        # Try alternative coordinate columns
        euler_pos_cols = ['x', 'y', 'z'] if all(c in euler_cols for c in ['x', 'y', 'z']) else []
    
    if len(euler_pos_cols) < 3:
        logger.warning(f"  Could not identify position columns in Eulerian data. Columns: {euler_cols[:10]}")
        for field in ['p', 'H2O', 'rho', 'T', 'U:0', 'U:1', 'U:2']:
            lag_t[f'euler_{field}'] = np.nan
        return lag_t
    
    euler_coords = euler_df[euler_pos_cols[:3]].values
    
    # Lagrangian particle positions
    lag_coords = lag_t[['Points:0', 'Points:1', 'Points:2']].values
    
    # Build KD-tree of Eulerian grid for fast neighbor search
    try:
        kdtree = cKDTree(euler_coords)
        distances, indices = kdtree.query(lag_coords, k=1)
    except Exception as e:
        logger.warning(f"  Failed KD-tree interpolation: {e}")
        for field in ['p', 'H2O', 'rho', 'mu', 'T']:
            lag_t[f'euler_{field}'] = np.nan
        return lag_t
    
    # Get Eulerian field values for nearest cells
    eulerian_fields = ['p', 'H2O', 'rho', 'T', 'U:0', 'U:1', 'U:2']
    
    for field in eulerian_fields:
        if field in euler_df.columns:
            lag_t[f'euler_{field}'] = euler_df.iloc[indices][field].values
        else:
            lag_t[f'euler_{field}'] = np.nan
    
    return lag_t


def main():
    input_file = '../data/step1_lagrangian_base.csv'
    euler_dir = '../raw_data/VTK/eulerian'
    output_file = '../data/step2_with_eulerian_features.csv'
    
    logger.info("="*80)
    logger.info("STEP 3: Interpolate Eulerian Fields to Lagrangian Particles")
    logger.info("(Loading Eulerian data on-demand per timestep)")
    logger.info("="*80)
    
    # Load Lagrangian data
    logger.info(f"Loading {input_file}...")
    lag_df = pd.read_csv(input_file)
    logger.info(f"  Shape: {lag_df.shape}")
    
    # Get timesteps
    lag_timesteps = sorted(lag_df['timestep'].unique())
    logger.info(f"Processing {len(lag_timesteps)} timesteps...")
    
    # Process each timestep separately
    total_rows = 0
    
    for t_idx, lag_t in enumerate(lag_timesteps):
        if t_idx % 20 == 0:
            logger.info(f"  Timestep {t_idx}/{len(lag_timesteps)}: t={int(lag_t)}")
        
        # Get particles at this timestep
        lag_t_data = lag_df[lag_df['timestep'] == lag_t].copy()
        
        if len(lag_t_data) == 0:
            continue
        
        # Load Eulerian data for this timestep
        euler_t_data = load_eulerian_at_timestep(euler_dir, int(lag_t))
        
        # Interpolate
        lag_t_interp = interpolate_eulerian_to_lagrangian(lag_t_data, euler_t_data)
        
        # Save incrementally
        if t_idx == 0:
            lag_t_interp.to_csv(output_file, index=False, mode='w')
        else:
            lag_t_interp.to_csv(output_file, index=False, mode='a', header=False)
        
        total_rows += len(lag_t_interp)
    
    logger.info(f"✓ Saved {total_rows} rows to {output_file}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
