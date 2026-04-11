#!/usr/bin/env python3
"""
Step 1: Load all Lagrangian files and create base dataset
Output: preprocess/processed_data/step1_lagrangian_base.csv
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_all_lagrangian_data(lag_dir: str) -> pd.DataFrame:
    """Load all Lagrangian CSV files and combine into single dataframe"""
    
    lag_path = Path(lag_dir)
    lag_files = sorted([f for f in lag_path.glob('Lagrangian_*.csv') if f.name != 'Lagrangian_0.csv'])
    
    logger.info(f"Found {len(lag_files)} Lagrangian files")
    
    all_data = []
    
    for i, lag_file in enumerate(lag_files):
        if i % 20 == 0:
            logger.info(f"  Loading {i}/{len(lag_files)}: {lag_file.name}")
        
        df = pd.read_csv(lag_file)
        
        if len(df) > 0:
            # Keep only essential columns
            cols_to_keep = [
                'TimeStep', 'origId', 'Points:0', 'Points:1', 'Points:2',
                'U:0', 'U:1', 'U:2', 'd', 'T', 'rho', 'mu', 'sigma',
                'nParticle', 'age', 'mass0'
            ]
            
            # Filter to available columns
            available_cols = [c for c in cols_to_keep if c in df.columns]
            df_subset = df[available_cols].copy()
            
            all_data.append(df_subset)
    
    # Combine all data
    logger.info(f"Combining {len(all_data)} timesteps...")
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # Rename TimeStep to timestep for consistency
    if 'TimeStep' in df_combined.columns:
        df_combined.rename(columns={'TimeStep': 'timestep'}, inplace=True)
    
    logger.info(f"Combined shape: {df_combined.shape}")
    logger.info(f"Timesteps: {df_combined['timestep'].min()} to {df_combined['timestep'].max()}")
    logger.info(f"Total unique origIds: {df_combined['origId'].nunique()}")
    
    return df_combined


def main():
    lag_dir = '../raw_data/VTK/lagrangian/sprayCloud'
    output_file = '../data/step1_lagrangian_base.csv'
    
    logger.info("="*80)
    logger.info("STEP 1: Load Lagrangian Base Data")
    logger.info("="*80)
    
    # Load data
    df = load_all_lagrangian_data(lag_dir)
    
    # Save
    logger.info(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    logger.info(f"✓ Saved {len(df)} rows")
    
    logger.info("="*80)


if __name__ == '__main__':
    main()
