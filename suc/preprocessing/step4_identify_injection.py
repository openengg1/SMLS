#!/usr/bin/env python3
"""
Step 4: Identify injection events
Input: step2_with_eulerian_features.csv
Output: preprocess/processed_data/step3_with_injection_labels.csv

An injection event: origId appears for the first time AND is close to injection location
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def identify_injection_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify injection events: All particles at their first appearance (first_timestep).
    
    KEY INSIGHT: In the spray simulation, every new particle that appears for the first time
    IS an injection event - the simulation doesn't create particles anywhere else.
    
    Note: You will observe that all new origIDs naturally occur only in timesteps 1-25
    (active injection period), but we keep the logic general without hardcoded timestep
    restrictions. The data itself enforces this constraint.
    
    A particle is marked as injection if: current_timestep == first_timestep
    
    Args:
        df: Lagrangian data with all fields
    """
    
    df = df.copy()
    
    # Get first appearance timestep for each origId
    first_timesteps = df.groupby('origId')['timestep'].min().reset_index()
    first_timesteps.rename(columns={'timestep': 'first_timestep'}, inplace=True)
    
    # Merge back to get first_timestep for every row
    df = df.merge(first_timesteps, on='origId', how='left')
    
    # Mark injection: is_injection_event = 1 if at first appearance, else 0
    df['is_injection_event'] = (df['timestep'] == df['first_timestep']).astype(int)
    
    return df


def main():
    input_file = '../data/step2_with_eulerian_features.csv'
    output_file = '../data/step3_with_injection_labels.csv'
    
    logger.info("="*80)
    logger.info("STEP 4: Identify Injection Events")
    logger.info("="*80)
    
    # Load data
    logger.info(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {len(df.columns)}")
    
    # Identify injection
    logger.info("Identifying injection events...")
    df = identify_injection_events(df)
    
    injection_count = df['is_injection_event'].sum()
    logger.info(f"  Found {injection_count} injection events")
    
    # Save
    logger.info(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    logger.info(f"✓ Saved {len(df)} rows")
    
    logger.info("="*80)


if __name__ == '__main__':
    main()
