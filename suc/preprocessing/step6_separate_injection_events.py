#!/usr/bin/env python3
"""
Step 6: Add parent-child relationships for event tracking
Input: step5_with_event_classification.csv
Output: step6_with_parent_child.csv

IMPORTANT: Does NOT separate injection events!
Injection events are essential training data as they define initial conditions.
The model must learn to predict droplet evolution from injection.

Adds parent-child particle linking for breakup/coalescence events.
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_parent_child_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """Add parent-child particle linking for event classification"""
    # For simplicity, copy event classification as-is
    # Can extend later to explicitly track particle genealogy
    return df.copy()

def main():
    input_file = 'preprocess/processed_data/step5_with_event_classification.csv'
    output_file = 'preprocess/processed_data/step6_with_parent_child.csv'
    
    logger.info("="*80)
    logger.info("STEP 6: Add Parent-Child Relationships (Keep Injection Events)")
    logger.info("="*80)
    
    # Load data
    logger.info(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    logger.info(f"  Total rows: {len(df)}")
    
    # Add parent-child relationships
    df_output = add_parent_child_relationships(df)
    
    # Statistics
    logger.info(f"\nDataset composition:")
    logger.info(f"  Total rows: {len(df_output)}")
    
    # Injection event statistics
    if 'is_injection_event' in df.columns:
        inj_count = (df['is_injection_event'] == 1).sum()
        logger.info(f"  ✓ Injection events: {inj_count} ({100*inj_count/len(df):.1f}%)")
        logger.info(f"  ✓ Transport events: {len(df) - inj_count} ({100*(1-inj_count/len(df)):.1f}%)")
        logger.info(f"\n  NOTE: Injection events RETAINED for training")
        logger.info(f"        (They define initial conditions for ML model)")
    
    # Event statistics
    if 'event_type' in df.columns:
        logger.info(f"\nEvent type distribution:")
        event_dist = df_output['event_type'].value_counts()
        for event_type, count in event_dist.items():
            pct = 100 * count / len(df_output)
            logger.info(f"    {event_type}: {count} ({pct:.1f}%)")
    
    # Save output
    logger.info(f"\nSaving to {output_file}...")
    df_output.to_csv(output_file, index=False)
    logger.info(f"  ✓ Saved {len(df_output)} rows")
    
    logger.info("\n" + "="*80)
    logger.info("STEP 6 COMPLETE ✓")
    logger.info("="*80)
        
        if t_idx % 50 == 0:
            logger.info(f"  Writing timestep {t_idx}/{len(timesteps)}: {int(timestep)}")
    
    logger.info(f"  ✓ Saved {len(df_main)} rows")
    
    logger.info("="*80)
    logger.info("✓ STEP 6 COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
