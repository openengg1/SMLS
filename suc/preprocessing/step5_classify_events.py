#!/usr/bin/env python3
"""
Step 5: Classify events (Breakup, Coalescence, Evaporation)  
INPUT: step3_with_injection_labels.csv
OUTPUT: step4_with_event_classification.csv

KEY: Non-injected parcels = breakup fragments
- Mark at first appearance timestep as 'breakup' events
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    input_file = 'preprocess/processed_data/step3_with_injection_labels.csv'
    output_file = 'preprocess/processed_data/step4_with_event_classification.csv'
    
    logger.info("="*80)
    logger.info("STEP 5: Classify Events (Breakup, Coalescence, Evaporation)")
    logger.info("="*80)
    
    # Load data
    logger.info(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    logger.info(f"  Shape: {df.shape}")
    
    # Get injected parcel origIds
    inj_parcels_set = set(df[df['is_injection_event'] == 1]['origId'].unique())
    logger.info(f"  Injected parcels: {len(inj_parcels_set)}")
    
    # =====================================================================
    # Build event_type mapping and parent-child relationships
    # =====================================================================
    logger.info("Building event classification mapping and parent-child tracking...")
    
    # Start with all 'normal_motion'
    event_map = {}  # origId -> event_type
    parent_map = {}  # origId -> parent_origId
    
    # Step 1: Mark non-injected parcels as 'breakup' (created by breakup)
    non_inj_parcels = df[~df['origId'].isin(inj_parcels_set)]['origId'].unique()
    for orig_id in non_inj_parcels:
        event_map[orig_id] = 'breakup'
        parent_map[orig_id] = -1  # Will be assigned when we find parent
    logger.info(f"  Marked {len(non_inj_parcels):,} non-injected parcels as 'breakup'")
    
    # Step 2: Identify disappearing parcels and match with newly appearing children
    df = df.sort_values(['timestep', 'origId']).reset_index(drop=True)
    
    # Create a dict for fast parent lookups: {timestep: {origId: row_data}}
    df_by_t = {}
    for t in df['timestep'].unique():
        df_t = df[df['timestep'] == t]
        df_by_t[t] = {row['origId']: row for _, row in df_t.iterrows()}
    
    timesteps = sorted(df['timestep'].unique())
    
    logger.info(f"Processing {len(timesteps)-1} timestep transitions for parent-child linking and evaporation...")
    
    for idx in range(len(timesteps) - 1):
        t_curr = timesteps[idx]
        t_next = timesteps[idx + 1]
        
        if idx % 20 == 0:
            logger.info(f"  Pair {idx}/{len(timesteps)-1}: t={int(t_curr)} -> t={int(t_next)}")
        
        df_t = df[df['timestep'] == t_curr]
        df_t_next = df[df['timestep'] == t_next]
        
        ids_t = set(df_t['origId'].values)
        ids_t_next = set(df_t_next['origId'].values)
        
        # Disappeared parcels (parents that broke up or evaporated)
        disappeared = ids_t - ids_t_next
        
        # Newly appearing parcels (potential children)
        newly_appeared = ids_t_next - ids_t
        
        if len(disappeared) == 0 or len(df_t_next) == 0:
            continue
        
        # Build KD-tree of next timestep (all particles)
        try:
            coords_t_next = df_t_next[['Points:0', 'Points:1', 'Points:2']].values
            kdtree = cKDTree(coords_t_next)
        except:
            kdtree = None
        
        # Classify each disappeared parcel and link to children
        for orig_id in disappeared:
            # Fast lookup from dict instead of DataFrame query
            parent = df_by_t[t_curr].get(orig_id)
            if parent is None:
                continue
            
            if pd.isna(parent['d']) or parent['d'] <= 0:
                continue
            
            parent_pos = np.array([parent['Points:0'], parent['Points:1'], parent['Points:2']])
            parent_d = parent['d']
            parent_vol = (np.pi / 6) * parent_d**3
            search_radius = 20.0 * parent_d
            
            event = 'evaporation'  # Default
            nearby_children = []
            
            if kdtree is not None:
                try:
                    nearby_indices = kdtree.query_ball_point(parent_pos, search_radius)
                    if len(nearby_indices) > 0:
                        nearby_particles = df_t_next.iloc[nearby_indices]
                        nearby_vols = (np.pi / 6) * nearby_particles['d'].values**3
                        total_vol = np.sum(nearby_vols)
                        vol_ratio = total_vol / parent_vol if parent_vol > 0 else 0
                        
                        # Check for breakup: newly appeared children with volume conservation
                        nearby_newly_appeared = nearby_particles[nearby_particles['origId'].isin(newly_appeared)]
                        if len(nearby_newly_appeared) > 0:
                            newly_vols = (np.pi / 6) * nearby_newly_appeared['d'].values**3
                            newly_vol = np.sum(newly_vols)
                            newly_ratio = newly_vol / parent_vol if parent_vol > 0 else 0
                            
                            # Breakup signature: new children with volume conservation
                            if 0.1 < newly_ratio < 3.0:  # Allow wide range for atomization losses
                                event = 'breakup'
                                nearby_children = nearby_newly_appeared['origId'].tolist()
                        
                        # Check for coalescence (if not breakup)
                        if event != 'breakup' and 0.5 < vol_ratio < 2.5:
                            event = 'coalescence'
                except:
                    pass
            
            event_map[orig_id] = event
            
            # Link children to parent
            for child_id in nearby_children:
                parent_map[child_id] = int(orig_id)
    
    # =====================================================================
    # Apply event_type and parent_origId mapping to dataframe
    # =====================================================================
    logger.info("Applying event classification and parent-child links to dataframe...")
    df['event_type'] = df['origId'].map(event_map).fillna('normal_motion')
    df['parent_origId'] = df['origId'].map(parent_map).fillna(-1).astype(int)
    
    # Save output
    logger.info(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    logger.info(f"✓ Saved {len(df):,} rows")
    
    # Print parent-child statistics
    linked_children = (df['parent_origId'] > 0).sum()
    unique_parent_child_pairs = df[df['parent_origId'] > 0][['parent_origId', 'origId']].drop_duplicates()
    logger.info(f"\nParent-child linking statistics:")
    logger.info(f"  Rows with parent link: {linked_children:,}")
    logger.info(f"  Unique parent-child pairs: {len(unique_parent_child_pairs):,}")
    
    logger.info("="*80)


if __name__ == '__main__':
    main()
