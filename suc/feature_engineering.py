"""
Physics-Informed Feature Engineering for Spray Droplet Prediction
===================================================================

Extracts 11D clustering features based on physics:
- We: Weber number (inertia vs surface tension)
- Oh: Ohnesorge number (viscosity vs surface tension)  
- Re: Reynolds number (inertia vs viscosity)
- Urel_mag: Relative velocity magnitude (drop - Eulerian)
- mass_proxy: Proxy for droplet mass (density × volume)
- delT_T_boil: Temperature change normalized by boiling temp
- del_nParticle: Change in particle count
- delUx: Change in U:0 component
- delUy: Change in U:1 component
- delUz: Change in U:2 component
- deld: Change in diameter

Plus original 17D (17D + 11D = 28D total) for gating network
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Physical constants
RHO_AIR = 1.2  # kg/m³ at ~25°C
SIGMA = 0.072  # Surface tension N/m for water at 20°C
T_BOIL = 373.15  # Boiling temperature K


def calculate_urel_mag(data):
    """Calculate relative velocity magnitude between drop and Eulerian field."""
    u_drop = data[['in_U:0', 'in_U:1', 'in_U:2']].values
    u_euler = data[['in_euler_U:0', 'in_euler_U:1', 'in_euler_U:2']].values
    u_rel = u_drop - u_euler
    urel_mag = np.linalg.norm(u_rel, axis=1)
    return urel_mag


def calculate_weber(data):
    """
    Weber number: We = ρ_air * U_rel² * d / σ
    Ratio of inertial forces to surface tension forces
    """
    urel_mag = calculate_urel_mag(data)
    d = data['in_d'].values
    we = RHO_AIR * (urel_mag ** 2) * d / SIGMA
    return we


def calculate_ohnesorge(data):
    """
    Ohnesorge number: Oh = μ / sqrt(ρ_liquid * σ * d)
    Ratio of viscous forces to surface tension forces
    """
    mu = data['in_mu'].values  # Dynamic viscosity of liquid
    rho_liquid = 1000  # Water density kg/m³
    d = np.maximum(data['in_d'].values, 1e-6)  # Clamp to avoid sqrt of tiny values
    oh = mu / np.sqrt(rho_liquid * SIGMA * d)
    oh = np.nan_to_num(oh, nan=0, posinf=1e6, neginf=0)  # Handle any remaining NaNs
    return oh


def calculate_reynolds(data):
    """
    Reynolds number: Re = ρ_air * U_rel * d / μ_air
    Ratio of inertial forces to viscous forces
    """
    urel_mag = calculate_urel_mag(data)
    d = data['in_d'].values
    mu_air = 1.81e-5  # Dynamic viscosity of air at 15°C Pa·s
    re = RHO_AIR * urel_mag * d / mu_air
    return re


def calculate_deltT_T_boil(data):
    """Temperature difference normalized by boiling point."""
    delta_t = data['out_T'] - data['in_T']
    dt_normalized = delta_t / T_BOIL
    return dt_normalized


def calculate_del_nparticle(data):
    """Change in number of particles (coalescence/breakup indicator)."""
    del_np = data['out_nParticle'] - data['in_nParticle']
    return del_np


def calculate_del_ux(data):
    """Change in U:0 (x-velocity component)."""
    del_ux = data['out_U:0'] - data['in_U:0']
    return del_ux


def calculate_del_uy(data):
    """Change in U:1 (y-velocity component)."""
    del_uy = data['out_U:1'] - data['in_U:1']
    return del_uy


def calculate_del_uz(data):
    """Change in U:2 (z-velocity component)."""
    del_uz = data['out_U:2'] - data['in_U:2']
    return del_uz


def calculate_del_d(data):
    """Change in diameter."""
    del_d = data['out_d'] - data['in_d']
    return del_d


def extract_clustering_features(data, verbose=True):
    """
    Extract 11D clustering features from raw data.
    
    Args:
        data: Input dataframe
        verbose: If False, suppress print statements
    
    Returns:
        features_11d: [N, 11] normalized clustering features
        input_28d: [N, 28] full features for gating (input 17D + normalized clustering 11D)
        scaler_11d: StandardScaler fitted on clustering features
    """
    if verbose:
        print("\n" + "="*70)
        print("EXTRACTING PHYSICS-INFORMED FEATURES")
        print("="*70)
    
    # Calculate all features
    if verbose:
        print("Calculating physics features...")
    we = calculate_weber(data)
    oh = calculate_ohnesorge(data)
    re = calculate_reynolds(data)
    urel_mag = calculate_urel_mag(data)
    mass_proxy = data['in_mass_proxy'].values
    del_t_t_boil = calculate_deltT_T_boil(data)
    del_nparticle = calculate_del_nparticle(data)
    del_ux = calculate_del_ux(data)
    del_uy = calculate_del_uy(data)
    del_uz = calculate_del_uz(data)
    del_d = calculate_del_d(data)
    
    if verbose:
        print("  ✓ Weber number calculated")
        print("  ✓ Ohnesorge number calculated")
        print("  ✓ Reynolds number calculated")
        print("  ✓ Urel_mag calculated")
        print("  ✓ Mass proxy extracted")
        print("  ✓ ΔT/T_boil calculated")
        print("  ✓ ΔnParticle calculated")
        print("  ✓ ΔUx calculated")
        print("  ✓ ΔUy calculated")
        print("  ✓ ΔUz calculated")
        print("  ✓ Δd calculated")
    
    # Stack into 11D feature matrix
    features_11d_raw = np.column_stack([
        we,
        oh,
        re,
        urel_mag,
        mass_proxy,
        del_t_t_boil,
        del_nparticle,
        del_ux,
        del_uy,
        del_uz,
        del_d
    ])
    
    # Normalize 11D features for clustering
    scaler_11d = StandardScaler()
    features_11d_norm = scaler_11d.fit_transform(features_11d_raw)
    
    if verbose:
        print("\n11D Clustering Features (normalized):")
        print(f"  Shape: {features_11d_norm.shape}")
        print(f"  Mean: {features_11d_norm.mean(axis=0)}")
        print(f"  Std: {features_11d_norm.std(axis=0)}")
    
    # Get original 17D input features (already normalized in paired file)
    input_cols = [col for col in data.columns if col.startswith('in_')]
    input_17d = data[input_cols].values
    
    # Extract We, Oh, Re for gating features
    we = calculate_weber(data)
    oh = calculate_ohnesorge(data)
    re = calculate_reynolds(data)
    
    # Create 20D gating features (17D input + We, Oh, Re)
    physics_3d = np.column_stack([we, oh, re])
    input_20d = np.column_stack([input_17d, physics_3d])
    
    if verbose:
        print(f"\n20D Gating Features (17D input + We, Oh, Re):")
        print(f"  Shape: {input_20d.shape}")
    
    return features_11d_norm, input_20d, scaler_11d


def create_clustering_data(train_data, val_data, test_data, verbose=True):
    """
    Process all splits with feature extraction.
    
    Args:
        train_data, val_data, test_data: Input dataframes
        verbose: If False, suppress print statements
    
    Returns clustering-ready data with non-persistence annotation.
    """
    if verbose:
        print("\nProcessing train split...")
    train_features_11d, train_features_28d, scaler_11d = extract_clustering_features(train_data, verbose=verbose)
    train_persistence = train_data['out_persists'].values.astype(int)
    
    # Extract 17D raw inputs for gating (first 17 columns of 28D)
    train_input_cols = [col for col in train_data.columns if col.startswith('in_')]
    train_features_17d = train_data[train_input_cols].values
    
    # Compute We, Oh, Re for gating (20D = 17D + 3D dimensionless)
    train_we = calculate_weber(train_data)
    train_oh = calculate_ohnesorge(train_data)
    train_re = calculate_reynolds(train_data)
    train_features_20d_raw = np.column_stack([train_features_17d, train_we, train_oh, train_re])
    
    # Normalize 20D features using StandardScaler
    scaler_20d = StandardScaler()
    train_features_20d = scaler_20d.fit_transform(train_features_20d_raw)
    
    if verbose:
        print("\nProcessing val split...")
    val_features_11d_raw = np.column_stack([
        calculate_weber(val_data),
        calculate_ohnesorge(val_data),
        calculate_reynolds(val_data),
        calculate_urel_mag(val_data),
        val_data['in_mass_proxy'].values,
        calculate_deltT_T_boil(val_data),
        calculate_del_nparticle(val_data),
        calculate_del_ux(val_data),
        calculate_del_uy(val_data),
        calculate_del_uz(val_data),
        calculate_del_d(val_data)
    ])
    val_features_11d = scaler_11d.transform(val_features_11d_raw)
    val_input_cols = [col for col in val_data.columns if col.startswith('in_')]
    val_features_28d = np.column_stack([val_data[val_input_cols].values, val_features_11d])
    val_persistence = val_data['out_persists'].values.astype(int)
    
    # Extract 20D gating features for val
    val_features_17d = val_data[val_input_cols].values
    val_we = calculate_weber(val_data)
    val_oh = calculate_ohnesorge(val_data)
    val_re = calculate_reynolds(val_data)
    val_features_20d_raw = np.column_stack([val_features_17d, val_we, val_oh, val_re])
    val_features_20d = scaler_20d.transform(val_features_20d_raw)
    
    if verbose:
        print("\nProcessing test split...")
    test_features_11d_raw = np.column_stack([
        calculate_weber(test_data),
        calculate_ohnesorge(test_data),
        calculate_reynolds(test_data),
        calculate_urel_mag(test_data),
        test_data['in_mass_proxy'].values,
        calculate_deltT_T_boil(test_data),
        calculate_del_nparticle(test_data),
        calculate_del_ux(test_data),
        calculate_del_uy(test_data),
        calculate_del_uz(test_data),
        calculate_del_d(test_data)
    ])
    test_features_11d = scaler_11d.transform(test_features_11d_raw)
    test_input_cols = [col for col in test_data.columns if col.startswith('in_')]
    test_features_28d = np.column_stack([test_data[test_input_cols].values, test_features_11d])
    test_persistence = test_data['out_persists'].values.astype(int)
    
    # Extract 20D gating features for test
    test_features_17d = test_data[test_input_cols].values
    test_we = calculate_weber(test_data)
    test_oh = calculate_ohnesorge(test_data)
    test_re = calculate_reynolds(test_data)
    test_features_20d_raw = np.column_stack([test_features_17d, test_we, test_oh, test_re])
    test_features_20d = scaler_20d.transform(test_features_20d_raw)
    
    return {
        'train': {
            'features_11d': train_features_11d,
            'features_28d': train_features_28d,
            'features_20d': train_features_20d,
            'persistence': train_persistence,
            'data': train_data
        },
        'val': {
            'features_11d': val_features_11d,
            'features_28d': val_features_28d,
            'features_20d': val_features_20d,
            'persistence': val_persistence,
            'data': val_data
        },
        'test': {
            'features_11d': test_features_11d,
            'features_28d': test_features_28d,
            'features_20d': test_features_20d,
            'persistence': test_persistence,
            'data': test_data
        },
        'scaler_11d': scaler_11d,
        'scaler_20d': scaler_20d
    }


if __name__ == '__main__':
    # Load data
    data_dir = Path('preprocess/processed_data/gnn_training_simple')
    train_data = pd.read_csv(data_dir / 'train_paired.csv').dropna()
    val_data = pd.read_csv(data_dir / 'val_paired.csv').dropna()
    test_data = pd.read_csv(data_dir / 'test_paired.csv').dropna()
    
    # Extract features
    processed = create_clustering_data(train_data, val_data, test_data)
    
    print("\n" + "="*70)
    print("Non-persistence distribution:")
    print(f"  Train: {(processed['train']['persistence']==0).sum()} / {len(processed['train']['persistence'])} ({100*(processed['train']['persistence']==0).sum()/len(processed['train']['persistence']):.2f}%)")
    print(f"  Val:   {(processed['val']['persistence']==0).sum()} / {len(processed['val']['persistence'])} ({100*(processed['val']['persistence']==0).sum()/len(processed['val']['persistence']):.2f}%)")
    print(f"  Test:  {(processed['test']['persistence']==0).sum()} / {len(processed['test']['persistence'])} ({100*(processed['test']['persistence']==0).sum()/len(processed['test']['persistence']):.2f}%)")
    print("="*70)
