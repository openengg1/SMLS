#!/usr/bin/env python3
"""
Add Dimensionless Numbers to Preprocessing Pipeline

This step enriches the interpolated data with physics-engineered features:
  - Re_d: Reynolds number (inertia vs viscosity)  
  - We_d: Weber number (inertia vs surface tension)
  - Oh: Ohnesorge number (viscosity vs surface tension)
  - Ca: Capillary number
  - Derived features for model input

Located between step4_identify_injection and prepare_gnn_data.

Input:
  preprocess/processed_data/step3_with_injection_labels.csv

Output:
  Updates the same file with added dimensionless number columns
"""

import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from calculate_dimensionless_numbers import add_dimensionless_numbers


def main():
    input_file = Path('../data/step3_with_injection_labels.csv')
    
    if not input_file.exists():
        print(f"✗ Input file not found: {input_file}")
        print("Make sure step1 → step3 → step4 have completed first")
        return 1
    
    print("="*80)
    print("ADDING DIMENSIONLESS NUMBERS TO PREPROCESSED DATA")
    print("="*80)
    
    # Load data
    print(f"\nLoading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Add dimensionless numbers
    print(f"\nBefore: {len(df.columns)} columns")
    df = add_dimensionless_numbers(df)
    print(f"After:  {len(df.columns)} columns (added dimensionless numbers)")
    
    # Save back (overwrite)
    print(f"\nSaving back to: {input_file}")
    df.to_csv(input_file, index=False)
    print(f"✓ Updated file with {len(df):,} rows")
    
    # Show statistics
    print(f"\n" + "="*80)
    print("NEW FEATURE STATISTICS")
    print("="*80)
    
    dimensionless_cols = ['Re_d', 'We_d', 'Oh', 'Ca', 'Dav', 'Oh_squared', 'Re_We_ratio', 'Z_number']
    for col in dimensionless_cols:
        if col in df.columns:
            print(f"{col:20s}: min={df[col].min():9.3f} | max={df[col].max():9.3f} | mean={df[col].mean():9.3f}")
    
    print("\n✓ Dimensionless numbers added successfully")
    print("Ready for GNN data preparation (prepare_gnn_data_simple.py)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
