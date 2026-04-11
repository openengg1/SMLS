#!/usr/bin/env python3
"""
Master Preprocessing Pipeline Orchestrator
===========================================

Runs all preprocessing steps in sequence:
  1. Load Lagrangian particle data
  2. Interpolate Eulerian fields to particle locations
  3. Identify injection events
  4. Add dimensionless numbers (physics-engineered features)
  5. Prepare GNN training data (pairs, normalization, splits)

Usage:
  python preprocess/run_preprocessing_pipeline.py

Output:
  preprocess/processed_data/gnn_training_simple/
    ├── train_paired.csv
    ├── val_paired.csv
    ├── test_paired.csv
    └── metadata.pkl
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logs_dir = Path('logs/preprocessing')
logs_dir.mkdir(exist_ok=True, parents=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = logs_dir / f'preprocessing_run_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_step(step_name: str, script_path: str) -> bool:
    """Run a preprocessing step and return success status."""
    logger.info("=" * 80)
    logger.info(f"RUNNING: {step_name}")
    logger.info("=" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            cwd=str(Path(__file__).parent.parent)
        )
        logger.info(f"✓ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {step_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"✗ {step_name} failed: {e}")
        return False


def main():
    logger.info("=" * 80)
    logger.info("PREPROCESSING PIPELINE - RAW DATA → GNN TRAINING DATA")
    logger.info("=" * 80)
    logger.info(f"Logging to: {log_file}")
    
    # Use self-contained paths relative to this script
    preprocess_dir = Path(__file__).parent.resolve()
    
    # Define pipeline steps (only steps needed for GNN)
    steps = [
        ("Step 1: Load Lagrangian", str(preprocess_dir / 'step1_load_lagrangian.py')),
        ("Step 2: Interpolate Eulerian", str(preprocess_dir / 'step3_interpolate_eulerian.py')),
        ("Step 3: Identify Injection Events", str(preprocess_dir / 'step4_identify_injection.py')),
        ("Step 4: Add Dimensionless Numbers", str(preprocess_dir / 'step5_add_dimensionless_numbers.py')),
        ("Step 5: Prepare GNN Training Data", str(preprocess_dir / 'prepare_gnn_data_simple.py')),
    ]
    
    logger.info(f"\nPipeline has {len(steps)} steps:")
    for i, (name, _) in enumerate(steps, 1):
        logger.info(f"  {i}. {name}")
    
    # Run all steps
    logger.info("\n" + "=" * 80)
    logger.info("STARTING PREPROCESSING")
    logger.info("=" * 80)
    
    results = {}
    for step_name, script_path in steps:
        success = run_step(step_name, script_path)
        results[step_name] = success
        
        if not success:
            logger.error(f"\n✗ Pipeline failed at {step_name}")
            logger.error(f"Check {log_file} for details")
            return 1
        
        logger.info("")
    
    # Summary
    logger.info("=" * 80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 80)
    
    for step_name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"{status} {step_name}")
    
    logger.info("\n" + "=" * 80)
    logger.info("OUTPUT LOCATION")
    logger.info("=" * 80)
    logger.info("GNN training data ready at:")
    logger.info("  preprocess/processed_data/gnn_training_simple/")
    logger.info("    ├── train_paired.csv     (normalized, ~493M)")
    logger.info("    ├── val_paired.csv       (normalized, ~95M)")
    logger.info("    ├── test_paired.csv      (normalized, ~95M)")
    logger.info("    └── metadata.pkl         (scalers + feature mappings)")
    logger.info("\nReady to train GNN!")
    logger.info(f"Full log: {log_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
