#!/bin/bash
# Complete SUC Workflow - From Raw Data to Trained Model
# 
# This script regenerates the entire pipeline from scratch:
#   1. Load Lagrangian particle data
#   2. Interpolate Eulerian fields to particles
#   3. Identify injection events
#   4. Add dimensionless numbers  
#   5. Prepare paired data for training
#   6. Add cluster assignments
#   7. Train the SUC model
#
# Usage:
#   cd model/suc
#   bash run_full_pipeline_from_raw.sh

set -e

echo "========================================================================"
echo "STOCHASTIC ML SPRAY - COMPLETE PIPELINE (Raw Data → Trained Model)"
echo "========================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Working directory: $SCRIPT_DIR"
echo ""

# Verify we have raw data
if [ ! -d "$SCRIPT_DIR/raw_data/VTK/lagrangian/sprayCloud" ]; then
    echo "❌ Raw data not found at: $SCRIPT_DIR/raw_data/VTK/lagrangian/sprayCloud"
    echo "   Expected Lagrangian_*.csv files"
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/raw_data/VTK/eulerian" ]; then
    echo "❌ Eulerian data not found at: $SCRIPT_DIR/raw_data/VTK/eulerian"
    exit 1
fi

echo "✓ Raw data verified"
echo "  • Lagrangian: $(ls -1 $SCRIPT_DIR/raw_data/VTK/lagrangian/sprayCloud/Lagrangian_*.csv 2>/dev/null | wc -l) files"
echo "  • Eulerian: $(ls -1 $SCRIPT_DIR/raw_data/VTK/eulerian/ 2>/dev/null | wc -l) files"
echo ""

# Create output directory
mkdir -p "$SCRIPT_DIR/data"

echo "========================================================================"
echo "STEP 1: Load Lagrangian Particle Data"
echo "========================================================================"
cd "$SCRIPT_DIR/preprocessing"
python step1_load_lagrangian.py
echo ""

echo "========================================================================"
echo "STEP 2: Interpolate Eulerian Fields to Particles"
echo "========================================================================"
python step3_interpolate_eulerian.py
echo ""

echo "========================================================================"
echo "STEP 3: Identify Injection Events"
echo "========================================================================"
python step4_identify_injection.py
echo ""

echo "========================================================================"
echo "STEP 4: Add Dimensionless Numbers (We, Re, Oh)"
echo "========================================================================"
python step5_add_dimensionless_numbers.py
echo ""

echo "========================================================================"
echo "STEP 5: Prepare GNN Training Data (normalized, paired timesteps)"
echo "========================================================================"
python prepare_gnn_data_simple.py
echo ""

echo "========================================================================"
echo "STEP 6: Add Cluster Assignments to Training Data"
echo "========================================================================"
cd "$SCRIPT_DIR"
python add_clustering_to_csv.py
echo ""

echo "========================================================================"
echo "STEP 7: Train SUC Model (Supervised Cluster Routing)"
echo "========================================================================"
python train_supervised_cluster_routing.py
echo ""

echo "========================================================================"
echo "PIPELINE COMPLETE!"
echo "========================================================================"
echo ""
echo "✓ Trained model saved to:"
echo "  $SCRIPT_DIR/checkpoints/suc_best_model.pt"
echo ""
echo "✓ Training data saved to:"
echo "  $SCRIPT_DIR/data/train_paired.csv"
echo "  $SCRIPT_DIR/data/val_paired.csv"
echo "  $SCRIPT_DIR/data/test_paired.csv"
echo ""
echo "✓ Results saved to:"
echo "  $SCRIPT_DIR/results/scaler_*.pkl"
echo "  $SCRIPT_DIR/results/gmm_*.pkl"
echo ""

exit 0
