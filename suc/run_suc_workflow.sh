#!/bin/bash
# SUC (Supervised Cluster Routing) - Self-Contained Workflow
# Run this script from the root directory or from model/suc/
# 
# Usage:
#   cd model/suc && bash run_suc_workflow.sh
#   OR
#   python3 model/suc/train_supervised_cluster_routing.py

set -e

echo "========================================================================"
echo "SUC (Supervised Cluster Routing) Workflow"
echo "========================================================================"

# Determine where we are running from
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WORK_DIR="$SCRIPT_DIR"

echo ""
echo "Working directory: $WORK_DIR"
echo ""

# Check if data files exist
if [ ! -f "$WORK_DIR/data/train_paired.csv" ]; then
    echo "❌ Data files not found in $WORK_DIR/data/"
    echo "   Please ensure train_paired.csv, val_paired.csv, test_paired.csv exist"
    exit 1
fi

echo "✓ Data files found"
echo ""

# Step 1: Add clustering to CSV
echo "========================================================================"
echo "STEP 1: Add Cluster Labels to CSVs (preprocessing)"
echo "========================================================================"
cd "$WORK_DIR"
python3 add_clustering_to_csv.py

echo ""

# Step 2: Train supervised model
echo "========================================================================"
echo "STEP 2: Train Supervised Cluster Routing Model"
echo "========================================================================"
cd "$WORK_DIR"
python3 train_supervised_cluster_routing.py

echo ""
echo "========================================================================"
echo "✓ SUC Workflow Complete"
echo "========================================================================"
