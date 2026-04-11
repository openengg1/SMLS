"""
SU-C Preprocessing Module
=========================

Supervised Unsupervised Clustering preprocessing pipeline.

Converts raw spray parcel data into training-ready datasets with:
- Clean paired t→t+1 transitions (no synthetic samples)
- Proper normalization (train scaler only)
- Physics-informed 3-cluster assignments

Example:
    from preprocessing.prepare_gnn_data_clean import CleanGNNDataPreparator
    from preprocessing.create_physics_clusters import PhysicsFeatureClusterer
    
    # Phase 1: Prepare data
    prep = CleanGNNDataPreparator(input_csv='path/to/data.csv')
    train, val, test, metadata = prep.prepare()
    
    # Phase 2: Add clusters
    cluster = PhysicsFeatureClusterer()
    train, val, test = cluster.cluster()

Or use the master script:
    python run_preprocessing_pipeline.py
"""

__version__ = '1.0.0'
__author__ = 'Preprocessing Pipeline'

from .prepare_gnn_data_clean import CleanGNNDataPreparator
from .create_physics_clusters import PhysicsFeatureClusterer

__all__ = [
    'CleanGNNDataPreparator',
    'PhysicsFeatureClusterer',
]
