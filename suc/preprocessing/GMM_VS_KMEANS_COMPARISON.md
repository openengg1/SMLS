# GMM vs KMeans: Clustering Comparison

## ✓ Both Clustering Methods Complete

You now have **two clustering approaches** with different properties:

---

## Comparison Table

| Feature | **KMeans** (Fast) | **GMM - Subsampled** (Probabilistic) |
|---------|---|---|
| **Fitting time** | ~5 min | ~10 min (on 500k subsample) |
| **Prediction time** | Instant | Instant |
| **Outputs** | Hard cluster IDs | Hard IDs + soft probabilities |
| **Cluster type** | Hard assignment | Soft (probabilistic) |
| **Model size** | 7.5 MB | 5.6 KB |
| **Interpretability** | "This is cluster X" | "90% cluster 0, 10% cluster 1" |
| **Best for** | Speed, simplicity | Uncertainty quantification |

---

## Cluster Distribution

### **KMeans** (hard assignment)
```
Cluster 0:  1,883,298 samples (97.1%)
Cluster 1:     45,898 samples (2.4%)
Cluster 2:     10,536 samples (0.5%)
```

**Issue**: Heavily imbalanced - almost everything is in cluster 0.

### **GMM Subsampled** (probabilistic)
```
Cluster 0:    608,994 samples (31.4%)
Cluster 1:  1,258,067 samples (64.9%)
Cluster 2:     72,671 samples (3.7%)
```

**Better**: More balanced distribution, better separation of physics regimes.

---

## GMM Advantages

✓ **Soft cluster assignments**:
- Each sample has probability for each cluster (sums to 1)
- Column added: `cluster_0_prob`, `cluster_1_prob`, `cluster_2_prob`
- Example: sample might be 85% Cluster 1, 15% Cluster 2

✓ **Probabilistic interpretation**:
- Can use membership probabilities as weights in training
- Better for borderline cases (mixed physics)

✓ **Better fit**:
- Model selection criteria: AIC = -17,452,013, BIC = -17,450,823
- Captures complex multimodal distribution

---

## Output Files Created

### GMM Datasets (with probabilities)
```
data/train_paired_gmm.csv          (1.5 GB, 1.9M samples)
data/val_paired_gmm.csv            (187 MB, 241K samples)
data/test_paired_gmm.csv           (187 MB, 242K samples)
```

**Columns added**:
- `cluster_id_physics` — Hard cluster (0, 1, or 2)
- `cluster_0_prob` — Probability for cluster 0
- `cluster_1_prob` — Probability for cluster 1
- `cluster_2_prob` — Probability for cluster 2
- All 5 physics features still included

### Comparison: GMM vs KMeans

| File | Type | Size |
|------|------|------|
| `train_paired_gmm.csv` | GMM (with probabilities) | 1.5 GB |
| `train_paired_clustered.csv` | KMeans (hard only) | 1.4 GB |
| `gmm_clustering_model_subsampled.pkl` | GMM model | 5.6 KB |
| `kmeans_clustering_model.pkl` | KMeans model | 7.5 MB |

---

## How GMM Subsampling Works

### 1. **Fit on 500k Random Subsample**
   - 500k samples randomly selected from 1.9M training data
   - Ensures statistical representativeness
   - Full GMM fitting completed in ~10 min

### 2. **Predict on Full 2.4M Dataset**
   - GMM model applied to all train/val/test samples
   - Prediction is fast (scales linearly)
   - Preserves probabilistic nature of GMM

### 3. **Soft Assignments Generated**
   - Each sample gets probability distribution over 3 clusters
   - Example: `[0.15, 0.75, 0.10]` = 75% likely Cluster 1

---

## Recommended Usage

### Use **GMM** when:
- You want **probabilistic interpretation** ("this is borderline")
- You need **cluster probabilities** for weighted training
- You want **uncertainty quantification** for physics
- Imbalanced hard clusters are a concern

### Use **KMeans** when:
- You want **fast, simple** cluster assignments
- Hard cluster IDs are sufficient for your analysis
- You need minimal model size
- Computational speed is critical

---

## Next Steps

### Option 1: Use GMM clusters
```bash
# Train model with soft cluster weights
python train_model.py --data data/train_paired_gmm.csv --use-cluster-probs
```

### Option 2: Use KMeans clusters
```bash
# Train model with hard cluster balancing
python train_model.py --data data/train_paired_clustered.csv
```

### Option 3: Compare both
Train both models and evaluate which clustering better captures spray physics.

---

## Summary

| Metric | Value |
|--------|-------|
| **Total samples** | 2,423,037 |
| **Clustering methods** | 2 (KMeans + GMM) |
| **Cluster assignments** | Hard + Probabilistic |
| **Physics features** | 7 (Re, We, ΔT, Δd, ΔnParticle, ...) |
| **Model ready** | ✓ Yes |

**Data is now ready for supervised model training with either clustering approach.**

---

**Generated**: 2026-03-23 15:30  
**Status**: ✓ Complete and ready for training
