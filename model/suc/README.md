# Supervised Cluster Routing (SUC) - Stochastic ML Spray Model

**A Self-Contained Machine Learning Pipeline for Spray Dynamics Prediction**

This folder contains everything needed to reproduce the results from the ILASS 2026 paper:

> **"A Stochastic Machine Learning Surrogate Model for Comprehensive Replacement of Spray Submodels"**  
> Rohit Mishra, Advaith Sankaranarayanan, Kapil Sachdev  
> ILASS-Americas 36th Annual Conference, May 2026

📄 **Paper:** See `paper/ilass_comprehensive.pdf`  
📜 **LaTeX source:** See `paper/ilass_comprehensive.tex`

---

## Quick Start

```bash
cd model/suc

# Option 1: Run FULL pipeline from raw VTK data (~30 mins)
bash run_full_pipeline_from_raw.sh

# Option 2: Train only (data already processed, ~10 mins)
python3 train_paper_reproduction.py

# Option 3: Generate figures
python3 generate_paper_figures.py
```

---

## Directory Structure

```
model/suc/
│
├── raw_data/                        # ═══ RAW INPUT DATA ═══
│   └── VTK/
│       ├── lagrangian/sprayCloud/   # 202 Lagrangian_*.csv (particle trajectories)
│       └── eulerian/                # 200 eulerian1_*.csv (gas field data)
│
├── preprocessing/                   # ═══ DATA PROCESSING ═══
│   ├── step1_load_lagrangian.py     # Load particle data
│   ├── step3_interpolate_eulerian.py# Interpolate gas fields to particles
│   ├── step4_identify_injection.py  # Mark injection events
│   ├── step5_add_dimensionless_numbers.py  # Compute We, Re, Oh
│   └── prepare_gnn_data_simple.py   # Create paired timestep data
│
├── data/                            # ═══ PROCESSED DATA ═══
│   ├── train_paired.csv             # 1.95M training samples
│   ├── val_paired.csv               # 314K validation samples
│   ├── test_paired.csv              # 238K test samples
│   └── metadata.pkl                 # Feature scalers
│
├── checkpoints/                     # ═══ TRAINED MODELS ═══
│   ├── suc_2clusters_best.pt        # Best model (paper results)
│   └── suc_best_model.pt            # Alternative 4-cluster model
│
├── results/                         # ═══ CLUSTERING ARTIFACTS ═══
│   ├── gmm_*.pkl                    # Fitted GMM models
│   └── scaler_*.pkl                 # Feature scalers
│
├── paper/                           # ═══ PUBLICATION ═══
│   ├── ilass_comprehensive.pdf      # Published paper
│   ├── ilass_comprehensive.tex      # LaTeX source
│   └── expert*_predictions.png      # Result figures
│
├── analysis/                        # Analysis & plotting scripts
├── figures/                         # Generated output plots
├── logs/                            # Training logs
│
└── *.py / *.sh                      # Main scripts (see below)
```

---

## Pipeline Overview

### Data Flow

```
Raw VTK Data (402 files)
        │
        ▼ step1_load_lagrangian.py
Lagrangian Base (particle positions, velocities, properties)
        │
        ▼ step3_interpolate_eulerian.py
+ Eulerian Fields (gas T, U, ρ, p interpolated to particles)
        │
        ▼ step4_identify_injection.py
+ Injection Labels (new particles marked)
        │
        ▼ step5_add_dimensionless_numbers.py
+ Physics Features (Weber, Reynolds, Ohnesorge numbers)
        │
        ▼ prepare_gnn_data_simple.py
Paired Training Data (train/val/test splits, normalized)
        │
        ▼ add_clustering_to_csv.py
+ Cluster Labels (GMM on physics features)
        │
        ▼ train_paper_reproduction.py
Trained SUC Model (2 experts + gating network)
```

### Pipeline Steps

| Step | Script | Output |
|------|--------|--------|
| 1 | `step1_load_lagrangian.py` | `data/step1_lagrangian_base.csv` |
| 2 | `step3_interpolate_eulerian.py` | `data/step2_with_eulerian_features.csv` |
| 3 | `step4_identify_injection.py` | `data/step3_with_injection_labels.csv` |
| 4 | `step5_add_dimensionless_numbers.py` | `data/step4_with_dimensionless.csv` |
| 5 | `prepare_gnn_data_simple.py` | `data/train_paired.csv`, `val_paired.csv`, `test_paired.csv` |
| 6 | `add_clustering_to_csv.py` | Adds `cluster_id` column to CSVs |
| 7 | `train_paper_reproduction.py` | `checkpoints/suc_2clusters_best.pt` |

---

## Model Architecture

**Supervised Cluster Routing (SUC)** - A physics-informed Mixture of Experts:

```
                    ┌─────────────────────────────────────┐
                    │         Input Features (17D)        │
                    │  d, U_xyz, T, n, ρ, μ, σ, Eulerian  │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │      Gating Network (Classifier)     │
                    │         17 → 64 → 32 → 2            │
                    │    CrossEntropyLoss(logits, label)   │
                    └─────────────────┬───────────────────┘
                                      │
                         ┌────────────┴────────────┐
                         │                         │
              ┌──────────▼──────────┐   ┌──────────▼──────────┐
              │     Expert 0        │   │     Expert 1        │
              │   (Large Drops)     │   │   (Small Drops)     │
              │    17→32→32→6       │   │    17→32→32→6       │
              │      86% data       │   │      14% data       │
              └──────────┬──────────┘   └──────────┬──────────┘
                         │                         │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │       Output Predictions (6D)        │
                    │   Δd, ΔU_x, ΔU_y, ΔU_z, ΔT, Δn      │
                    └─────────────────────────────────────┘
```

### Input Features (17D)
| Category | Features |
|----------|----------|
| Lagrangian (6) | d, U_x, U_y, U_z, T, nParticle |
| Material (3) | ρ (density), μ (viscosity), σ (surface tension) |
| Eulerian (7) | T_gas, U_gas (3), H2O, p, ρ_gas |
| Computed (1) | mass_proxy (d³ × nParticle) |

### Output Predictions (6D)
| Feature | Description |
|---------|-------------|
| Δd | Diameter change (evaporation/breakup) |
| ΔU_x, ΔU_y, ΔU_z | Velocity changes (drag, acceleration) |
| ΔT | Temperature change (heat transfer) |
| Δn_particle | Particle count change (breakup/coalescence) |

### Cluster Physics
| Cluster | % Data | Physical Regime |
|---------|--------|-----------------|
| 0 (Large drops) | 86% | Slow evaporation, no breakup, inertia-dominated |
| 1 (Small drops) | 14% | Rapid evaporation, active breakup, surface-tension-dominated |

---

## Expected Results

From `train_paper_reproduction.py`:

### Overall Performance
| Metric | Value |
|--------|-------|
| **Gating Accuracy** | 95-99% |
| **Expert 0 R² (large drops)** | 0.9915 |
| **Expert 1 R² (small drops)** | 0.9818 |
| **Ensemble R²** | 0.9827 |

### Per-Feature R² (Expert 0 - Large Drops)
| Feature | R² | Interpretation |
|---------|-----|----------------|
| Δd | 0.9995 | Perfectly learned |
| ΔU_x | 0.9972 | Near-perfect |
| ΔU_y | 0.9987 | Near-perfect |
| ΔU_z | 0.9974 | Near-perfect |
| ΔT | 0.9563 | Well-captured |
| Δn | 1.0000 | Perfect |

### Per-Feature R² (Expert 1 - Small Drops)
| Feature | R² | Interpretation |
|---------|-----|----------------|
| Δd | 0.9955 | Near-perfect |
| ΔU_x | 0.9940 | Near-perfect |
| ΔU_y | 0.9928 | Near-perfect |
| ΔU_z | 0.9941 | Near-perfect |
| ΔT | 0.9546 | Well-captured |
| Δn | 0.9599 | Well-captured |

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `run_full_pipeline_from_raw.sh` | **Main script** - runs entire pipeline |
| `train_paper_reproduction.py` | 2-cluster training (reproduces paper) |
| `train_supervised_cluster_routing.py` | 4-cluster training (alternative) |
| `add_clustering_to_csv.py` | GMM clustering preprocessing |
| `hybrid_ruc_supervised.py` | SUC model class definition |
| `feature_engineering.py` | Feature extraction utilities |
| `generate_paper_figures.py` | Generate publication-ready figures |

---

## Requirements

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn tqdm
```

- Python 3.8+
- PyTorch ≥1.9
- pandas, numpy, scikit-learn
- matplotlib, seaborn (for plotting)
- tqdm (for progress bars)

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{mishra2026stochastic,
  title={A Stochastic Machine Learning Surrogate Model for Comprehensive 
         Replacement of Spray Submodels},
  author={Mishra, Rohit and Sankaranarayanan, Advaith and Sachdev, Kapil},
  booktitle={ILASS-Americas 36th Annual Conference on Liquid Atomization 
             and Spray Systems},
  year={2026},
  address={May 11-14, 2026}
}
```

---

## Contact

- **Rohit Mishra** - rmishra@tamu.edu  
  Department of Mechanical Engineering, Texas A&M University

---

## License

This code is provided for research purposes. Please contact the authors for commercial use.
