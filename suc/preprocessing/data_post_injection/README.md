# Post-Injection Phase Sample Dataset (t=50-59)

## Physical Context

**Time Range**: Timesteps 50-59 (50-59 ms, well after injection ended)  
**Phase**: Post-injection spray evolution - stable dispersion  
**Physics**: Injection has stopped (~t=30), spray now evolves under evaporation/convection alone

## Dataset Characteristics

- **Total Training Samples**: 156,365 paired transitions
- **Unique Particles**: 17,939
- **Validation Samples**: 19,560
- **Test Samples**: 19,570
- **Total Columns**: 55 (normalized + absolute + clustering features)

## Key Physics Indicators

### nParticle Dynamics (delnParticle_abs)
- **Positive increases**: 0 samples (0.00%) ✓
- **Negative decreases**: 0 samples (0.00%)
- **Zero (no change)**: 156,365 samples (100.00%)
- **Interpretation**: 
  - **ZERO breakup**: No more secondary atomization
  - **ZERO particle loss**: No evaporation-induced disappearance
  - **STABLE system**: Parcels maintaining constant particle counts
  - **Post-injection dynamics**: Pure advection/convection, no source/sink

### Diameter Evolution (del_d_abs)
- **Mean change**: -7.99e-6 m
- **All values negative**: Droplets shrinking
- **Max shrinkage**: -1.20e-4 m
- **Interpretation**: Slow evaporation of stable droplets

### Temperature Change (del_T_abs)
- **Mean change**: -1.12e-2 K
- **Range**: -167 K to +79 K
- **Interpretation**: Minor temperature variations in cooler spray region

### Non-Dimensional Numbers
- **Weber (We_abs)**: 0.56 to 268 (lower range, less breakup pressure)
- **Reynolds (Re_abs)**: 2.8 to 4,049 (wider turbulence spectrum)

## Data Quality

✓ **No NaN values** in entire dataset  
✓ **No INF values** in entire dataset  
✓ All 54 columns properly populated with physical values  
✓ **Largest dataset**: 156.4k samples (most training data)

## Key Difference from Active Injection Phases

| Metric | Early (t=0-10) | Peak (t=21-29) | Post (t=50-59) |
|--------|----------------|----------------|----------------|
| Pos increases | 25.85% | 5.73% | **0.00%** |
| Mean increase | +45.2 | +0.82 | **0.00** |
| Breakup | **Chaotic** | **Transitional** | **None** |
| System state | **Atomizing** | **Stabilizing** | **Stable** |

## Column Categories

Same structure, but different physics:
- **Metadata** (3): origId, timestep, timestep_next
- **Normalized** (23): in_* (17) + out_* (6)
- **Absolute** (23): in_*_abs (17) + out_*_abs (6)
- **Clustering** (6): We_abs, Re_abs, del_T_abs, del_d_abs, delnParticle_abs (zero), delUrelMag_abs

## Physics Interpretation

### Why delnParticle_abs = 0?

This is **physically correct**:
- **Injection stopped** (~t=30): No new fuel entering
- **No breakup** (t≥30): Parcels too cool for further atomization
- **No coalescence**: Droplets too dispersed
- **Steady-state transport**: Particles advected by flow, no multiplication

### Spray maturity indicators

- Pre-injection (t=0-10): **Breakup dominated** (high We, Re)
- Steady-state (t=21-29): **Transition zone** (moderate breakup)
- Post-injection (t=50-59): **Transport dominated** (only evaporation)

## Use Cases

- **Evaporation modeling**: Train on steady-state droplet cooling/shrinking
- **Late-stage spray**: Behavior far from injection
- **Dispersion models**: Particle cloud spreading without source
- **Baseline reference**: Compare against injection phases

## Important Notes

- **Largest dataset** (156k samples): Best for statistical learning
- **Simplest physics**: No breakup complicates neural networks
- **Representative spray**: Captures ~80% of simulation time
- **Realistic conditions**: Injection usually short, post-injection long

## Data Integrity Notes

- delnParticle_abs exactly zero (no numerical artifacts)
- del_T_abs and del_d_abs non-zero (normal evaporation)
- All persistent particles retained (no disappearance)
- Stable statistics: Low variance in nParticle changes

## Recommendation for ML Training

**Use this dataset for:**
- ✓ General spray dynamics (77% of simulation time in this phase)
- ✓ Evaporation/cooling models
- ✓ Statistically robust learning (largest sample size)
- ✗ Breakup dynamics (use early/peak injection instead)
- ✗ Injection modeling (injection has ended)
