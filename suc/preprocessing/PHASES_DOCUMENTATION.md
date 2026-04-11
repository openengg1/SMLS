# Spray Dataset Phases Summary and Physics Documentation

## Overview

The complete spray dataset spans 200 milliseconds of simulation time. Spray physics changes dramatically depending on whether fuel is actively being injected. This document organizes data into three physically distinct phases.

---

## Three Data Phases

### Phase 1: Early Injection (t=0-10 ms)
- **Location**: `data_early_injection/`
- **Samples**: 36,028 training, 4,503 val, 4,536 test
- **nParticle changes**: +25.85% positive (high breakup)
- **Mean nParticle Δ**: +45.2 particles/parcel
- **Status**: Raw spray entering chamber, massive secondary atomization

### Phase 2: Peak Injection (t=21-29 ms)  
- **Location**: `data_peak_injection/`
- **Samples**: 151,800 training, 18,975 val, 19,009 test
- **nParticle changes**: +5.73% positive (moderate breakup)
- **Mean nParticle Δ**: +0.82 particles/parcel
- **Status**: Steady injection ongoing, atomization stabilizing
- **Note**: Largest dataset (4.2x early, similar to post)

### Phase 3: Post-Injection (t=50-59 ms)
- **Location**: `data_post_injection/`
- **Samples**: 156,365 training, 19,560 val, 19,570 test
- **nParticle changes**: +0.00% (zero breakup)
- **Mean nParticle Δ**: 0.0 particles/parcel
- **Status**: Injection ended (~t=30), spray dispersing by evaporation alone
- **Note**: Largest dataset for late-stage spray modeling

---

## Physics by Phase

### Early Injection (t=0-10): ATOMIZATION
```
Fuel injection → Secondary breakup → Parcel multiplication
Breakup mechanism: High Weber number from injection velocity
Parcel nParticle: INCREASES (1 parcel → many smaller parcels)
Dominant physics: Turbulent jet atomization
```

**Characteristics:**
- 1 in 4 parcels undergo secondary breakup
- Some parcels split into 2,979 smaller parcels!
- Violently chaotic flow with rapid phase change
- Maximum energy dissipation

**Best for:**
- Atomization models
- Jet dynamics
- Primary + secondary breakup
- High-energy spray mechanics

---

### Peak Injection (t=21-29): TRANSITION
```
Steady injection ongoing → System stabilizing → Breakup waning
Breakup mechanism: Waning energy, fewer secondary events
Parcel nParticle: MOSTLY CONSTANT (5.73% still break up)
Dominant physics: Transitional between injection-driven and flow-driven
```

**Characteristics:**
- Only 5.73% of parcels still breaking up (vs 25.85% at t=0-10)
- When breakup occurs, it's small (mean +0.82 particles)
- Coexists with evaporation, coalescence dynamics
- Large-scale spray structure forming

**Best for:**
- General spray modeling (representative of "average" spray condition)
- **Largest dataset**: 4.2x more samples than early phase
- Evaporation + flow interaction
- Droplet clustering analysis
- Neural network training (good data volume, varied physics)

---

### Post-Injection (t=50-59): DISPERSION
```
Injection stopped → No more source → Pure transport + evaporation
Breakup mechanism: NONE (system cooled below breakup threshold)
Parcel nParticle: CONSTANT (perfectly stable)
Dominant physics: Lagrangian particle transport + Eulerian diffusion
```

**Characteristics:**
- **Zero breakup across all 156,365 samples**
- **Zero particle multiplication** (delnParticle_abs = 0.000 everywhere)
- Only evaporation causes change (diameter shrinking)
- Stable, predictable dynamics

**Best for:**
- Evaporation modeling
- Late-stage spray datasets
- Post-injection behavior
- Baseline reference (simplest physics)
- Machine learning **with 156k samples** (largest dataset)

---

## Why Three Phases?

### Reason 1: Different Physics
| Aspect | Early | Peak | Post |
|--------|-------|------|------|
| Injection status | **Just started** | **Ongoing (steady)** | **Ended (~t=30)** |
| Dominant process | Breakup | Breakup + evaporation | Evaporation only |
| Breakup % | 25.85% | 5.73% | 0.00% |
| Max breakup size | 2,979 particles | 290 particles | 0 particles |
| Flow regime | Chaotic | Transitional | Ordered |

### Reason 2: Training Data Availability
- **Early phase** (t=0-10): Only 10 ms available → 36k samples
- **Peak phase** (t=21-29): 9 ms available → 152k samples ✓ BEST
- **Post phase** (t=50-59): 10 ms available → 156k samples ✓ BEST

### Reason 3: Application-Specific Models
- **Need breakup model?** → Use early/peak data
- **Need transport model?** → Use post data  
- **General spray model?** → Use peak data (largest, most varied)

---

## Column Definitions

All phases have 54 columns organized as:

### Metadata (3)
- `origId`: Individual droplet/parcel ID
- `timestep`: Current time index
- `timestep_next`: Next time index

### Normalized Inputs (17)
- `in_d`, `in_U:0/1/2`, `in_T`, `in_nParticle`: Particle properties at t
- `in_rho`, `in_mu`, `in_sigma`: Fluid properties
- `in_euler_T`, `in_euler_U:0/1/2`, `in_euler_H2O`, `in_euler_p`, `in_euler_rho`: Grid values (1-NN) at t
- `in_mass_proxy`: ρ × d³ proxy

### Normalized Outputs (6)
- `out_d`, `out_U:0/1/2`, `out_T`, `out_nParticle`: Properties at t+1

### Absolute Inputs (17)
- All `in_*_abs`: Physical scale values (not normalized)

### Absolute Outputs (6)
- All `out_*_abs`: Physical scale values at t+1

### Clustering Features (5)
- `We_abs = ρ × |U|² × d / σ`: Weber number (breakup criterion)
- `Re_abs = ρ × |U| × d / μ`: Reynolds number (flow regime)
- `del_T_abs = out_T_abs - in_T_abs`: Temperature change
- `del_d_abs = out_d_abs - in_d_abs`: Diameter change (always negative = evaporation)
- `delnParticle_abs = out_nParticle_abs - in_nParticle_abs`: **The key differentiator between phases**

---

## Key Insight: nParticle as Phase Discriminator

The `delnParticle_abs` column perfectly identifies each phase:

```
Phase          | delnParticle_abs
---------------|------------------
Early (t=0-10) | 25.85% positive
Peak (t=21-29) | 5.73% positive
Post (t=50-59) | 0.00% positive
```

This is **not a bug**—it's **physics**:
- Early: Injection → high energy → breakup → nParticle increases
- Peak: Steady state → energy declining → fewer breakup events
- Post: Injection ended → no energy source → no breakup

---

## Data Quality

All three phases verified:
- ✓ No NaN values
- ✓ No INF values
- ✓ All 54 columns populated
- ✓ Physical ranges sensible
- ✓ Statistics consistent with spray physics
- ✓ No negative pressures, diameters, or densities

---

## Next Steps: Full Dataset Processing

Ready to generate **full 200-timestep dataset**:

```bash
cd /home/rmishra/projects/stochasticMLSpray/model/suc/preprocessing
python3 raw_to_paired.py --output-dir ./data
```

**Expected output:**
- ~1.7M paired samples (195k × 200/23 timesteps on average)
- All 54 columns with correct physics
- Training/validation/test split (80/10/10)
- Three phases well-represented:
  - Early injection: 5% of data (t=0-10)
  - Peak injection: 11% of data (t=21-29)
  - Post-injection: 55% of data (t=30-200)
  - Transition zones: 29% of data

This balanced distribution captures complete spray lifecycle while emphasizing post-injection physics (most realistic for real combustors where injection is brief).

---

## References

- **Early phase details**: See `data_early_injection/README.md`
- **Peak phase details**: See `data_peak_injection/README.md`
- **Post phase details**: See `data_post_injection/README.md`
- **Preprocessing code**: `raw_to_paired.py`
- **Raw data**: `/data/case1/VTK/lagrangian/sprayCloud/` and `/eulerian/`
