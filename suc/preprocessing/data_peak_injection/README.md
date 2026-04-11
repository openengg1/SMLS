# Peak Injection Phase Sample Dataset (t=21-29)

## Physical Context

**Time Range**: Timesteps 21-29 (21-29 ms, steady injection period)  
**Phase**: Active spray injection - transitional atomization  
**Physics**: System stabilizing with injection ongoing; breakup slowing down

## Dataset Characteristics

- **Total Training Samples**: 151,800 paired transitions
- **Unique Particles**: 108,854
- **Validation Samples**: 18,975
- **Test Samples**: 19,009
- **Total Columns**: 55 (normalized + absolute + clustering features)

## Key Physics Indicators

### nParticle Dynamics (delnParticle_abs)
- **Positive increases**: 8,696 samples (5.73%)
- **Mean increase**: +0.82 particles per parcel
- **Max increase**: +290 particles
- **Interpretation**: 
  - Only 5.7% of parcels still undergoing breakup (vs 25.85% in early injection)
  - Much smaller breakup events (mean +0.82 vs +45.2)
  - Spray atomization process stabilizing
  - Transition from chaotic to ordered flow

### Diameter Evolution (del_d_abs)
- **Mean change**: -7.99e-6 m (evaporation)
- **All values negative**: Droplets shrinking
- **Max shrinkage**: -1.20e-4 m
- **Interpretation**: Sustained evaporation throughout active injection

### Temperature Change (del_T_abs)
- **Mean change**: +19.6 K
- **Range**: -5.45 K to +122.8 K
- **Interpretation**: Fuel continues heating in chamber environment

### Non-Dimensional Numbers
- **Weber (We_abs)**: 0.56 to 1,290 (breakup regime active but waning)
- **Reynolds (Re_abs)**: 348 to 62,144 (turbulent conditions)

## Data Quality

✓ **No NaN values** in entire dataset  
✓ **No INF values** in entire dataset  
✓ All 54 columns properly populated with physical values  
✓ **4.2x larger dataset** than early injection (151.8k vs 36k samples)

## Key Difference from Early Injection

| Metric | Early (t=0-10) | Peak (t=21-29) |
|--------|----------------|----------------|
| Pos increases | 25.85% | 5.73% |
| Mean increase | +45.2 particles | +0.82 particles |
| Max increase | +2,979 particles | +290 particles |
| Implication | **Chaotic breakup zone** | **Stabilizing atomization** |

## Column Categories

Same as early injection:
- **Metadata** (3): origId, timestep, timestep_next
- **Normalized** (23): in_* (17) + out_* (6)
- **Absolute** (23): in_*_abs (17) + out_*_abs (6)
- **Clustering** (6): We_abs, Re_abs, del_T_abs, del_d_abs, delnParticle_abs, delUrelMag_abs

## Physics Transitions

This phase captures the transition where:
1. **Injection continues** → new fuel entering
2. **Breakup subsides** → fewer secondary atomization events
3. **Spray structure forms** → large-scale flow patterns emerge
4. **Evaporation ongoing** → all droplets shrinking

## Use Cases

- **Injection modeling**: Model steady-state injection behavior
- **Atomization mechanics**: Study scale-down of secondary breakup
- **Spray flame interaction**: Input for combustion models (larger dataset)
- **Machine learning**: 4x more training samples for statistical robustness

## Notes

- Peak injection window: Most representative of "typical" injection condition
- 4x larger than early phase: Better for neural network training
- Still shows 5.7% breakup: Not fully stabilized yet
- Transition point: t≥30 shows zero breakup (see post-injection)
