# Early Injection Phase Sample Dataset (t=0-10)

## Physical Context

**Time Range**: Timesteps 0-10 (first 10 ms of spray injection)  
**Phase**: Early spray injection - initial atomization and breakup  
**Physics**: Raw fuel entering chamber with maximum secondary breakup activity

## Dataset Characteristics

- **Total Training Samples**: 36,028 paired transitions
- **Unique Particles**: 17,939
- **Validation Samples**: 4,503
- **Test Samples**: 4,536
- **Total Columns**: 55 (normalized + absolute + clustering features)

## Key Physics Indicators

### nParticle Dynamics (delnParticle_abs)
- **Positive increases**: 9,314 samples (25.85%)
- **Mean increase**: +45.2 particles per parcel
- **Max increase**: +2,979 particles
- **Interpretation**: 
  - 1 in 4 parcels undergo secondary breakup/splitting
  - NEW parcels being created from injected fuel
  - Highest breakup activity in entire spray lifecycle

### Diameter Evolution (del_d_abs)
- **Mean change**: -7.99e-6 m (evaporation)
- **All values negative**: Droplets are shrinking
- **Max shrinkage**: -1.20e-4 m
- **Interpretation**: Early evaporation begins immediately upon injection

### Temperature Change (del_T_abs)
- **Mean change**: +19.6 K
- **Range**: -5.45 K to +122.8 K
- **Interpretation**: Fuel heating from ambient chamber conditions

### Non-Dimensional Numbers
- **Weber (We_abs)**: 0.56 to 1,290 (full breakup regime)
- **Reynolds (Re_abs)**: 348 to 62,144 (turbulent flow)

## Data Quality

✓ **No NaN values** in entire dataset  
✓ **No INF values** in entire dataset  
✓ All 54 columns properly populated with physical values  

## Column Categories

**Metadata** (3 cols):
- origId: Particle ID
- timestep, timestep_next: Time indices

**Normalized Features** (23 cols):
- in_* (17): Input droplet/field properties (mean=0, std=1)
- out_* (6): Output properties at next timestep

**Absolute Features** (23 cols):
- in_*_abs (17): Physical scale inputs
- out_*_abs (6): Physical scale outputs

**Clustering Features** (6 cols):
- We_abs: Weber number
- Re_abs: Reynolds number
- del_T_abs: Temperature change
- del_d_abs: Diameter change
- delnParticle_abs: Particle count change
- delUrelMag_abs: Velocity magnitude change

## Input Features (17)

**Particle Properties**:
- d: diameter
- U:0, U:1, U:2: velocity components
- T: temperature
- nParticle: particle count per parcel
- rho, mu, sigma: fluid properties

**Eulerian Field** (1-NN interpolated):
- euler_T, euler_U:*, euler_H2O, euler_p, euler_rho

**Derived**:
- mass_proxy: density × diameter³

## Output Features (6)

Actual values at t+1 (not deltas):
- d, U:0, U:1, U:2, T, nParticle

Then deltas are computed as: out_value - in_value

## Use Cases

- **Physics-informed models**: Train on injection dynamics
- **Breakup modeling**: Learn secondary atomization patterns
- **Multi-phase coupling**: Fuel injection interaction with chamber
- **Clustering analysis**: Identify breakup event signatures

## Notes

- Persistent particles only: Only droplets surviving t→t+1 included
- 1-NN interpolation: Eulerian values from nearest grid cell
- StandardScaler normalized: Applied to training data
- High nParticle variance: Reflects active breakup zone
