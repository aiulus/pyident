"""Interpretation Guide for PE vs Visible Subspace Experiments

This guide helps researchers understand and interpret the results from sim_pe.py experiments,
connecting the theoretical hypotheses to the empirical evidence.

## Core Theoretical Framework

The experiment tests the hypothesis:
**"Visible subspace dimension acts as a ceiling, PE order acts as a floor"**

### What This Means:
1. **Floor Effect (PE Order)**: Estimation errors should DROP when PE order r exceeds critical thresholds
2. **Ceiling Effect (Visible Subspace)**: Some errors cannot improve beyond visible subspace limitations

## Key Plots and Their Interpretation

### 1. **Hypothesis Dashboard** (`hypothesis_dashboard.png`)
**Purpose**: One-stop validation of all theoretical predictions
**What to Look For**:
- Threshold effects: Are errors systematically lower after critical PE thresholds?
- Markov validation: Do E_k parameters drop when r > k? (Theory: Yes)
- Ceiling vs floor: Do V⊥ errors stay high while V errors improve? (Theory: Yes)
- Statistical significance: Which effects have p < 0.05?

**Red Flags**:
- No threshold effects visible → PE order not acting as "floor"
- Markov parameters don't follow theory → Problems with moment-PE estimation
- V and V⊥ errors both improve equally → No "ceiling" effect

### 2. **Markov Parameter Validation** (`pe_vs_visible_markov_parameters.png`)
**Purpose**: Direct test of Propositions 2.30/4.20 from theory
**What to Look For**:
- Each subplot shows E_k = ||Â^k B̂ - A^k B||_F vs moment-PE order
- Red vertical line at r = k+1 marks theoretical threshold
- Errors should drop noticeably when r > k

**Interpretation**:
- Clear drops after red lines → Theory confirmed
- No pattern → Moment-PE estimation may be unreliable
- Opposite pattern → Possible implementation bug

### 3. **Subspace Ceiling Effect** (`pe_vs_visible_subspace_ceiling_effect.png`)
**Purpose**: Show that some directions cannot be improved regardless of PE order
**What to Look For**:
- V(x₀) errors: Should improve with PE order (floor effect)
- V(x₀)⊥ errors: Should stay high regardless of PE (ceiling effect)
- Clear separation between the two curves

**Red Flags**:
- Both curves identical → No subspace structure detected
- V⊥ improving → Possible numerical issues or wrong V computation

### 4. **Standard Basis Plots**
**Purpose**: Traditional view of errors vs PE order
**What to Look For**:
- "Elbow" behavior: Rapid improvement then saturation
- Different behavior between partial vs full visibility scenarios
- Consistent patterns across different PE order estimation methods

## Statistical Summary (`hypothesis_tests.csv`)

### Key Columns:
- **hypothesis**: Specific theoretical prediction being tested
- **effect_size_pct**: % improvement when crossing threshold
- **p_value**: Statistical significance (< 0.05 = significant)
- **significant**: YES/NO based on p < 0.05

### Interpretation Guidelines:
- **Large effect sizes** (>20%) + significant p-values → Strong evidence
- **Small effects** (<5%) even if significant → Practically negligible
- **Non-significant** results → Need more data or effect doesn't exist

## Common Issues and Debugging

### 1. **No Clear Threshold Effects**
**Possible Causes**:
- Insufficient data (increase --n-trials, --n-systems)
- PE orders too close together (try wider range)
- System matrices not sufficiently different between scenarios
- Numerical precision issues

**Solutions**:
```bash
# More data
python -m pyident.experiments.sim_pe --n-systems 200 --n-trials 50

# Wider PE range  
python -m pyident.experiments.sim_pe --max-pe-order 12

# Different system parameters
python -m pyident.experiments.sim_pe --n 8 --vdim 3
```

### 2. **Markov Parameter Theory Violations**
**Symptoms**: E_k increases when r > k (opposite of theory)
**Possible Causes**:
- Moment-PE estimation unreliable for short horizons
- Numerical conditioning issues
- Wrong discrete-time vs continuous-time assumptions

**Solutions**:
```bash
# Longer horizons
python -m pyident.experiments.sim_pe --T 100

# Disable exact PE enforcement to see if it's a PRBS issue
python -m pyident.experiments.sim_pe --no-exact-pe

# Add noise to test robustness
python -m pyident.experiments.sim_pe --noise-std 0.01
```

### 3. **No Ceiling Effect**
**Symptoms**: V and V⊥ errors behave identically
**Possible Causes**:
- Visible subspace computation incorrect
- All systems accidentally fully visible
- Numerical rank determination issues

**Solutions**:
```bash
# Check visible dimensions are actually different
grep "dim_visible" output/pe_vs_visible.csv | sort | uniq

# Try more extreme visibility difference
python -m pyident.experiments.sim_pe --n 10 --vdim 2

# Check visible basis tolerance
python -m pyident.experiments.sim_pe --visible-tol 1e-6
```

## Advanced Analysis Tips

### 1. **Effect Size Interpretation**
- **10-30%**: Small but meaningful effect
- **30-70%**: Moderate effect  
- **>70%**: Large effect (strong theoretical validation)

### 2. **Sample Size Considerations**
- Start with n-systems=50, n-trials=20 for quick tests
- Use n-systems=200, n-trials=50+ for publication-quality results
- More systems > more trials (reduces system-specific artifacts)

### 3. **Parameter Sensitivity Analysis**
```bash
# Test different discretization
python -m pyident.experiments.sim_pe --dt 0.05

# Test different input scaling
python -m pyident.experiments.sim_pe --u-scale 1.0

# Test different system sizes
python -m pyident.experiments.sim_pe --n 8 --vdim 4
```

## Expected Results Summary

### Strong Theory Support:
- Threshold effects with p < 0.01, effect sizes > 30%
- Clear Markov parameter patterns following E_k ↓ when r > k
- Obvious ceiling effects (V⊥ >> V errors)
- Consistent patterns across multiple PE order estimation methods

### Weak/Inconclusive:
- Marginal significance (0.01 < p < 0.05)
- Small effect sizes (< 20%)
- High variance (wide confidence intervals)
- Inconsistent patterns between block vs moment PE

### Theory Violation (Investigate):
- Opposite threshold effects (errors increase with PE)
- Markov parameters violate theoretical predictions  
- No ceiling effects despite partial visibility
- Inconsistent results across replications

## Citation and Context

When using these results, remember to:
1. Report both statistical significance AND effect sizes
2. Acknowledge limitations (discrete-time, specific system classes, etc.)
3. Mention sensitivity to parameter choices (T, dt, noise levels)
4. Compare with theoretical predictions explicitly
5. Discuss practical vs structural identifiability implications
"""