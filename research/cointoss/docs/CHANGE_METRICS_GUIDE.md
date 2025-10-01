# Change/Delta Metrics Analysis Guide

**Complete temporal analysis of learning dynamics and convergence rates**

---

## Overview

The Coin Toss Model now includes comprehensive **change/delta metrics** that track the **rate of change** of all key quantities per observation. These metrics provide deep insights into:

- **Learning dynamics**: How fast is the model learning?
- **Convergence behavior**: Is uncertainty reducing at expected rates?
- **Information flow**: What's the marginal value of each new observation?
- **Parameter evolution**: How do posterior parameters update?

---

## 📊 Delta Metrics Catalog

### 1. Free Energy Change Rate
**Metric**: `delta_free_energy` (ΔFE/n)

**Definition**: Change in Bethe Free Energy per observation

**Formula**:
```
ΔFE/n = (FE(n) - FE(n-1)) / Δn
```

**Interpretation**:
- **Negative values**: Free energy decreasing (model improving)
- **Magnitude**: Rate of model optimization
- **Trend**: Should approach zero as model converges

**Use Cases**:
- Monitor convergence speed
- Detect learning plateaus
- Identify optimal stopping points

---

### 2. Model Evidence Change Rate
**Metric**: `delta_log_ml` (Δlog P(y)/n)

**Definition**: Change in log marginal likelihood per observation

**Formula**:
```
Δlog P(y)/n = (log P(y₁:ₙ) - log P(y₁:ₙ₋₁)) / Δn
```

**Interpretation**:
- **Positive values**: Each observation increases model evidence
- **Magnitude**: How much each observation improves the model
- **Trend**: Typically decreases as more data is observed

**Use Cases**:
- Quantify value of additional data
- Model comparison across sample sizes
- Budget planning for data collection

---

### 3. Expected Log Likelihood Change Rate
**Metric**: `delta_expected_ll` (ΔE[LL]/n)

**Definition**: Change in expected log likelihood per observation

**Formula**:
```
ΔE[LL]/n = (E_q[log p(y|θ)]ₙ - E_q[log p(y|θ)]ₙ₋₁) / Δn
```

**Interpretation**:
- Measures how well the posterior predicts the data
- Related to model fit improvement
- Should stabilize as posterior converges

**Use Cases**:
- Track predictive performance evolution
- Identify when model has sufficient data
- Compare learning curves

---

### 4. Learning Rate (Information Gain Rate)
**Metric**: `learning_rate` (ΔKL/n)

**Definition**: KL divergence change (information gain) per observation

**Formula**:
```
Learning Rate = ΔKL(q||p)/n = (KL(qₙ||p) - KL(qₙ₋₁||p)) / Δn
```

**Interpretation**:
- **Positive values**: Each observation adds information
- **Magnitude**: Bits of information gained per observation
- **Trend**: Decreases as posterior saturates

**Use Cases**:
- Measure learning efficiency
- Compare different priors
- Quantify diminishing returns of data
- Active learning / experimental design

**Key Property**: `learning_rate ≡ delta_kl`

---

### 5. Convergence Rate (Uncertainty Reduction Rate)
**Metric**: `convergence_rate` (Δσ²/n)

**Definition**: Variance reduction per observation

**Formula**:
```
Convergence Rate = (σ²ₙ₋₁ - σ²ₙ) / Δn
```

**Interpretation**:
- **Positive values**: Uncertainty decreasing (good)
- **Magnitude**: How fast we're gaining certainty
- **Trend**: Decreases over time (harder to reduce uncertainty)

**Use Cases**:
- Monitor convergence to ground truth
- Predict required sample size
- Confidence interval planning
- Stopping rules for sequential experiments

---

### 6. Posterior Mean Change Rate
**Metric**: `delta_posterior_mean` (Δμ/n)

**Definition**: Change in posterior mean per observation

**Formula**:
```
Δμ/n = (μₙ - μₙ₋₁) / Δn
```

**Interpretation**:
- **Sign**: Direction of belief update
- **Magnitude**: Strength of evidence in observation
- **Trend**: Should approach zero as beliefs stabilize

**Use Cases**:
- Track belief evolution
- Detect concept drift
- Measure responsiveness to new data

---

### 7. Posterior Std Change Rate
**Metric**: `delta_posterior_std` (Δσ/n)

**Definition**: Change in posterior standard deviation per observation

**Formula**:
```
Δσ/n = (σₙ - σₙ₋₁) / Δn
```

**Interpretation**:
- **Negative values**: Uncertainty decreasing (expected)
- **Magnitude**: Rate of uncertainty reduction
- Related to `convergence_rate` (variance version)

**Use Cases**:
- Monitor uncertainty evolution
- Credible interval forecasting

---

### 8. Parameter Change Rates
**Metrics**: `delta_alpha` (Δα/n), `delta_beta` (Δβ/n)

**Definition**: Change in Beta distribution parameters per observation

**Formulas**:
```
Δα/n = (αₙ - αₙ₋₁) / Δn
Δβ/n = (βₙ - βₙ₋₁) / Δn
```

**Interpretation**:
- **Δα/n**: Rate of "heads" evidence accumulation
- **Δβ/n**: Rate of "tails" evidence accumulation
- **Expected**: Δα/n ≈ P(heads), Δβ/n ≈ P(tails)

**Use Cases**:
- Validate conjugate update rules
- Track parameter growth
- Verify analytical solutions

**Key Property**: For Beta-Bernoulli:
```
Δα/n → E[y] = θ
Δβ/n → E[1-y] = 1-θ
```

---

## 📈 Visualization Integration

### Individual Timeseries Plots

All delta metrics are visualized as individual plots:

```
outputs/timeseries/
  delta_free_energy_timeseries.png
  delta_log_ml_timeseries.png
  delta_expected_ll_timeseries.png
  delta_kl_timeseries.png
  delta_alpha_timeseries.png
  delta_beta_timeseries.png
  delta_posterior_mean_timeseries.png
  delta_posterior_std_timeseries.png
  convergence_rate_timeseries.png
  learning_rate_timeseries.png
```

Each plot shows:
- X-axis: Number of observations
- Y-axis: Change rate (per observation)
- Zero reference line (dashed)
- Legend and title

---

### Graphical Abstract Integration

**New ROW 5** (panels 17-20) in the 28-panel graphical abstract:

| Panel | Metric | Description |
|-------|--------|-------------|
| 17 | ΔFE/n | Free Energy change rate |
| 18 | Δlog P(y)/n | Model Evidence change rate |
| 19 | ΔKL/n | Learning rate (info gain/obs) |
| 20 | Δσ²/n | Convergence rate (uncertainty reduction/obs) |

**Layout**: 7 rows × 4 columns = 28 panels total
**Size**: 2400×4200 pixels

---

## 💾 Data Export

### CSV Format

All delta metrics exported to: `outputs/timeseries/temporal_evolution_data.csv`

**Total columns**: 34 (up from 24)

**New columns**:
```
- delta_free_energy
- delta_log_ml
- delta_expected_ll
- delta_kl
- delta_alpha
- delta_beta
- delta_posterior_mean
- delta_posterior_std
- convergence_rate
- learning_rate
```

**Structure**:
```csv
n_samples,delta_free_energy,delta_log_ml,...
1,0.0,0.0,...
2,-0.523,0.234,...
3,-0.412,0.189,...
...
```

---

## 🔬 Practical Applications

### 1. Sample Size Planning

**Question**: How many observations do we need?

**Approach**:
1. Plot `convergence_rate` vs. `n_samples`
2. Find where rate drops below threshold (e.g., 0.001)
3. That's your minimum sample size

**Code**:
```julia
evolution = compute_temporal_evolution(data, prior_a, prior_b)
threshold = 0.001
sufficient_n = findlast(evolution["convergence_rate"] .> threshold)
```

---

### 2. Learning Efficiency Analysis

**Question**: Which prior learns faster?

**Approach**:
1. Run experiments with different priors
2. Compare `learning_rate` curves
3. Higher initial learning rate = more efficient prior

**Interpretation**:
- Informative priors: High initial rate, fast saturation
- Vague priors: Low initial rate, slow saturation
- Optimal: Balanced between flexibility and efficiency

---

### 3. Diminishing Returns Detection

**Question**: When should we stop collecting data?

**Approach**:
1. Monitor `delta_log_ml` (value per observation)
2. When cost > value, stop collecting
3. Use for active learning / experimental design

**Decision Rule**:
```julia
cost_per_observation = 10.0  # dollars
if abs(delta_log_ml[end]) * value_per_bit < cost_per_observation
    stop_experiment = true
end
```

---

### 4. Convergence Diagnostics

**Question**: Has the posterior converged?

**Approach**:
1. Check `delta_posterior_mean` near zero
2. Check `convergence_rate` positive and decreasing
3. Check `delta_free_energy` near zero

**Criteria**:
```julia
converged = (
    abs(delta_posterior_mean[end]) < 0.001 &&
    convergence_rate[end] > 0 &&
    abs(delta_free_energy[end]) < 0.01
)
```

---

### 5. Parameter Update Validation

**Question**: Are parameters updating correctly?

**Approach**:
1. Plot `delta_alpha` vs. empirical head rate
2. Should match closely for conjugate models
3. Deviations indicate numerical issues

**Validation**:
```julia
empirical_rate = evolution["head_rate"]
theoretical_delta_alpha = empirical_rate
actual_delta_alpha = evolution["delta_alpha"][2:end]  # Skip first point

correlation = cor(theoretical_delta_alpha, actual_delta_alpha)
@assert correlation > 0.99  # Should be nearly perfect
```

---

## 📊 Typical Behavior Patterns

### Early Learning Phase (n < 50)

**Characteristics**:
- `delta_free_energy`: Large negative values (rapid optimization)
- `learning_rate`: High (lots of info per observation)
- `convergence_rate`: High (rapid uncertainty reduction)
- `delta_posterior_mean`: Large (beliefs updating significantly)

**Interpretation**: Model rapidly incorporating new information

---

### Mid Learning Phase (50 < n < 200)

**Characteristics**:
- `delta_free_energy`: Moderate negative values
- `learning_rate`: Decreasing
- `convergence_rate`: Decreasing
- `delta_posterior_mean`: Moderate

**Interpretation**: Continued learning, but diminishing returns

---

### Late Learning Phase (n > 200)

**Characteristics**:
- `delta_free_energy`: Near zero
- `learning_rate`: Very small
- `convergence_rate`: Very small  
- `delta_posterior_mean`: Near zero

**Interpretation**: Posterior has converged, new data adds little

---

## 🎯 Interpretation Guidelines

### Positive vs. Negative Values

| Metric | Positive | Negative | Zero |
|--------|----------|----------|------|
| delta_free_energy | Getting worse | Getting better | Converged |
| delta_log_ml | Evidence increasing | Evidence decreasing | No change |
| learning_rate | Learning | Forgetting (rare) | No learning |
| convergence_rate | Uncertainty reducing | Uncertainty increasing | Stable |
| delta_posterior_mean | Belief increasing | Belief decreasing | Stable |

---

### Magnitude Interpretation

- **Large magnitude**: Strong signal, rapid change
- **Small magnitude**: Weak signal, slow change
- **Decreasing magnitude**: Approaching convergence
- **Increasing magnitude**: Divergence or non-stationarity

---

### Trend Analysis

- **Monotonic decrease**: Healthy convergence
- **Oscillation**: Potential numerical issues
- **Plateau**: Convergence reached
- **Sudden spike**: Outlier or regime change

---

## 🔍 Advanced Topics

### Information Theory Connections

**Learning Rate** measures the rate of information flow:
```
I(Y; Θ) = KL(q(θ|y) || p(θ))
Learning Rate = dI/dn
```

**Total Information**: Integration of learning rate
```
Total Info = ∫ learning_rate(n) dn ≈ Σ learning_rate × Δn
```

---

### Statistical Properties

For Beta-Bernoulli with true parameter θ:

**Expected Learning Rate**:
```
E[ΔKL/n] ≈ Fisher Information / (2n)
```

**Expected Convergence Rate**:
```
E[Δσ²/n] ≈ -σ²(n) × Fisher Information
```

**Asymptotic Behavior**:
```
As n → ∞:
  - learning_rate → 0
  - convergence_rate → 0
  - delta_posterior_mean → 0
```

---

### Computational Notes

**First Point**: All delta metrics are 0 at n=1 (no previous point)

**Division by Δn**: Normalizes to "per observation" basis
- If sample points are [1, 2, 5, 10], Δn values are [1, 1, 3, 5]
- Each delta is divided by its corresponding Δn

**Numerical Stability**: Use log-space for very small/large values

---

## 📚 References

### Theory
- **KL Divergence**: Information theory, relative entropy
- **Free Energy**: Variational inference, ELBO
- **Fisher Information**: Cramér-Rao bound, asymptotic variance

### RxInfer.jl
- **Memory Addon**: Message tracing
- **Free Energy Tracking**: Convergence monitoring
- **Callbacks**: Iteration tracking

---

## ✅ Summary

**10 New Metrics**:
1. ΔFree Energy (per obs)
2. ΔLog Marginal Likelihood (per obs)
3. ΔExpected Log Likelihood (per obs)
4. ΔKL Divergence (per obs)
5. Learning Rate (ΔKL/n)
6. Convergence Rate (Δσ²/n)
7. Δα (per obs)
8. Δβ (per obs)
9. ΔPosterior Mean (per obs)
10. ΔPosterior Std (per obs)

**Outputs**:
- ✅ 34 columns in CSV export
- ✅ 25 individual timeseries plots
- ✅ 4 panels in graphical abstract (ROW 5)
- ✅ Complete documentation

**Applications**:
- Sample size planning
- Learning efficiency analysis
- Diminishing returns detection
- Convergence diagnostics
- Parameter update validation

---

*For implementation details, see `src/timeseries_diagnostics.jl`*

