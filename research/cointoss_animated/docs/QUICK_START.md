# Coin Toss Model - Quick Start Guide

## 1-Minute Setup

```bash
cd research/cointoss
julia run.jl
```

That's it! The experiment will:
- ✓ Generate synthetic coin toss data
- ✓ Run Bayesian inference using RxInfer
- ✓ Create comprehensive visualizations
- ✓ Export results in multiple formats
- ✓ Generate animations showing Bayesian updating

## What You Get

### Outputs Created
- **Plots** → `outputs/plots/` - Dashboard and individual diagnostic plots
- **Animations** → `outputs/animations/` - Bayesian updating GIF
- **Data** → `outputs/data/` - Raw observations CSV
- **Results** → `outputs/results/` - JSON and CSV result bundles
- **Logs** → `outputs/logs/` - Execution logs

### Default Experiment
- **500 coin tosses** with true bias θ = 0.75
- **Beta(4, 8) prior** (skeptical of heads)
- **10 inference iterations** with free energy tracking
- **Comprehensive diagnostics** and validation

## Common Commands

### Run Experiments

```bash
# Default settings
julia run.jl

# Custom parameters
julia run.jl --n=1000 --theta=0.6

# Different themes
julia run.jl --theme=dark
julia run.jl --theme=colorblind

# Quick test (no animation)
julia run.jl --no-animation --quiet

# Performance benchmark
julia run.jl --benchmark --verbose
```

### Quick Demo

```bash
# Fast demo without full visualization
julia simple_demo.jl
```

### Run Tests

```bash
julia test/runtests.jl
```

## Configuration

Edit `config.toml` to customize:

```toml
# Quick tweaks
[data]
n_samples = 1000        # More data
theta_real = 0.6        # Different true bias

[model]
prior_a = 1.0           # Uniform prior
prior_b = 1.0

[visualization]
theme = "dark"          # Dark theme
```

## Understanding the Output

### Key Plots

1. **`comprehensive_dashboard.png`** - Everything at a glance
   - Prior vs posterior distributions
   - Observed data histogram
   - Credible intervals
   - Posterior predictive checks
   - Free energy convergence

2. **`bayesian_update.gif`** - Watch the posterior evolve
   - Shows sequential Bayesian updating
   - See how more data narrows uncertainty

### Results Structure

```
outputs/results/coin_toss_bayesian_inference_TIMESTAMP/
├── results.json          # Complete structured results
├── results.csv          # Flattened for analysis
└── metadata.json        # Experiment metadata
```

## Key Statistics Explained

### From Console Output

```
Posterior Statistics:
  Mean: 0.7456          # Best estimate of θ
  Mode: 0.7442          # Most likely value
  Std: 0.0189           # Uncertainty
  95% CI: [0.7084, 0.7820]  # Credible interval
```

**Interpretation:**
- We're 95% confident θ is between 0.71 and 0.78
- True value (0.75) falls in this interval ✓
- Standard deviation shows our remaining uncertainty

### Diagnostic Metrics

- **KL Divergence**: Information gained from data
- **Log Marginal Likelihood**: Model evidence
- **Empirical Rate**: Observed proportion of heads
- **Information Gain**: How much we learned

## Next Steps

### 1. Explore Visualizations
Open `outputs/plots/comprehensive_dashboard.png` to see all diagnostics at once.

### 2. Check Animation
View `outputs/animations/bayesian_update.gif` to see how the posterior evolves with data.

### 3. Analyze Results
Load `outputs/results/*/results.json` to access all computed statistics.

### 4. Customize
Edit `config.toml` to run different scenarios:
- Try different priors (informative vs uninformative)
- Vary sample sizes (10 vs 10,000)
- Change true bias parameter

### 5. Extend
Add your own analyses in `src/`:
- Custom visualizations in `visualization.jl`
- New diagnostics in `inference.jl`
- Additional statistics in `utils.jl`

## Troubleshooting

### Dependencies Missing
```bash
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

### Plots Not Saving
Check that output directories exist - they're created automatically, but verify permissions.

### Animation Fails
Ensure GR backend is working:
```julia
using Plots
gr()  # Set GR backend
```

### Test Failures
Run with verbose output:
```bash
julia --project=.. test/runtests.jl
```

## Resources

- **Full Documentation**: See `README.md`
- **Configuration Reference**: See `config.toml` with inline comments
- **Code Examples**: See `simple_demo.jl` for minimal example
- **Tests**: See `test/runtests.jl` for usage patterns

## CLI Options Quick Reference

| Option | Effect |
|--------|--------|
| `--help` | Show detailed help |
| `--verbose` | Detailed logging |
| `--quiet` | Minimal output |
| `--n=N` | Number of tosses |
| `--theta=θ` | True coin bias |
| `--seed=S` | Random seed |
| `--theme=T` | Visualization theme |
| `--no-animation` | Skip animation |
| `--benchmark` | Performance analysis |

---

**That's it! You're ready to explore Bayesian inference with the coin toss model.**

For detailed documentation, see `README.md`. For questions, check the test suite in `test/runtests.jl` for usage examples.

