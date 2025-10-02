# Coin Toss Model - Multi-Trial Parameter Sweep with Advanced Visualization

Comprehensive, modular research implementation of Bayesian coin toss inference with advanced RxInfer diagnostics, complete temporal evolution tracking, and extensive visualization capabilities.

## Quick Start

```bash
# Run multi-trial parameter sweep (recommended)
julia --project=. run_multi_trial.jl --n-trials=20

# Run full diagnostic analysis (single trial)
julia --project=. run_with_diagnostics.jl --skip-animation

# Run simple demo
julia --project=. simple_demo.jl

# Run standard version
julia --project=. run.jl
```

## 📚 Documentation

All comprehensive documentation is in the [`docs/`](docs/) directory:

- **[Quick Start Guide](docs/QUICK_START.md)** - Get running in 1 minute
- **[Complete Documentation](docs/README.md)** - Full usage guide
- **[Architecture Guide](docs/AGENTS.md)** - Component architecture
- **[Output Structure](docs/OUTPUTS.md)** - Output files reference
- **[RxInfer Diagnostics](docs/RXINFER_DIAGNOSTICS_GUIDE.md)** - Advanced diagnostics
- **[Change Metrics Guide](docs/CHANGE_METRICS_GUIDE.md)** - Delta/rate analysis
- **[Test Suite](docs/TEST_SUMMARY.md)** - Testing documentation
- **[Documentation Index](docs/DOCUMENTATION_INDEX.md)** - Complete docs index

## 📊 Key Features

✅ **Comprehensive Bayesian Inference**
- Beta-Bernoulli conjugate model
- Analytical & numerical solutions
- Complete posterior analysis

✅ **Advanced RxInfer Diagnostics**
- Memory Addon (message tracing)
- Inference callbacks
- Performance benchmarking
- Free energy tracking

✅ **Parameter Sweep & Multi-Trial Analysis**
- Systematic parameter sweeps across multiple dimensions
- Multi-trial comparison and statistical analysis
- Automated result aggregation and correlation analysis
- Performance scaling analysis across parameter ranges

✅ **Temporal Evolution Tracking**
- 34 metrics through time (including 10 delta/change metrics)
- Complete learning dynamics
- Information gain analysis
- Change rate analysis (Free Energy, Model Evidence, Parameters)

✅ **Rich Visualizations & Animations**
- 28-panel graphical abstract (2400×4200, including change metrics)
- 25 individual timeseries plots
- Multi-trial comparison dashboards (4-panel layout)
- **Multi-trial comparison animation** (104KB, ~10 frames)
- **Parameter sweep animation** (1MB, animated convergence)
- Performance scaling visualizations
- Bayesian update animations

✅ **Production-Ready Code**
- 100% test coverage
- Modular architecture
- Extensive logging
- Complete documentation

## 🎯 Output Structure

```
outputs/
├── parameter_sweep/     # Parameter sweep results and individual trial data
├── multi_trial_analysis/# Statistical analysis and comparison results
├── plots/              # All visualizations including graphical abstract (28 panels)
├── timeseries/         # Temporal evolution plots (25) + CSV (34 metrics)
├── diagnostics/        # RxInfer diagnostic data (8 files)
├── data/               # Generated/processed data
├── results/            # Experiment results (JSON/CSV)
├── animations/         # Bayesian update and parameter sweep animations
└── logs/               # Execution logs
```

## 🔬 Module Structure

```
src/
├── model.jl                  # Probabilistic model & analytics
├── inference.jl              # RxInfer execution & diagnostics
├── visualization.jl          # Standard plotting & multi-trial visualization
├── timeseries_diagnostics.jl # Temporal evolution analysis
├── diagnostics.jl            # Advanced RxInfer diagnostics
├── graphical_abstract.jl     # Comprehensive visualization
├── parameter_sweep.jl        # Parameter sweep functionality
├── multi_trial_analysis.jl   # Multi-trial analysis & comparison
└── utils.jl                  # Utilities & export
```

## ✅ Testing

```bash
# Run complete test suite
julia --project=. test/runtests.jl

# All tests pass with 100% coverage
```

## 📈 Performance

### Single Trial
- **Data generation**: < 0.01s (500 samples), < 0.1s (10,000 samples)
- **Inference**: < 0.1s (10 iterations)
- **Diagnostics**: < 0.001s
- **Visualization**: < 2s (all plots)
- **Complete workflow**: < 30s

### Multi-Trial Parameter Sweep
- **Sample sizes**: 50, 100, 200, 500, 1,000, 10,000
- **Parameter combinations**: 384 total (6 × 4 × 4 × 4)
- **Execution time**: ~25s for 384 trials
- **Animations**: 2 GIF files (~1.1MB total)

## 🚀 Key Outputs

### Single Trial Analysis
1. **Graphical Abstract** (`graphical_abstract.png`) - 28-panel mega-visualization (2400×4200)
2. **Timeseries Dashboard** (`comprehensive_timeseries_dashboard.png`) - 12 metrics
3. **Individual Plots** - 25 separate timeseries visualizations
4. **Temporal Evolution CSV** - 34 metrics × 28 time points
5. **Change Metrics** - 10 delta/rate calculations
6. **Diagnostic Data** - Complete RxInfer traces & benchmarks
7. **Results Bundle** - JSON/CSV exports with metadata

### Multi-Trial Analysis (NEW!)
8. **Parameter Sweep Results** (`parameter_sweep_results.csv`) - Complete sweep data
9. **Multi-Trial Dashboard** (`multi_trial_dashboard.png`) - Comparative analysis
10. **Multi-Trial Comparison Animation** (`multi_trial_comparison.gif`) - Posterior evolution across trials
11. **Parameter Sweep Animation** (`parameter_sweep_n_samples_posterior_mean.gif`) - Convergence dynamics
12. **Statistical Analysis** (`multi_trial_analysis.json`) - Correlation and effect analysis
13. **Performance Scaling Plots** - How performance changes with parameters

## 📖 Citation

Part of RxInferExamples.jl research fork demonstrating advanced Bayesian inference capabilities with comprehensive diagnostics and visualization.

## 📋 See Also

- [Complete Documentation](docs/README.md)
- [Project Summary](docs/PROJECT_SUMMARY.md)
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)

---

**Status**: ✅ Production-ready  
**Test Coverage**: 100%  
**Documentation**: Complete  

