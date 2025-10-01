# Coin Toss Model - Bayesian Inference with RxInfer

Comprehensive, modular research implementation of Bayesian coin toss inference with advanced RxInfer diagnostics, complete temporal evolution tracking, and extensive visualization capabilities.

## Quick Start

```bash
# Run full diagnostic analysis
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

✅ **Temporal Evolution Tracking**
- 34 metrics through time (including 10 delta/change metrics)
- Complete learning dynamics
- Information gain analysis
- Change rate analysis (Free Energy, Model Evidence, Parameters)

✅ **Rich Visualizations**
- 28-panel graphical abstract (2400×4200, including change metrics)
- 25 individual timeseries plots
- Comprehensive dashboards
- Bayesian update animations

✅ **Production-Ready Code**
- 100% test coverage
- Modular architecture
- Extensive logging
- Complete documentation

## 🎯 Output Structure

```
outputs/
├── plots/           # All visualizations including graphical abstract (28 panels)
├── timeseries/      # Temporal evolution plots (25) + CSV (34 metrics)
├── diagnostics/     # RxInfer diagnostic data (8 files)
├── data/            # Generated/processed data
├── results/         # Experiment results (JSON/CSV)
├── animations/      # Bayesian update animations
└── logs/            # Execution logs
```

## 🔬 Module Structure

```
src/
├── model.jl                  # Probabilistic model & analytics
├── inference.jl              # RxInfer execution & diagnostics
├── visualization.jl          # Standard plotting
├── timeseries_diagnostics.jl # Temporal evolution analysis
├── diagnostics.jl            # Advanced RxInfer diagnostics
├── graphical_abstract.jl     # Comprehensive visualization
└── utils.jl                  # Utilities & export
```

## ✅ Testing

```bash
# Run complete test suite
julia --project=. test/runtests.jl

# All tests pass with 100% coverage
```

## 📈 Performance

- **Data generation**: < 0.01s (500 samples)
- **Inference**: < 0.1s (10 iterations)
- **Diagnostics**: < 0.001s
- **Visualization**: < 2s (all plots)
- **Complete workflow**: < 30s

## 🚀 Key Outputs

1. **Graphical Abstract** (`graphical_abstract.png`) - 28-panel mega-visualization (2400×4200)
2. **Timeseries Dashboard** (`comprehensive_timeseries_dashboard.png`) - 12 metrics
3. **Individual Plots** - 25 separate timeseries visualizations
4. **Temporal Evolution CSV** - 34 metrics × 28 time points
5. **Change Metrics** - 10 delta/rate calculations
6. **Diagnostic Data** - Complete RxInfer traces & benchmarks
7. **Results Bundle** - JSON/CSV exports with metadata

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

