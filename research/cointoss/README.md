# Coin Toss Model - Comprehensive Bayesian Inference Research Fork

A production-ready, modular implementation of Bayesian inference for the classic Beta-Bernoulli coin toss problem, featuring comprehensive logging, diagnostics, visualization, and statistical analysis capabilities.

## Overview

This research fork extends the basic coin toss example with a complete suite of analysis, validation, and visualization tools, demonstrating best practices for Bayesian inference research workflows using RxInfer.jl.

### Key Features

#### 🎯 **Comprehensive Inference**
- **Conjugate Bayesian Inference**: Beta-Bernoulli model with analytical and numerical solutions
- **Free Energy Tracking**: Monitor convergence and model evidence
- **Convergence Diagnostics**: Automatic detection and reporting
- **KL Divergence Analysis**: Information gain quantification
- **Posterior Predictive Checks**: Model validation

#### 📊 **Statistical Analysis**
- **Credible Intervals**: Bayesian uncertainty quantification
- **Posterior Statistics**: Mean, mode, variance, quantiles
- **Model Comparison**: Log marginal likelihood computation
- **Diagnostic Metrics**: Comprehensive inference diagnostics
- **Empirical Validation**: Compare theoretical and observed statistics

#### 🎨 **Advanced Visualization**
- **Multiple Themes**: Default, dark, and colorblind-friendly color schemes
- **Comprehensive Dashboards**: Multi-panel diagnostic visualizations
- **Convergence Plots**: Free energy and parameter evolution
- **Credible Interval Plots**: Visual uncertainty representation
- **Animations**: Sequential Bayesian updating visualization
- **Posterior Predictive Plots**: Model checking visualizations

#### 🛠 **Production-Ready Features**
- **Plaintext Configuration**: TOML-based parameter management
- **Modular Architecture**: Separated concerns for extensibility
- **Comprehensive Logging**: Console, file, structured JSON formats
- **Data Export**: CSV and JSON output formats
- **Performance Tracking**: Timing and resource usage metrics
- **CLI Interface**: Command-line argument parsing
- **Test Suite**: Comprehensive unit and integration tests

## Project Structure

```
research/cointoss/
├── config.toml              # Plaintext configuration file
├── config.jl                # Configuration module
├── meta.jl                  # Project metadata
├── Project.toml             # Julia package dependencies
├── run.jl                   # Main execution script with CLI
├── README.md                # This documentation
├── src/                     # Core implementation modules
│   ├── model.jl            # Probabilistic model and data generation
│   ├── inference.jl        # RxInfer execution and diagnostics
│   ├── visualization.jl    # Plotting and animation
│   └── utils.jl            # Logging, export, and utilities
├── test/                    # Comprehensive test suite
│   └── runtests.jl         # Unit and integration tests
└── outputs/                 # Generated outputs (created on run)
    ├── data/               # Generated data files
    ├── plots/              # Visualization outputs
    ├── animations/         # GIF animations
    ├── results/            # Comprehensive results bundles
    └── logs/               # Execution logs
```

## Quick Start

### Prerequisites

- Julia 1.10 or later
- Required packages (automatically installed from Project.toml)

### Basic Usage

```bash
# Navigate to the project directory
cd research/cointoss

# Run with default settings
julia run.jl

# Run with custom parameters
julia run.jl --n=1000 --theta=0.6

# Run with verbose logging
julia run.jl --verbose

# Use dark theme for visualizations
julia run.jl --theme=dark

# Show help
julia run.jl --help
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--help, -h` | Show comprehensive help message |
| `--verbose` | Enable detailed logging |
| `--quiet` | Minimize logging output |
| `--no-animation` | Disable animation generation |
| `--benchmark` | Run performance benchmarks |
| `--n=N` | Number of coin tosses (default: 500) |
| `--theta=θ` | True coin bias parameter (default: 0.75) |
| `--seed=S` | Random seed for reproducibility (default: 42) |
| `--theme=THEME` | Visualization theme: default, dark, colorblind |

### Configuration

All parameters can be customized via `config.toml`:

```toml
# Data generation
[data]
n_samples = 500
theta_real = 0.75
seed = 42

# Model prior
[model]
prior_a = 4.0
prior_b = 8.0

# Inference settings
[inference]
iterations = 10
free_energy_tracking = true
convergence_check = true

# Visualization
[visualization]
theme = "default"
plot_resolution = 1000
save_plots = true

# Animation
[animation]
enabled = true
fps = 10
sample_increments = [10, 25, 50, 100, 200, 500]

# Output directories
[output]
output_dir = "outputs"
plots_dir = "outputs/plots"
animations_dir = "outputs/animations"
```

## Module Descriptions

### Model Module (`src/model.jl`)

Probabilistic model definition and data generation:

- **`coin_model(y, a, b)`**: RxInfer model definition for Beta-Bernoulli inference
- **`generate_coin_data()`**: Synthetic data generation with reproducibility
- **`analytical_posterior()`**: Analytical conjugate posterior computation
- **`posterior_statistics()`**: Comprehensive posterior statistics
- **`log_marginal_likelihood()`**: Model evidence calculation

**Key Functions:**
```julia
# Generate data
data = generate_coin_data(n=500, theta_real=0.75, seed=42)

# Compute analytical posterior
posterior = analytical_posterior(data.observations, prior_a, prior_b)

# Get statistics
stats = posterior_statistics(posterior, credible_level=0.95)
```

### Inference Module (`src/inference.jl`)

Bayesian inference execution and diagnostics:

- **`run_inference()`**: Execute RxInfer with comprehensive tracking
- **`InferenceResult`**: Structured result container
- **`kl_divergence()`**: KL divergence between Beta distributions
- **`posterior_predictive_check()`**: Model validation
- **`compute_convergence_diagnostics()`**: Convergence analysis

**Key Functions:**
```julia
# Run inference
result = run_inference(
    data, prior_a, prior_b;
    iterations=10,
    track_fe=true,
    convergence_check=true
)

# Access results
posterior = result.posterior
diagnostics = result.diagnostics
free_energy = result.free_energy
```

### Visualization Module (`src/visualization.jl`)

Comprehensive plotting and animation:

- **`plot_prior_posterior()`**: Prior vs. posterior comparison
- **`plot_credible_interval()`**: Credible interval visualization
- **`plot_convergence()`**: Free energy convergence plot
- **`plot_comprehensive_dashboard()`**: Multi-panel diagnostic dashboard
- **`create_inference_animation()`**: Sequential update animation

**Key Functions:**
```julia
# Create dashboard
dashboard = plot_comprehensive_dashboard(
    prior, posterior, data, free_energy;
    theta_real=true_theta,
    theme="dark"
)

# Create animation
anim = create_inference_animation(
    data, prior_a, prior_b, [10, 50, 100, 500];
    theta_real=true_theta,
    fps=10
)
```

### Utils Module (`src/utils.jl`)

Logging, export, and utility functions:

- **`setup_logging()`**: Configurable logging system
- **`Timer`**: Execution timing utilities
- **`export_to_csv()` / `export_to_json()`**: Data export
- **`save_experiment_results()`**: Comprehensive results bundling
- **`compute_summary_statistics()`**: Statistical analysis
- **`bernoulli_confidence_interval()`**: Frequentist confidence intervals

**Key Functions:**
```julia
# Setup logging
setup_logging(verbose=true, structured=true, performance=true)

# Time operations
timer = Timer("operation_name")
# ... do work ...
close(timer)

# Export results
export_to_json(results, "outputs/results/experiment.json")
```

## Output Files

### Generated Outputs

Running the experiment produces the following outputs:

#### Plots (`outputs/plots/`)
- `comprehensive_dashboard.png`: Multi-panel overview
- `prior_posterior.png`: Distribution comparison
- `credible_interval.png`: Uncertainty visualization
- `data_histogram.png`: Observed data distribution
- `posterior_predictive.png`: Model validation
- `free_energy_convergence.png`: Inference diagnostics

#### Animations (`outputs/animations/`)
- `bayesian_update.gif`: Sequential Bayesian updating visualization

#### Data (`outputs/data/`)
- `coin_toss_observations.csv`: Raw observation data

#### Results (`outputs/results/coin_toss_bayesian_inference_TIMESTAMP/`)
- `results.json`: Comprehensive structured results
- `results.csv`: Flattened results for analysis
- `metadata.json`: Experiment metadata

#### Logs (`outputs/logs/`)
- `cointoss.log`: Human-readable log file
- `cointoss_structured.jsonl`: JSON Lines structured logs
- `cointoss_performance.csv`: Performance metrics

## Usage Examples

### Basic Inference

```julia
using Pkg
Pkg.activate(".")

include("src/model.jl")
include("src/inference.jl")

using .CoinTossModel
using .CoinTossInference

# Generate data
data = generate_coin_data(n=500, theta_real=0.75, seed=42)

# Run inference
result = run_inference(data.observations, 4.0, 8.0)

# Analyze results
println("Posterior mean: ", mean(result.posterior))
println("Converged: ", result.converged)
```

### Custom Analysis

```julia
# Load complete experiment
include("run.jl")

# Run with custom configuration
config = load_config()
config["data"]["n_samples"] = 1000
config["data"]["theta_real"] = 0.6
config["animation"]["enabled"] = false

results = run_experiment(config)

# Access specific results
posterior_mean = results["results"]["inference"]["posterior_statistics"]["mean"]
kl_div = results["results"]["inference"]["diagnostics"]["kl_divergence"]
```

### Visualization Only

```julia
include("src/model.jl")
include("src/inference.jl")
include("src/visualization.jl")

using .CoinTossModel
using .CoinTossInference
using .CoinTossVisualization

# Load or generate data
data = generate_coin_data(n=200, theta_real=0.65)

# Run inference
result = run_inference(data.observations, 2.0, 3.0)

# Create custom visualization
plot = plot_prior_posterior(
    result.prior,
    result.posterior;
    theta_real=0.65,
    theme="colorblind"
)

save_plot(plot, "custom_plot.png")
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
julia test/runtests.jl

# Or using Pkg
julia -e 'using Pkg; Pkg.activate("."); Pkg.test()'
```

### Test Coverage

- **Configuration**: Parameter validation and loading
- **Data Generation**: Reproducibility and edge cases
- **Model**: Analytical computations and conjugacy
- **Inference**: Convergence, diagnostics, KL divergence
- **Visualization**: Plot creation and themes
- **Utilities**: Timing, export, statistics
- **Integration**: End-to-end workflows

## Mathematical Background

### Beta-Bernoulli Model

**Generative Model:**
```
θ ~ Beta(a, b)           # Prior over coin bias
y_i ~ Bernoulli(θ)       # Observations
```

**Posterior (Conjugate Update):**
```
θ | y ~ Beta(a + n_heads, b + n_tails)
```

**Log Marginal Likelihood:**
```
log p(y) = log B(a + n_heads, b + n_tails) - log B(a, b)
```

where `B(α, β)` is the Beta function.

### Key Quantities

- **Prior Mean**: `E[θ] = a / (a + b)`
- **Posterior Mean**: `E[θ|y] = (a + n_heads) / (a + b + n)`
- **Posterior Mode**: `(a + n_heads - 1) / (a + b + n - 2)` for `a, b > 1`
- **Credible Interval**: `[F⁻¹(α/2), F⁻¹(1-α/2)]` where `F` is the posterior CDF

## Performance

Typical execution times (on modern hardware):

- **Data Generation**: < 0.01s (500 samples)
- **Inference**: < 0.1s (10 iterations)
- **Visualization**: < 2s (all plots)
- **Animation**: < 5s (6 frames)
- **Total Experiment**: < 10s

Memory usage: < 100 MB

## Extensions and Customization

### Adding New Analyses

1. Create new function in appropriate module (e.g., `src/inference.jl`)
2. Add to experiment pipeline in `run.jl`
3. Update configuration in `config.toml`
4. Add tests in `test/runtests.jl`

### Custom Priors

Modify `config.toml`:
```toml
[model]
prior_a = 1.0  # Uniform prior: Beta(1, 1)
prior_b = 1.0
```

Or use command line:
```bash
julia run.jl --prior-a=1.0 --prior-b=1.0
```

### Alternative Visualizations

Add new plot functions to `src/visualization.jl`:

```julia
function plot_custom_analysis(posterior::Beta, ...)
    # Your custom visualization
end
```

## References

### Core References
- **RxInfer.jl Documentation**: [https://rxinfer.ml](https://rxinfer.ml)
- **Beta-Binomial Model**: Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
- **Conjugate Priors**: Gelman et al. (2013). Bayesian Data Analysis.

### Advanced Topics
- **Variational Inference**: Blei et al. (2017). Variational Inference: A Review for Statisticians.
- **Model Selection**: Kass & Raftery (1995). Bayes Factors.
- **Credible Intervals**: Kruschke, J. (2014). Doing Bayesian Data Analysis.

## Contributing

This is a research fork demonstrating best practices. To contribute:

1. Follow the modular architecture pattern
2. Add comprehensive tests for new features
3. Update documentation and examples
4. Maintain backward compatibility with upstream examples

## License

Follows the same license as RxInferExamples.jl.

## Acknowledgments

Based on the original Coin Toss Model example from RxInferExamples.jl, extended with comprehensive research workflow capabilities following patterns from the Active Inference Mountain Car research implementation.

---

**For questions or issues, refer to the main RxInferExamples.jl repository or RxInfer.jl documentation.**

