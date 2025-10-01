# Coin Toss Model - Unified Output Structure Documentation

## Overview

All experiment outputs are centralized in a **single `outputs/` directory** for easy management, archival, and analysis. This document describes the complete output structure, file formats, and usage patterns.

## Directory Structure

```
outputs/
├── data/                    # Raw and processed data
├── plots/                   # Static visualizations
├── animations/              # Dynamic visualizations
├── results/                 # Comprehensive results bundles
└── logs/                    # Execution logs and metrics
```

---

## 1. Data Directory (`outputs/data/`)

### Purpose
Store raw observations and processed datasets.

### Files Generated

#### `coin_toss_observations.csv`
**Format**: CSV with headers  
**When Created**: Stage 1 (Data Generation)  
**Content**:
```csv
observation_id,outcome
1,1.0
2,0.0
3,1.0
...
```

**Columns**:
- `observation_id`: Sequential integer (1 to n)
- `outcome`: Binary float (0.0 or 1.0)

**Usage**:
```julia
using CSV, DataFrames
data = CSV.read("outputs/data/coin_toss_observations.csv", DataFrame)
```

**Size**: ~10 KB for 500 observations

---

## 2. Plots Directory (`outputs/plots/`)

### Purpose
Store all static diagnostic visualizations.

### Files Generated

#### `comprehensive_dashboard.png`
**Type**: Multi-panel overview  
**When Created**: Stage 4 (Visualization)  
**Dimensions**: 1600×1800 pixels (with free energy) or 1600×1200 (without)  
**Content**:
- Panel 1: Prior vs Posterior distributions
- Panel 2: Observed data histogram
- Panel 3: Credible interval visualization
- Panel 4: Posterior predictive check
- Panel 5: Free energy convergence (if tracked)

**File Size**: ~200-500 KB

---

#### `prior_posterior.png`
**Type**: Distribution comparison plot  
**Dimensions**: 800×600 pixels  
**Content**:
- Prior distribution (Beta(a, b))
- Posterior distribution (Beta(a+n₁, b+n₀))
- True θ value (vertical line)
- Posterior mean (vertical dash-dot line)
- Posterior mode (vertical dotted line)

**Annotations**:
- Legend with all elements
- True θ value displayed
- Posterior statistics

**File Size**: ~100-150 KB

---

#### `credible_interval.png`
**Type**: Uncertainty quantification plot  
**Dimensions**: 800×600 pixels  
**Content**:
- Full posterior distribution
- Highlighted credible interval region
- True θ value marker
- Coverage indicator (✓ or ✗)

**Annotations**:
- CI bounds in legend
- Coverage status
- Credible level (default 95%)

**File Size**: ~100-150 KB

---

#### `data_histogram.png`
**Type**: Observed data summary  
**Dimensions**: 800×600 pixels  
**Content**:
- Bar chart: Tails count vs Heads count
- Counts and percentages
- Comparison to true θ

**Annotations**:
- Count labels above bars
- Percentage labels
- True θ vs empirical comparison

**File Size**: ~50-100 KB

---

#### `posterior_predictive.png`
**Type**: Model validation plot  
**Dimensions**: 800×600 pixels  
**Content**:
- Observed proportion of heads
- Predicted proportion (from posterior)
- Error bars (standard errors)

**Annotations**:
- Proportions with uncertainties
- Visual comparison

**File Size**: ~50-100 KB

---

#### `free_energy_convergence.png`
**Type**: Convergence diagnostic  
**Dimensions**: 800×600 pixels  
**Content**:
- Free energy trajectory over iterations
- Line plot with markers
- Final value annotation

**When Created**: Only if free energy tracking enabled

**File Size**: ~80-120 KB

---

#### `graphical_abstract.png`
**Type**: Comprehensive mega-visualization  
**Dimensions**: 2400×4200 pixels  
**Layout**: 7 rows × 4 columns = 28 panels  
**When Created**: run_with_diagnostics.jl only  
**Content**:
- ROW 1: Posterior distributions and statistics
- ROW 2: Data analysis and validation
- ROW 3: Temporal evolution (key metrics)
- ROW 4: Parameter evolution
- ROW 5: **Change metrics** (ΔFE, ΔML, Learning Rate, Convergence Rate)
- ROW 6: Final distributions and comparisons
- ROW 7: Computational diagnostics and summary

**File Size**: ~1.5-2.5 MB

**Purpose**: Single publication-ready visualization integrating all statistical, computational, and diagnostic information.

**See**: [Change Metrics Guide](CHANGE_METRICS_GUIDE.md) for ROW 5 interpretation

---

#### `comprehensive_timeseries_dashboard.png`
**Type**: 12-metric timeseries overview  
**Dimensions**: 2400×2400 pixels  
**Layout**: 4×3 grid  
**When Created**: run_with_diagnostics.jl only  
**Content**: 12 key metrics plotted through time

**File Size**: ~500-800 KB

---

### Total Plots Directory Size
- Standard run: ~600 KB - 1.2 MB (6-9 plots)
- Diagnostic run: ~3-5 MB (10+ plots including graphical abstract)

---

## 3. Animations Directory (`outputs/animations/`)

### Purpose
Store dynamic visualizations showing temporal evolution.

### Files Generated

#### `bayesian_update.gif`
**Type**: Animated GIF  
**When Created**: Stage 5 (Animation) - optional  
**Dimensions**: 800×600 pixels per frame  
**Frame Count**: Configurable (default: 6 frames)  
**FPS**: Configurable (default: 10)  
**Duration**: ~0.6 seconds (6 frames at 10 FPS)

**Content Per Frame**:
- Prior distribution (faded)
- Posterior with n observations (highlighted)
- True θ value
- Statistics overlay:
  - Number of observations
  - Heads count and percentage
  - Posterior mean and std

**Frame Sequence** (default):
1. n = 10 observations
2. n = 25 observations
3. n = 50 observations
4. n = 100 observations
5. n = 200 observations
6. n = 500 observations

**Visual Evolution**:
- Posterior narrows with more data
- Mean converges to true value
- Uncertainty decreases

**File Size**: ~800 KB - 2 MB

**Usage**:
- View in any GIF viewer/browser
- Embed in presentations
- Use for educational purposes

---

## 4. Timeseries Directory (`outputs/timeseries/`)

### Purpose
Store temporal evolution analysis and change metrics visualizations.

### Files Generated

#### Temporal Evolution Data

##### `temporal_evolution_data.csv`
**Format**: CSV with 34 columns  
**When Created**: During temporal analysis (run_with_diagnostics.jl)  
**Content**: 34 metrics tracked across 28 time points

**Columns** (34 total):
- **Basic metrics** (13): n_samples, posterior_mean, posterior_mode, posterior_std, posterior_var, ci_lower, ci_upper, ci_width, empirical_mean, n_heads, n_tails, head_rate, etc.
- **Parameter evolution** (4): posterior_alpha, posterior_beta, alpha_growth, beta_growth
- **Information theory** (5): kl_divergence, information_gain, free_energy, log_marginal_likelihood, expected_log_likelihood
- **Learning dynamics** (2): posterior_prior_diff, uncertainty_reduction
- **Change/Delta metrics** (10): delta_free_energy, delta_log_ml, delta_expected_ll, delta_kl, delta_alpha, delta_beta, delta_posterior_mean, delta_posterior_std, convergence_rate, learning_rate

**File Size**: ~15-25 KB

---

#### Timeseries Visualizations (25 plots)

**Standard Metrics** (15 plots):
- `posterior_mean_timeseries.png`
- `posterior_mode_timeseries.png`
- `posterior_std_timeseries.png`
- `posterior_var_timeseries.png`
- `ci_width_timeseries.png`
- `posterior_alpha_timeseries.png`
- `posterior_beta_timeseries.png`
- `kl_divergence_timeseries.png`
- `free_energy_timeseries.png`
- `log_marginal_likelihood_timeseries.png`
- `expected_log_likelihood_timeseries.png`
- `empirical_mean_timeseries.png`
- `head_rate_timeseries.png`
- `uncertainty_reduction_timeseries.png`
- `posterior_prior_diff_timeseries.png`

**Change/Delta Metrics** (10 plots):
- `delta_free_energy_timeseries.png` - Free energy change rate
- `delta_log_ml_timeseries.png` - Model evidence change rate
- `delta_expected_ll_timeseries.png` - Expected LL change rate
- `delta_kl_timeseries.png` - Information gain rate
- `delta_alpha_timeseries.png` - α parameter change rate
- `delta_beta_timeseries.png` - β parameter change rate
- `delta_posterior_mean_timeseries.png` - Mean change rate
- `delta_posterior_std_timeseries.png` - Std change rate
- `convergence_rate_timeseries.png` - Uncertainty reduction rate
- `learning_rate_timeseries.png` - Learning efficiency

**Dimensions**: 800×600 pixels per plot  
**File Size**: ~50-100 KB per plot

---

#### Comprehensive Timeseries Dashboard

##### `comprehensive_timeseries_dashboard.png`
**Type**: 12-metric dashboard  
**Dimensions**: 2400×2400 pixels  
**Layout**: 4×3 grid  
**Content**: 12 key metrics plotted through time

**File Size**: ~500-800 KB

---

### Total Timeseries Directory Size
Typical: 2-3 MB (25 plots + dashboard + CSV)

---

## 5. Results Directory (`outputs/results/`)

### Purpose
Store comprehensive experiment results in multiple formats.

### Directory Structure

Each experiment creates a timestamped subdirectory:

```
outputs/results/
└── coin_toss_bayesian_inference_YYYY-MM-DD_HH-MM-SS/
    ├── results.json
    ├── results.csv
    └── metadata.json
```

**Timestamp Format**: `YYYY-MM-DD_HH-MM-SS` (e.g., `2025-10-01_14-30-45`)

---

### Files in Results Bundle

#### `results.json`
**Format**: Pretty-printed JSON with 2-space indentation  
**When Created**: Stage 6 (Export)

**Structure**:
```json
{
  "experiment_name": "coin_toss_bayesian_inference",
  "timestamp": "2025-10-01 14:30:45",
  "config": {
    "data": {...},
    "model": {...},
    "inference": {...},
    "visualization": {...},
    "analysis": {...}
  },
  "results": {
    "inference": {
      "execution_time": 0.0823,
      "iterations": 10,
      "converged": true,
      "convergence_iteration": 2,
      "posterior_statistics": {
        "mean": 0.7456,
        "mode": 0.7442,
        "variance": 0.000358,
        "std": 0.0189,
        "credible_interval": [0.7084, 0.7820],
        "credible_level": 0.95,
        "alpha": 379.0,
        "beta": 129.0
      },
      "diagnostics": {
        "n_observations": 500,
        "n_heads": 375,
        "n_tails": 125,
        "empirical_rate": 0.75,
        "kl_divergence": 125.47,
        "information_gain": 125.47,
        "expected_log_likelihood": -344.26,
        "posterior_mean": 0.7456,
        "posterior_variance": 0.000358,
        "posterior_mode": 0.7442,
        "prior_mean": 0.3333,
        "prior_variance": 0.0154,
        "mean_shift": 0.4123,
        "variance_reduction": 0.0150,
        "final_free_energy": -350.12,
        "initial_free_energy": -225.65,
        "free_energy_reduction": 124.47,
        "mean_fe_change": 13.83,
        "max_fe_change": 98.52,
        "final_fe_change": 0.00012
      }
    },
    "analysis": {
      "analytical_posterior": {
        "alpha": 379.0,
        "beta": 129.0
      },
      "log_marginal_likelihood": -349.87
    },
    "posterior_predictive": {
      "pp_prob_heads": 0.7458,
      "predictive_mean": 0.7452,
      "predictive_variance": 0.1906,
      "n_samples": 10000
    }
  }
}
```

**File Size**: ~5-15 KB

**Usage**:
```julia
using JSON
results = JSON.parsefile("outputs/results/.../results.json")
posterior_mean = results["results"]["inference"]["posterior_statistics"]["mean"]
```

---

#### `results.csv`
**Format**: Two-column CSV (key-value pairs)  
**When Created**: Stage 6 (Export)

**Structure** (flattened from JSON):
```csv
key,value
experiment_name,coin_toss_bayesian_inference
timestamp,2025-10-01 14:30:45
config.data.n_samples,500
config.data.theta_real,0.75
config.model.prior_a,4.0
config.model.prior_b,8.0
results.inference.execution_time,0.0823
results.inference.converged,true
results.inference.posterior_statistics.mean,0.7456
results.inference.posterior_statistics.std,0.0189
results.inference.diagnostics.kl_divergence,125.47
...
```

**File Size**: ~3-8 KB

**Usage**:
```julia
using CSV, DataFrames
results = CSV.read("outputs/results/.../results.csv", DataFrame)
# Filter for specific keys
posterior_stats = filter(row -> startswith(row.key, "results.inference.posterior"), results)
```

**Benefits**:
- Easy to import into spreadsheets
- Simple grep/search operations
- Flat structure for quick analysis

---

#### `metadata.json`
**Format**: JSON  
**When Created**: Stage 6 (Export)

**Content**:
```json
{
  "experiment_name": "coin_toss_bayesian_inference",
  "timestamp": "2025-10-01_14-30-45",
  "julia_version": "1.10.0"
}
```

**File Size**: ~200 bytes

**Purpose**: Track execution environment and provenance

---

### Total Results Size
Typical: ~10-25 KB per experiment

---

## 6. Diagnostics Directory (`outputs/diagnostics/`)

### Purpose
Store advanced RxInfer diagnostic data from Memory Addon, Callbacks, and Benchmarks.

### Files Generated (8 total)

**When Created**: Only with `run_with_diagnostics.jl`

#### Message Tracing Files
- `memory_trace.json` (320KB) - Complete message computation history
- `message_trace_report.txt` (157KB) - Human-readable trace report

#### Callback Files
- `callback_trace.json` (4.3KB) - Complete event log
- `iteration_events.csv` (646B) - Iteration timing data
- `marginal_updates.csv` (1.6KB) - Posterior evolution
- `iteration_trace_report.txt` (1.6KB) - Event summary

#### Benchmark Files
- `benchmark_stats.csv` (489B) - Detailed statistics
- `benchmark_summary.json` (258B) - Summary metrics

**Total Directory Size**: ~485 KB

**See**: [RxInfer Diagnostics Guide](RXINFER_DIAGNOSTICS_GUIDE.md) for details

---

## 7. Logs Directory (`outputs/logs/`)

### Purpose
Store execution logs and performance metrics.

### Files Generated

#### `cointoss.log`
**Format**: Plain text, timestamped log messages  
**When Created**: Throughout execution (if `log_to_file = true`)

**Content Example**:
```
[ Info: ================================================================================
[ Info: Coin Toss Model - Bayesian Inference Experiment
[ Info: ================================================================================
[ Info: Configuration Summary:
[ Info:   Data: n=500, θ=0.75, seed=42
[ Info:   Prior: Beta(4.0, 8.0)
[ Info:   Inference: 10 iterations
[ Info:   Theme: default
[ Info: 
[ Info: ================================================================================
[ Info: Step 1: Data Generation
[ Info: ================================================================================
[ Info: PERF data_generation elapsed_seconds=0.0042
[ Info: Generated 500 coin tosses
[ Info:   True θ: 0.75
[ Info:   Observed heads: 375 (75.0%)
[ Info: Saved observations to: outputs/data/coin_toss_observations.csv
...
```

**File Size**: ~15-30 KB for full experiment

**Usage**:
- Debugging
- Understanding execution flow
- Timing analysis
- Error diagnosis

---

#### `cointoss_structured.jsonl`
**Format**: JSON Lines (one JSON object per line)  
**When Created**: Throughout execution (if `structured = true`)

**Content Example**:
```jsonl
{"timestamp":"2025-10-01T14:30:45.123","level":"info","message":"Configuration Summary","data":{"n_samples":500,"theta_real":0.75}}
{"timestamp":"2025-10-01T14:30:45.156","level":"info","message":"PERF data_generation","elapsed_seconds":0.0042}
{"timestamp":"2025-10-01T14:30:45.178","level":"info","message":"Starting inference","n_observations":500,"prior_a":4.0,"prior_b":8.0}
{"timestamp":"2025-10-01T14:30:45.261","level":"info","message":"Inference completed","execution_time":0.0823,"converged":true}
...
```

**File Size**: ~10-20 KB

**Usage**:
```julia
using JSON
logs = readlines("outputs/logs/cointoss_structured.jsonl")
parsed_logs = [JSON.parse(line) for line in logs]

# Filter for performance logs
perf_logs = filter(log -> startswith(log["message"], "PERF"), parsed_logs)

# Analyze timing
timings = Dict(
    split(log["message"])[2] => log["elapsed_seconds"]
    for log in perf_logs
)
```

**Benefits**:
- Machine-parseable
- Structured queries
- Log aggregation
- Time-series analysis

---

#### `cointoss_performance.csv`
**Format**: CSV with timing metrics  
**When Created**: Throughout execution (if `performance = true`)

**Structure**:
```csv
timestamp,operation,elapsed_seconds,memory_mb
2025-10-01 14:30:45.156,data_generation,0.0042,45.2
2025-10-01 14:30:45.261,bayesian_inference,0.0823,52.1
2025-10-01 14:30:45.389,statistical_analysis,0.0128,52.3
2025-10-01 14:30:47.512,visualization,2.1234,67.8
2025-10-01 14:30:52.678,animation_generation,5.1656,89.4
2025-10-01 14:30:52.789,results_export,0.1112,89.6
```

**File Size**: ~500 bytes - 2 KB

**Usage**:
```julia
using CSV, DataFrames
perf = CSV.read("outputs/logs/cointoss_performance.csv", DataFrame)

# Analyze bottlenecks
sort!(perf, :elapsed_seconds, rev=true)

# Total time
total_time = sum(perf.elapsed_seconds)

# Memory usage
max_memory = maximum(perf.memory_mb)
```

**Benefits**:
- Performance profiling
- Bottleneck identification
- Resource usage tracking
- Optimization targets

---

## Output Size Summary

### Standard Experiment (run.jl)

| Directory | Files | Total Size |
|-----------|-------|------------|
| `data/` | 1 | ~10 KB |
| `plots/` | 9 | ~1-2 MB |
| `animations/` | 1 | ~800 KB - 2 MB |
| `results/` | 3 | ~10-25 KB |
| `logs/` | 1-3 | ~25-50 KB |
| **Total** | **15-17** | **~2-4 MB** |

### Advanced Diagnostics (run_with_diagnostics.jl)

| Directory | Files | Total Size |
|-----------|-------|------------|
| `data/` | 1 | ~10 KB |
| `plots/` | 10 | ~3-5 MB |
| `timeseries/` | 26 | ~2-3 MB |
| `diagnostics/` | 8 | ~485 KB |
| `animations/` | 1 | ~800 KB - 2 MB |
| `results/` | 3 | ~10-25 KB |
| `logs/` | 1-3 | ~25-50 KB |
| **Total** | **50-52** | **~7-11 MB** |

### Scaling

For different sample sizes:

| n_samples | Data | Plots | Animation | Total |
|-----------|------|-------|-----------|-------|
| 100 | 3 KB | 600 KB | 600 KB | ~1.2 MB |
| 500 | 10 KB | 800 KB | 1.5 MB | ~2.3 MB |
| 1000 | 20 KB | 1 MB | 2 MB | ~3 MB |
| 10000 | 200 KB | 1.5 MB | 3 MB | ~4.7 MB |

---

## Output Management

### Cleanup

To remove old outputs:
```bash
# Remove specific experiment
rm -rf outputs/results/coin_toss_bayesian_inference_2025-10-01_14-30-45/

# Remove all results older than 7 days
find outputs/results/ -type d -mtime +7 -exec rm -rf {} +

# Clean everything except latest
ls -t outputs/results/ | tail -n +2 | xargs -I {} rm -rf "outputs/results/{}"
```

### Archival

To archive experiments:
```bash
# Archive specific experiment
tar -czf experiment_2025-10-01.tar.gz outputs/

# Archive with date
tar -czf cointoss_$(date +%Y%m%d).tar.gz outputs/

# Compress results only
tar -czf results_archive.tar.gz outputs/results/
```

### Sharing

To share outputs:
```bash
# Create shareable bundle
zip -r cointoss_outputs.zip outputs/

# Include only plots and results
zip -r cointoss_summary.zip outputs/plots/ outputs/results/

# Exclude logs
zip -r cointoss_clean.zip outputs/ -x "outputs/logs/*"
```

---

## Output File Formats Reference

### Image Formats
- **PNG**: Lossless, high quality, larger file size
- **DPI**: 100 (configurable via `config.toml`)
- **Resolution**: Configurable per plot type

### Animation Format
- **GIF**: Universal compatibility
- **Looping**: Enabled by default
- **Optimization**: Enabled for smaller files

### Data Formats
- **CSV**: Human-readable, spreadsheet-compatible
- **JSON**: Machine-parseable, preserves structure
- **JSONL**: Streamable, log-friendly

---

## Programmatic Output Access

### Load All Results

```julia
using JSON, CSV, DataFrames

# Load results from latest experiment
results_dirs = readdir("outputs/results/", join=true)
latest_dir = sort(results_dirs)[end]

# Load JSON results
results_json = JSON.parsefile(joinpath(latest_dir, "results.json"))

# Load CSV results
results_csv = CSV.read(joinpath(latest_dir, "results.csv"), DataFrame)

# Load metadata
metadata = JSON.parsefile(joinpath(latest_dir, "metadata.json"))
```

### Analyze Outputs

```julia
# Extract key statistics
posterior_mean = results_json["results"]["inference"]["posterior_statistics"]["mean"]
kl_div = results_json["results"]["inference"]["diagnostics"]["kl_divergence"]
converged = results_json["results"]["inference"]["converged"]

# Load observations
data = CSV.read("outputs/data/coin_toss_observations.csv", DataFrame)
empirical_mean = mean(data.outcome)

# Compare
println("Posterior mean: $posterior_mean")
println("Empirical mean: $empirical_mean")
println("KL divergence: $kl_div")
println("Converged: $converged")
```

### Batch Analysis

```julia
# Analyze multiple experiments
using Glob

all_results = []
for results_file in glob("outputs/results/*/results.json")
    push!(all_results, JSON.parsefile(results_file))
end

# Compare convergence across experiments
convergence_rates = [r["results"]["inference"]["converged"] for r in all_results]
mean_convergence = mean(convergence_rates)

# Extract all posterior means
posterior_means = [r["results"]["inference"]["posterior_statistics"]["mean"] 
                  for r in all_results]
```

---

## Output Verification Checklist

After running an experiment, verify:

- [ ] `outputs/data/coin_toss_observations.csv` exists and has n rows
- [ ] `outputs/plots/` contains 6 PNG files
- [ ] `outputs/animations/bayesian_update.gif` exists (if animation enabled)
- [ ] `outputs/results/coin_toss_bayesian_inference_*/` directory exists
- [ ] Results directory contains `results.json`, `results.csv`, `metadata.json`
- [ ] `outputs/logs/` contains log files (if logging enabled)
- [ ] All images are viewable
- [ ] JSON files are valid (parse without errors)
- [ ] CSV files have headers and proper formatting

### Quick Verification Script

```julia
function verify_outputs()
    checks = Dict(
        "Data CSV" => isfile("outputs/data/coin_toss_observations.csv"),
        "Dashboard" => isfile("outputs/plots/comprehensive_dashboard.png"),
        "Prior-Posterior" => isfile("outputs/plots/prior_posterior.png"),
        "Results exist" => !isempty(readdir("outputs/results/")),
        "Logs exist" => isfile("outputs/logs/cointoss.log")
    )
    
    all_pass = all(values(checks))
    
    for (check, status) in checks
        println("$check: ", status ? "✓" : "✗")
    end
    
    return all_pass
end

verify_outputs()
```

---

## Configuration for Outputs

Control output generation via `config.toml`:

```toml
[visualization]
save_plots = true          # Enable/disable plot saving
show_plots = false         # Display vs save only

[animation]
enabled = true             # Enable/disable animations

[export]
comprehensive = true       # Export all data
formats = ["csv", "json"]  # Output formats

[logging]
log_to_file = true        # Save logs to file
structured = true          # JSON Lines logs
performance = true         # Performance CSV
```

---

## Summary

The unified `outputs/` directory provides:

✅ **Centralized Storage**: All outputs in one location  
✅ **Organized Structure**: Clear subdirectories by type  
✅ **Multiple Formats**: CSV, JSON, PNG, GIF  
✅ **Comprehensive**: Data, plots, results, logs  
✅ **Timestamped**: Automatic experiment tracking  
✅ **Accessible**: Easy programmatic and manual access  
✅ **Shareable**: Simple archival and distribution  

For implementation details, see `run.jl` (orchestration) and `src/utils.jl` (export agents).

