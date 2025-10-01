# Coin Toss Model - Agent and Component Architecture

## Overview

This document provides a comprehensive technical reference for all components, modules, and "agents" in the Coin Toss Model research fork. Each component is designed with clear responsibilities, interfaces, and interactions.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   run.jl     │  │simple_demo.jl│  │   CLI Args   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │      Configuration Layer            │
          │  ┌────────────┐  ┌──────────────┐  │
          │  │ config.jl  │  │ config.toml  │  │
          │  └─────┬──────┘  └──────┬───────┘  │
          └────────┼─────────────────┼──────────┘
                   │                 │
          ┌────────┴─────────────────┴──────────┐
          │         Core Module Layer           │
          │  ┌──────────────────────────────┐  │
          │  │      CoinTossModel           │  │
          │  │  - Data Generation Agent     │  │
          │  │  - Model Definition Agent    │  │
          │  │  - Analytical Solver Agent   │  │
          │  └──────────┬───────────────────┘  │
          │             │                       │
          │  ┌──────────┴───────────────────┐  │
          │  │    CoinTossInference         │  │
          │  │  - RxInfer Execution Agent   │  │
          │  │  - Diagnostics Agent         │  │
          │  │  - Convergence Monitor Agent │  │
          │  └──────────┬───────────────────┘  │
          │             │                       │
          │  ┌──────────┴───────────────────┐  │
          │  │   CoinTossVisualization      │  │
          │  │  - Plotting Agent            │  │
          │  │  - Animation Agent           │  │
          │  │  - Theme Manager Agent       │  │
          │  └──────────┬───────────────────┘  │
          │             │                       │
          │  ┌──────────┴───────────────────┐  │
          │  │     CoinTossUtils            │  │
          │  │  - Logging Agent             │  │
          │  │  - Export Agent              │  │
          │  │  - Statistics Agent          │  │
          │  └──────────────────────────────┘  │
          └─────────────┬───────────────────────┘
                        │
          ┌─────────────┴─────────────────────┐
          │        Output Layer               │
          │  ┌────────────────────────────┐  │
          │  │  outputs/                  │  │
          │  │  ├── data/                 │  │
          │  │  ├── plots/                │  │
          │  │  ├── animations/           │  │
          │  │  ├── results/              │  │
          │  │  └── logs/                 │  │
          │  └────────────────────────────┘  │
          └───────────────────────────────────┘
```

---

## Component Catalog

### 1. Configuration Agents

#### 1.1 Configuration Loader Agent
**Module**: `Config` in `config.jl`  
**Primary Function**: `load_config(config_file::String)`

**Responsibilities**:
- Parse TOML configuration files
- Provide default configuration fallback
- Merge CLI arguments with file configuration
- Expose configuration to all modules

**Inputs**:
- `config.toml` file path
- Default configuration dictionary

**Outputs**:
- Validated configuration dictionary

**Key Methods**:
```julia
load_config(config_file::String = "config.toml") -> Dict
get_default_config() -> Dict
get_config() -> Dict
```

**Dependencies**: TOML.jl, Dates

#### 1.2 Configuration Validator Agent
**Module**: `Config` in `config.jl`  
**Primary Function**: `validate_config(config::Dict)`

**Responsibilities**:
- Validate all parameter ranges
- Check type consistency
- Report configuration issues
- Ensure parameter compatibility

**Validation Rules**:
- `n_samples` > 0
- `theta_real` ∈ [0, 1]
- `prior_a`, `prior_b` > 0
- `credible_interval` ∈ (0, 1)
- `iterations` > 0

**Outputs**:
- List of validation issues (empty if valid)

---

### 2. Data Generation Agents

#### 2.1 Synthetic Data Generator Agent
**Module**: `CoinTossModel` in `src/model.jl`  
**Primary Function**: `generate_coin_data()`

**Responsibilities**:
- Generate synthetic coin toss observations
- Ensure reproducibility via seeded RNG
- Package data with metadata
- Validate generation parameters

**Algorithm**:
```julia
1. Initialize MersenneTwister(seed)
2. Sample from Bernoulli(θ_real) n times
3. Convert to Float64 array
4. Package with CoinData struct
```

**Inputs**:
- `n::Int`: Number of tosses
- `theta_real::Float64`: True bias parameter
- `seed::Int`: Random seed

**Outputs**:
- `CoinData` struct containing:
  - `observations::Vector{Float64}`
  - `theta_real::Float64`
  - `n_samples::Int`
  - `seed::Int`
  - `timestamp::String`

**Data Contract**:
- All observations ∈ {0.0, 1.0}
- Reproducible for same seed
- Metadata tracks provenance

---

### 3. Model Definition Agents

#### 3.1 Probabilistic Model Agent
**Module**: `CoinTossModel` in `src/model.jl`  
**Primary Function**: `@model coin_model(y, a, b)`

**Responsibilities**:
- Define Beta-Bernoulli factor graph
- Specify prior and likelihood distributions
- Enable RxInfer inference

**Model Structure**:
```julia
θ ~ Beta(a, b)              # Prior
y[i] ~ Bernoulli(θ)         # Likelihood (IID)
```

**Inputs**:
- `y`: Observation vector
- `a::Float64`: Beta prior α parameter
- `b::Float64`: Beta prior β parameter

**Outputs**:
- RxInfer-compatible factor graph model

**Mathematical Properties**:
- Conjugate prior-likelihood pair
- Analytical posterior available
- Closed-form marginal likelihood

#### 3.2 Analytical Solver Agent
**Module**: `CoinTossModel` in `src/model.jl`  
**Primary Function**: `analytical_posterior()`

**Responsibilities**:
- Compute conjugate posterior analytically
- Provide ground truth for validation
- Enable rapid parameter exploration

**Algorithm**:
```julia
Given: data, prior_a, prior_b
n_heads = sum(data)
n_tails = length(data) - n_heads
posterior = Beta(prior_a + n_heads, prior_b + n_tails)
```

**Properties**:
- Exact solution (no approximation)
- Instant computation
- Useful for validation

#### 3.3 Posterior Statistics Agent
**Module**: `CoinTossModel` in `src/model.jl`  
**Primary Function**: `posterior_statistics()`

**Responsibilities**:
- Compute posterior mean, mode, variance
- Calculate credible intervals
- Extract distribution parameters

**Outputs**:
```julia
Dict(
    "mean" => Float64,
    "mode" => Float64,
    "variance" => Float64,
    "std" => Float64,
    "credible_interval" => (lower, upper),
    "credible_level" => Float64,
    "alpha" => Float64,
    "beta" => Float64
)
```

---

### 4. Inference Execution Agents

#### 4.1 RxInfer Execution Agent
**Module**: `CoinTossInference` in `src/inference.jl`  
**Primary Function**: `run_inference()`

**Responsibilities**:
- Execute variational inference via RxInfer
- Track free energy evolution
- Monitor convergence
- Time execution
- Package comprehensive results

**Execution Flow**:
```julia
1. Initialize timer
2. Call RxInfer.infer() with:
   - Factor graph model
   - Observed data
   - Inference parameters
3. Extract posterior distribution
4. Compute diagnostics
5. Check convergence
6. Return InferenceResult
```

**Inputs**:
- `data::Vector{Float64}`: Observations
- `prior_a`, `prior_b`: Prior parameters
- `iterations::Int`: Inference iterations
- `track_fe::Bool`: Track free energy
- `convergence_check::Bool`: Monitor convergence
- `convergence_tol::Float64`: Tolerance
- `showprogress::Bool`: Display progress

**Outputs**:
- `InferenceResult` struct with:
  - `posterior::Beta`
  - `prior::Beta`
  - `observations::Vector{Float64}`
  - `free_energy::Union{Vector{Float64}, Nothing}`
  - `execution_time::Float64`
  - `iterations::Int`
  - `converged::Bool`
  - `convergence_iteration::Union{Int, Nothing}`
  - `diagnostics::Dict{String, Any}`

#### 4.2 Diagnostics Agent
**Module**: `CoinTossInference` in `src/inference.jl`  
**Primary Function**: `compute_inference_diagnostics()`

**Responsibilities**:
- Compute KL divergence
- Calculate expected log-likelihood
- Track information gain
- Monitor variance reduction
- Analyze free energy trajectory

**Diagnostic Metrics**:
```julia
- n_observations: Sample size
- n_heads, n_tails: Counts
- empirical_rate: Observed proportion
- kl_divergence: KL(posterior || prior)
- information_gain: Bits learned from data
- expected_log_likelihood: E_q[log p(data|θ)]
- posterior_mean, variance, mode: Distribution stats
- prior_mean, variance: Prior stats
- mean_shift: Posterior - Prior mean
- variance_reduction: Prior - Posterior variance
- final_free_energy: Terminal free energy
- free_energy_reduction: Total reduction
- mean_fe_change: Average iteration change
```

**Analytical Methods**:
- **KL Divergence**: Exact formula using digamma functions
- **Expected Log-Likelihood**: Analytical expectation under Beta posterior

#### 4.3 Convergence Monitor Agent
**Module**: `CoinTossInference` in `src/inference.jl`  
**Primary Function**: Embedded in `run_inference()`

**Responsibilities**:
- Monitor free energy changes per iteration
- Detect convergence via tolerance threshold
- Report convergence iteration
- Warn if not converged

**Algorithm**:
```julia
For each iteration i from 2 to N:
    fe_change = |FE[i] - FE[i-1]|
    if fe_change < tolerance:
        Mark as converged at iteration i
        Break
```

**Outputs**:
- `converged::Bool`
- `convergence_iteration::Union{Int, Nothing}`

#### 4.4 Posterior Predictive Agent
**Module**: `CoinTossInference` in `src/inference.jl`  
**Primary Function**: `posterior_predictive_check()`

**Responsibilities**:
- Generate predictive samples from posterior
- Compute predictive statistics
- Enable model validation

**Algorithm**:
```julia
1. Sample θ_samples from posterior Beta distribution
2. For each θ in θ_samples:
     Sample outcome from Bernoulli(θ)
3. Compute predictive statistics
```

**Outputs**:
```julia
Dict(
    "theta_samples" => Vector{Float64},
    "predictive_samples" => Vector{Int},
    "predictive_mean" => Float64,
    "predictive_variance" => Float64,
    "pp_prob_heads" => Float64,
    "n_samples" => Int
)
```

---

### 5. Visualization Agents

#### 5.1 Theme Manager Agent
**Module**: `CoinTossVisualization` in `src/visualization.jl`  
**Primary Function**: `get_theme_colors()`

**Responsibilities**:
- Provide color schemes for different themes
- Support accessibility (colorblind-friendly)
- Ensure visual consistency

**Themes**:
1. **Default**: High-contrast standard colors
2. **Dark**: Dark background with bright colors
3. **Colorblind**: Scientifically-validated color palette

**Color Assignments**:
```julia
- background: Plot background
- prior: Prior distribution color
- posterior: Posterior distribution color
- data: Observed data color
- true_value: Ground truth marker color
- grid: Grid line color
```

#### 5.2 Plotting Agent
**Module**: `CoinTossVisualization` in `src/visualization.jl`  
**Primary Functions**: `plot_prior_posterior()`, `plot_credible_interval()`, etc.

**Responsibilities**:
- Create static diagnostic plots
- Apply theme consistently
- Add informative annotations
- Save plots to disk

**Plot Types**:

1. **Prior-Posterior Comparison**
   - Function: `plot_prior_posterior()`
   - Shows: Prior and posterior PDFs with true value
   - Annotations: Mean, mode, true θ

2. **Credible Interval**
   - Function: `plot_credible_interval()`
   - Shows: Posterior with highlighted CI region
   - Annotations: CI bounds, coverage check

3. **Data Histogram**
   - Function: `plot_data_histogram()`
   - Shows: Observed heads/tails counts
   - Annotations: Proportions, true θ

4. **Posterior Predictive**
   - Function: `plot_predictive()`
   - Shows: Observed vs predicted proportions
   - Annotations: Standard errors

5. **Free Energy Convergence**
   - Function: `plot_convergence()`
   - Shows: Free energy trajectory
   - Annotations: Final value

6. **Comprehensive Dashboard**
   - Function: `plot_comprehensive_dashboard()`
   - Shows: Multi-panel overview
   - Layout: 2×2 or 3×2 grid

#### 5.3 Animation Agent
**Module**: `CoinTossVisualization` in `src/visualization.jl`  
**Primary Function**: `create_inference_animation()`

**Responsibilities**:
- Create sequential Bayesian update animations
- Show posterior evolution with data
- Generate GIF files

**Algorithm**:
```julia
For each sample size n in [10, 25, 50, 100, 200, 500]:
    1. Compute posterior with first n observations
    2. Create plot showing:
       - Prior (faded)
       - Current posterior (highlighted)
       - True value
       - Statistics annotation
    3. Add frame to animation
Save as GIF with specified FPS
```

**Features**:
- Consistent y-axis scaling across frames
- Progressive uncertainty reduction visualization
- Statistics overlay per frame

---

### 6. Utility Agents

#### 6.1 Logging Agent
**Module**: `CoinTossUtils` in `src/utils.jl`  
**Primary Function**: `setup_logging()`

**Responsibilities**:
- Configure multi-format logging
- Control verbosity levels
- Route logs to files
- Structure log messages

**Logging Formats**:
1. **Console**: Human-readable, verbosity-controlled
2. **File**: Persistent text logs
3. **Structured (JSON Lines)**: Machine-parseable
4. **Performance (CSV)**: Timing metrics

**Log Levels**:
- `Info`: Standard progress messages
- `Warn`: Configuration issues, non-convergence
- `Error`: Fatal errors
- `Debug`: Detailed diagnostic information

#### 6.2 Timer Agent
**Module**: `CoinTossUtils` in `src/utils.jl`  
**Primary Type**: `Timer` struct

**Responsibilities**:
- Time code blocks
- Log elapsed time automatically
- Support nested timing

**Usage Pattern**:
```julia
timer = Timer("operation_name")
# ... do work ...
elapsed = close(timer)  # Logs automatically
```

**Outputs**:
- Log message: `"PERF operation_name elapsed_seconds=X.XXXX"`
- Return value: Elapsed seconds

#### 6.3 Export Agent
**Module**: `CoinTossUtils` in `src/utils.jl`  
**Primary Functions**: `export_to_csv()`, `export_to_json()`, `save_experiment_results()`

**Responsibilities**:
- Export data in multiple formats
- Flatten nested dictionaries for CSV
- Create timestamped result bundles
- Ensure directory structure

**Export Formats**:

1. **CSV Export**
   - Function: `export_to_csv()`
   - Strategy: Flatten nested dicts with dot notation
   - Output: Two-column (key, value) DataFrame

2. **JSON Export**
   - Function: `export_to_json()`
   - Strategy: Preserve nested structure
   - Output: Pretty-printed JSON with indentation

3. **Comprehensive Bundle**
   - Function: `save_experiment_results()`
   - Creates: Timestamped directory with:
     - `results.json`
     - `results.csv`
     - `metadata.json`

**Directory Structure**:
```
outputs/results/coin_toss_bayesian_inference_YYYY-MM-DD_HH-MM-SS/
├── results.json
├── results.csv
└── metadata.json
```

#### 6.4 Statistics Agent
**Module**: `CoinTossUtils` in `src/utils.jl`  
**Primary Functions**: `compute_summary_statistics()`, `bernoulli_confidence_interval()`

**Responsibilities**:
- Compute descriptive statistics
- Calculate confidence intervals
- Provide frequentist comparisons

**Statistics Computed**:
```julia
Summary Statistics:
- mean, median
- std, var
- min, max
- q25, q75 (quartiles)
- n (sample size)

Confidence Intervals:
- Wilson score interval (exact for Bernoulli)
- Asymptotic normal approximation
```

#### 6.5 Progress Tracking Agent
**Module**: `CoinTossUtils` in `src/utils.jl`  
**Primary Type**: `ProgressBar` struct

**Responsibilities**:
- Display progress bars for long operations
- Update progress incrementally
- Finish and clean up display

**Usage**:
```julia
pb = ProgressBar(total_steps, desc="Progress")
for i in 1:total_steps
    # ... do work ...
    update!(pb, i)
end
finish!(pb)
```

---

### 7. Orchestration Agents

#### 7.1 Experiment Runner Agent
**Module**: Main script in `run.jl`  
**Primary Function**: `run_experiment()`

**Responsibilities**:
- Orchestrate complete experimental pipeline
- Coordinate all sub-agents
- Manage experiment lifecycle
- Generate comprehensive outputs

**Pipeline Stages**:

```julia
Stage 1: Data Generation
├── Generate synthetic data
├── Save to CSV
└── Log statistics

Stage 2: Bayesian Inference
├── Run RxInfer
├── Track free energy
├── Check convergence
└── Compute diagnostics

Stage 3: Statistical Analysis
├── Compute analytical posterior
├── Validate against RxInfer
├── Calculate log marginal likelihood
├── Run posterior predictive checks
└── Log comprehensive diagnostics

Stage 4: Visualization
├── Create comprehensive dashboard
├── Generate individual plots
├── Apply theme consistently
└── Save all visualizations

Stage 5: Animation (Optional)
├── Create Bayesian update animation
├── Generate GIF
└── Save to disk

Stage 6: Export Results
├── Bundle all results
├── Export to JSON/CSV
├── Create metadata
└── Log completion
```

**Error Handling**:
- Validates configuration before starting
- Catches and logs errors gracefully
- Provides helpful error messages
- Maintains partial results on failure

#### 7.2 CLI Argument Parser Agent
**Module**: Main script in `run.jl`  
**Primary Function**: `parse_args()`

**Responsibilities**:
- Parse command-line arguments
- Override configuration parameters
- Handle help requests
- Validate argument values

**Supported Arguments**:
```julia
--help, -h          : Show help
--verbose           : Detailed logging
--quiet             : Minimal logging
--no-animation      : Skip animation
--benchmark         : Run benchmarks
--n=N               : Number of tosses
--theta=θ           : True bias
--seed=S            : Random seed
--theme=THEME       : Visualization theme
```

---

## Data Flow Architecture

### Input Data Flow
```
config.toml → Config.load_config() → Validated Config Dict
                                           ↓
CLI Args → parse_args() → Merged Config → Experiment Runner
                                           ↓
User Parameters → generate_coin_data() → CoinData struct
```

### Processing Data Flow
```
CoinData.observations → run_inference() → InferenceResult
                              ↓                ↓
                    Diagnostics Agent   Convergence Monitor
                              ↓                ↓
                       Diagnostic Metrics   Convergence Status
```

### Output Data Flow
```
InferenceResult → Visualization Agents → Plots (PNG)
                                      → Dashboard (PNG)
                                      → Animation (GIF)
                ↓
         Export Agents → JSON (results/)
                      → CSV (results/)
                      → Logs (logs/)
```

---

## Agent Communication Patterns

### 1. Direct Function Calls
Most agents communicate via direct function calls with typed parameters.

**Example**:
```julia
data = generate_coin_data(n=500, theta_real=0.75)
result = run_inference(data.observations, 4.0, 8.0)
```

### 2. Shared Configuration
All agents access global configuration via `Config` module.

**Example**:
```julia
config = load_config()
theme = config["visualization"]["theme"]
colors = get_theme_colors(theme)
```

### 3. Result Passing
Agents pass structured results (structs/dicts) containing all relevant information.

**Example**:
```julia
struct InferenceResult
    posterior::Beta
    diagnostics::Dict{String, Any}
    # ... more fields
end
```

### 4. File-Based Communication
Some agents communicate via file system (outputs/).

**Example**:
```julia
save_plot(dashboard, "outputs/plots/dashboard.png")
# Later: Load and analyze saved plots
```

---

## Agent State Management

### Stateless Agents
Most agents are **stateless** (pure functions):
- Data Generator
- Model Definition
- Analytical Solver
- Plotting functions
- Export functions

**Benefits**:
- Reproducible
- Testable
- Composable
- No hidden dependencies

### Stateful Agents
Some agents maintain state:

1. **Timer Agent**
   - State: `start_time`, `end_time`
   - Mutable for timing measurements

2. **ProgressBar Agent**
   - State: Current progress
   - Mutable for incremental updates

3. **Logger Agent**
   - State: Log file handles
   - Mutable for persistent logging

---

## Testing Strategy for Agents

### Unit Testing
Each agent has dedicated unit tests:

```julia
@testset "Data Generation Agent" begin
    @test generate_coin_data(n=100, theta_real=0.75, seed=42)
    # Test reproducibility, validity, edge cases
end

@testset "Inference Agent" begin
    @test run_inference(data, 4.0, 8.0)
    # Test convergence, diagnostics, timing
end
```

### Integration Testing
Test agent interactions:

```julia
@testset "End-to-End Workflow" begin
    # Generate data
    data = generate_coin_data(n=200, theta_real=0.65)
    
    # Run inference
    result = run_inference(data.observations, 2.0, 3.0)
    
    # Verify interactions
    @test result.posterior isa Beta
    @test haskey(result.diagnostics, "empirical_rate")
end
```

---

## Performance Characteristics

### Agent Complexity

| Agent | Time Complexity | Space Complexity | Notes |
|-------|----------------|------------------|-------|
| Data Generator | O(n) | O(n) | n = sample size |
| Analytical Solver | O(1) | O(1) | Closed-form |
| RxInfer Execution | O(k·n) | O(n) | k = iterations |
| Diagnostics | O(n) | O(1) | Linear scan |
| Plotting | O(r) | O(r) | r = resolution |
| Animation | O(f·r) | O(r) | f = frames |
| Export | O(m) | O(m) | m = data size |

### Typical Performance
- **Data Generation**: < 0.01s (500 samples)
- **Inference**: < 0.1s (10 iterations)
- **Diagnostics**: < 0.001s
- **Visualization**: < 2s (all plots)
- **Animation**: < 5s (6 frames)
- **Export**: < 0.1s

---

## Agent Extension Guide

### Adding a New Analysis Agent

1. **Create Module**:
```julia
module NewAnalysis
using Distributions
export new_analysis_function

function new_analysis_function(posterior::Beta, data::Vector{Float64})
    # Your analysis logic
    return results
end

end
```

2. **Add to Pipeline** (`run.jl`):
```julia
# In run_experiment()
new_results = new_analysis_function(
    inference_result.posterior,
    coin_data.observations
)
experiment_results["results"]["new_analysis"] = new_results
```

3. **Add Configuration** (`config.toml`):
```toml
[new_analysis]
parameter1 = value1
parameter2 = value2
```

4. **Add Tests** (`test/runtests.jl`):
```julia
@testset "New Analysis Agent" begin
    @test new_analysis_function(posterior, data)
end
```

---

## Agent Dependencies Graph

```
CoinTossModel (Base)
    ├── RxInfer (model definition)
    ├── Distributions (Beta, Bernoulli)
    └── SpecialFunctions (logbeta)

CoinTossInference
    ├── CoinTossModel (uses coin_model)
    ├── RxInfer (inference execution)
    └── SpecialFunctions (digamma, logbeta)

CoinTossVisualization
    ├── Plots (all visualization)
    ├── StatsPlots (statistical plots)
    └── Distributions (distribution functions)

CoinTossUtils
    ├── DataFrames (data export)
    ├── CSV (CSV export)
    ├── JSON (JSON export)
    ├── ProgressMeter (progress bars)
    └── Statistics (summary stats)

Config
    ├── TOML (config parsing)
    └── Dates (timestamps)
```

---

## Unified Output Structure

All agents write to a **single unified outputs/ directory**:

```
outputs/
├── data/                           # Data Agent outputs
│   └── coin_toss_observations.csv
├── plots/                          # Plotting Agent outputs
│   ├── comprehensive_dashboard.png
│   ├── prior_posterior.png
│   ├── credible_interval.png
│   ├── data_histogram.png
│   ├── posterior_predictive.png
│   └── free_energy_convergence.png
├── animations/                     # Animation Agent outputs
│   └── bayesian_update.gif
├── results/                        # Export Agent outputs
│   └── coin_toss_bayesian_inference_YYYY-MM-DD_HH-MM-SS/
│       ├── results.json
│       ├── results.csv
│       └── metadata.json
└── logs/                          # Logging Agent outputs
    ├── cointoss.log
    ├── cointoss_structured.jsonl
    └── cointoss_performance.csv
```

**Benefits**:
- Single location for all outputs
- Easy to archive/share
- Clear organizational structure
- No scattered files

---

## Agent Interaction Sequence

### Typical Execution Sequence

```
1. CLI Parser Agent
   ↓ config dict
2. Config Validator Agent
   ↓ validated config
3. Directory Setup Agent (Utils)
   ↓ directory structure
4. Logging Agent
   ↓ configured logging
5. Data Generator Agent
   ↓ CoinData
6. Model Definition Agent (implicit)
   ↓ factor graph
7. RxInfer Execution Agent
   ↓ InferenceResult
8. Diagnostics Agent
   ↓ diagnostic metrics
9. Analytical Solver Agent (validation)
   ↓ analytical posterior
10. Plotting Agent (multiple calls)
    ↓ plot objects
11. Animation Agent (optional)
    ↓ animation object
12. Export Agent
    ↓ saved files
13. Summary Logger Agent
    ↓ final report
```

---

## Agent Best Practices

### 1. Single Responsibility
Each agent has one clear purpose.

### 2. Explicit Interfaces
Clear input/output contracts.

### 3. Error Handling
Validate inputs, handle errors gracefully.

### 4. Logging
Log important operations and timing.

### 5. Documentation
Comprehensive docstrings for all functions.

### 6. Testing
Unit tests for each agent.

### 7. Type Safety
Use type annotations and assertions.

### 8. Modularity
Agents can be used independently.

---

## Summary

The Coin Toss Model implements a **multi-agent architecture** where specialized components handle distinct responsibilities:

- **Configuration Agents**: Load, validate, and manage parameters
- **Data Agents**: Generate and validate synthetic data
- **Model Agents**: Define probabilistic models and compute analytical solutions
- **Inference Agents**: Execute RxInfer, monitor convergence, compute diagnostics
- **Visualization Agents**: Create plots, animations, and dashboards
- **Utility Agents**: Handle logging, export, timing, and statistics
- **Orchestration Agents**: Coordinate the complete experimental pipeline

All agents communicate through **well-defined interfaces**, write to a **unified output structure**, and are **independently testable** while working together to provide comprehensive Bayesian inference capabilities.

For implementation details, see the source code in `src/` and tests in `test/`.

