# Generic Agent-Environment Framework

A fully-typed, modular framework for Active Inference agents with comprehensive visualization and output management.

**Version:** 0.1.1  
**Status:** ✅ Production Ready & Confirmed Working  
**Updated:** October 2, 2025  
**Last Test:** 250-step run successful (Oct 2, 2025 14:11 PDT)

---

## Overview

This framework provides a complete research environment for Active Inference with:

- **Strong Type Safety** - Compile-time dimension checking via `StateVector{N}`, `ActionVector{M}`, `ObservationVector{K}`
- **Real RxInfer Integration** - Actual variational inference with message passing
- **Comprehensive Visualization** - Automatic plots and animations for all simulations  
- **Complete Output Management** - Automatic data, diagnostics, and report generation
- **Config-Driven** - Runtime selection of agent-environment combinations via TOML
- **Modular Design** - Easy to add new agents and environments

---

## ✨ New in v0.1.1: Complete Visualization Suite

Every simulation now automatically generates:
- 📊 **Static Plots** - Trajectory, phase space, landscape, diagnostics
- 🎬 **Animations** - Animated GIFs showing real-time evolution
- 📁 **Complete Data** - CSV trajectories, JSON diagnostics, metadata
- 📝 **Reports** - Comprehensive markdown reports with all metrics

---

## Quick Start

```bash
# Navigate to framework
cd research/agent

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run with full visualization
julia --project=. run.jl simulate

# Or run examples
julia --project=. examples/mountain_car.jl
julia --project=. examples/simple_nav.jl

# Check outputs
ls outputs/*/
```

---

## Example Output

Each simulation creates a timestamped directory:

```
outputs/mountaincar_20251002_140530/
├── REPORT.md                      # Comprehensive report
├── metadata.json                  # Run configuration
├── plots/                        # Static visualizations (PNG)
│   ├── trajectory_2d.png
│   ├── mountain_car_landscape.png
│   └── diagnostics.png
├── animations/                   # Animated visualizations (GIF)
│   └── trajectory_2d.gif
├── data/                         # Raw data (CSV)
│   ├── trajectory.csv
│   └── observations.csv
├── diagnostics/                  # Performance metrics (JSON)
│   ├── diagnostics.json
│   └── performance.json
└── results/                      # Summary statistics (CSV)
    └── summary.csv
```

---

## Documentation

### 📚 Framework Status

| Document | Description |
|----------|-------------|
| **[status.md](status.md)** | Current framework status (quick reference) |

### 📖 Complete Documentation

| Document | Description |
|----------|-------------|
| **[Quick Start](docs/quickstart.md)** | 5-minute getting started guide |
| **[Complete Guide](docs/complete_guide.md)** | Comprehensive framework guide |
| **[Generic Agent Interface](docs/generic_agent_interface.md)** | Composability and interface design |
| **[API Reference](docs/index.md)** | Complete API documentation |
| **[Visualization Guide](docs/visualization_guide.md)** | Plotting and animation guide |
| **[Documentation Index](docs/README.md)** | Complete documentation navigation |
| **[Comprehensive Summary](docs/comprehensive_summary.md)** | Framework overview and capabilities |
| **[Enhancements Summary](docs/enhancements_summary.md)** | v0.1.1 enhancements details |
| **[Implementation Details](docs/implementation_complete.md)** | Full implementation report |
| **[Visualization Setup](docs/visualization_fix.md)** | Setup and troubleshooting |
| **[Working Status](docs/working_status.md)** | Detailed status and verification |
| **[Output Verification](docs/output_verification.md)** | Output structure and verification |
| **[Navigation Guide](docs/navigation.md)** | Documentation navigation help |

### 🎯 Quick Links

- **New Users**: Start with [Quick Start](docs/quickstart.md)
- **Current Status**: See [status.md](status.md)
- **Complete Guide**: Read [Complete Guide](docs/complete_guide.md)
- **API Reference**: Check [docs/index.md](docs/index.md)
- **Troubleshooting**: See [Visualization Setup](docs/visualization_fix.md)

---

## Key Features

✅ **Type Safety** - Compile-time dimension checking prevents mismatched pairs  
✅ **Real RxInfer** - Actual variational message passing  
✅ **Comprehensive Visualization** - Automatic plots, animations, and reports  
✅ **Modular** - Clean separation between agents, environments, and simulation  
✅ **Config-Driven** - Runtime selection without code changes  
✅ **Extensible** - Easy to add new agents and environments  
✅ **Well-Tested** - Comprehensive test suite including visualization tests  
✅ **Production-Ready** - Complete output management for research and publication

---

## Architecture

### Type System

```julia
StateVector{N}       # N-dimensional state
ActionVector{M}      # M-dimensional action
ObservationVector{K} # K-dimensional observation

# Example: Mountain Car (2D state, 1D action, 2D observation)
agent::AbstractActiveInferenceAgent{2,1,2}
env::AbstractEnvironment{2,1,2}  # Must match!
```

### Components

- **Agents** - `MountainCarAgent`, `SimpleNavAgent`
- **Environments** - `MountainCarEnv`, `SimpleNavEnv`
- **Infrastructure** - Simulation, diagnostics, logging, visualization
- **Configuration** - TOML-based runtime configuration

---

## Usage Patterns

### Pattern 1: Config-Driven

```bash
# Edit config.toml to set parameters
julia --project=. run.jl simulate
```

### Pattern 2: Explicit Construction

```julia
# Create environment and agent
env = MountainCarEnv(initial_position = -0.5)
agent = MountainCarAgent(horizon=20, goal_state=..., ...)

# Run simulation
config = SimulationConfig(max_steps=50, enable_diagnostics=true)
result = run_simulation(agent, env, config)

# Save with full visualization
save_simulation_outputs(result, output_dir, goal_state)
```

### Pattern 3: Custom Analysis

```julia
# Load saved data
using CSV, DataFrames, JSON

trajectory = CSV.read("outputs/myrun/data/trajectory.csv", DataFrame)
diagnostics = JSON.parsefile("outputs/myrun/diagnostics/diagnostics.json")
metadata = JSON.parsefile("outputs/myrun/metadata.json")

# Perform custom analysis
plot(trajectory.step, trajectory.position)
```

---

## Testing

```bash
# Run full test suite
julia --project=. test/runtests.jl

# Quick verification
julia --project=. quick_test_visualization.jl
```

---

## Dependencies

- **RxInfer.jl** - Reactive message passing for probabilistic inference
- **StaticArrays.jl** - Efficient fixed-size arrays for type system
- **Plots.jl** - Visualization and animation
- **CSV.jl, DataFrames.jl, JSON.jl** - Data handling
- **TOML.jl** - Configuration file parsing

---

## Project Structure

```
research/agent/
├── README.md                    # This file
├── Project.toml                # Julia project configuration
├── config.toml                 # Runtime configuration
├── run.jl                      # Main runner script
├── quick_test_visualization.jl # Quick verification script
├── test_enhancements.jl        # Enhancement verification
├── src/                        # Source code
│   ├── types.jl               # Type system
│   ├── agents/                # Agent implementations
│   ├── environments/          # Environment implementations
│   ├── simulation.jl          # Simulation runner
│   ├── config.jl              # Configuration loader
│   ├── diagnostics.jl         # Diagnostics system
│   ├── logging.jl             # Logging system
│   └── visualization.jl       # Visualization module
├── examples/                   # Explicit examples
│   ├── mountain_car.jl
│   └── simple_nav.jl
├── test/                       # Test suite
│   ├── runtests.jl
│   ├── test_types.jl
│   ├── test_agents.jl
│   ├── test_environments.jl
│   ├── test_integration.jl
│   └── test_visualization.jl
├── docs/                       # Complete documentation
│   ├── index.md               # API reference
│   ├── QUICKSTART.md
│   ├── VISUALIZATION_GUIDE.md
│   ├── COMPREHENSIVE_SUMMARY.md
│   ├── ENHANCEMENTS_SUMMARY.md
│   ├── IMPLEMENTATION_COMPLETE.md
│   ├── VISUALIZATION_FIX.md
│   ├── WORKING_STATUS.md
│   └── OUTPUT_VERIFICATION.md
└── outputs/                    # Simulation outputs (auto-generated)
    └── README.md              # Outputs directory guide
```

---

## Contributing

See individual documentation files for:
- **Adding Agents**: [docs/index.md](docs/index.md#creating-a-new-agent)
- **Adding Environments**: [docs/index.md](docs/index.md#creating-a-new-environment)
- **Testing**: [test/runtests.jl](test/runtests.jl)

---

## License

Part of RxInferExamples.jl - same license applies.

---

## Acknowledgments

Based on the `car_agent/` framework, refactored for generic agent-environment separation.

**Framework Version:** 0.1.1  
**Last Updated:** October 2, 2025  
**Maintainers:** RxInferExamples Contributors

---

**🎉 Ready for Active Inference Research!**
