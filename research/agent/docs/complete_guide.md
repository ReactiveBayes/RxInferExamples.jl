# Complete Framework Guide

**The definitive guide to the Generic Agent-Environment Framework**

**Version:** 0.1.1  
**Date:** October 2, 2025  
**Status:** Production Ready

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Usage Patterns](#usage-patterns)
5. [Visualization](#visualization)
6. [API Reference](#api-reference)
7. [Development](#development)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

The Generic Agent-Environment Framework provides a complete research environment for Active Inference with:

- **Type-Safe Design** - Compile-time dimension checking
- **Real RxInfer Integration** - Actual variational inference
- **Comprehensive Visualization** - Automatic plots and animations
- **Complete Output Management** - Everything saved automatically
- **Modular Architecture** - Easy to extend
- **Production Ready** - Tested and documented

### What Makes This Framework Special

1. **No Mocks** - Real Active Inference with RxInfer.jl
2. **Type Safety** - Catch errors at compile time
3. **Automatic Visualization** - Every run generates plots and animations
4. **Complete Outputs** - Data, diagnostics, reports - all automatic
5. **Well Documented** - Comprehensive guides and examples

---

## Quick Start

### Installation

```bash
# Navigate to framework
cd research/agent

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### First Run

```bash
# Run with full visualization
julia --project=. run.jl simulate

# Check outputs
ls outputs/*/
```

### What You Get

Every run creates a timestamped directory with:
- 📊 Static plots (PNG)
- 🎬 Animations (GIF)
- 📁 Data files (CSV)
- 📄 Diagnostics (JSON)
- 📝 Report (Markdown)

---

## Architecture

### Type System

```julia
StateVector{N}       # N-dimensional state
ActionVector{M}      # M-dimensional action
ObservationVector{K} # K-dimensional observation
```

**Example:**
```julia
# Mountain Car: 2D state, 1D action, 2D observation
agent::AbstractActiveInferenceAgent{2,1,2}
env::AbstractEnvironment{2,1,2}  # Must match!
```

### Components

**Agents** (in `src/agents/`)
- `AbstractActiveInferenceAgent{S,A,O}` - Base interface
- `MountainCarAgent` - Mountain car implementation
- `SimpleNavAgent` - 1D navigation implementation

**Environments** (in `src/environments/`)
- `AbstractEnvironment{S,A,O}` - Base interface
- `MountainCarEnv` - Mountain car physics
- `SimpleNavEnv` - 1D navigation physics

**Infrastructure** (in `src/`)
- `simulation.jl` - Simulation runner with output management
- `visualization.jl` - Plotting and animation
- `diagnostics.jl` - Performance tracking
- `logging.jl` - Multi-format logging
- `config.jl` - Configuration and factories

---

## Usage Patterns

### Pattern 1: Config-Driven

```bash
# Edit config.toml to set parameters
vim config.toml

# Run
julia --project=. run.jl simulate
```

**Pros:** Easy, no coding required  
**Cons:** Limited flexibility

### Pattern 2: Examples

```bash
# Run pre-built examples
julia --project=. examples/mountain_car.jl
julia --project=. examples/simple_nav.jl
```

**Pros:** See working code  
**Cons:** Need to modify for custom use

### Pattern 3: Custom Code

```julia
# Create custom script
using Pkg
Pkg.activate("path/to/agent")

include("src/simulation.jl")
using .Main: StateVector, ActionVector, ObservationVector

# Create agent and environment
env = MountainCarEnv(initial_position=-0.5)
agent = MountainCarAgent(horizon=20, ...)

# Run
config = SimulationConfig(max_steps=50)
result = run_simulation(agent, env, config)

# Save with visualization
save_simulation_outputs(result, "my_exp", goal_state)
```

**Pros:** Full control  
**Cons:** More code to write

---

## Visualization

### Automatic Generation

All simulations automatically create:

**1D State Space:**
- `trajectory_1d.png` - Position and actions over time
- `trajectory_1d.gif` - Animated trajectory

**2D State Space:**
- `trajectory_2d.png` - 4-panel plot (position, velocity, phase space, actions)
- `mountain_car_landscape.png` - Terrain with trajectory overlay
- `trajectory_2d.gif` - 4-panel animated evolution

**Diagnostics:**
- `diagnostics.png` - Memory, inference time, uncertainty

### API Usage

```julia
# Generate all visualizations
generate_all_visualizations(result, output_dir, state_dim)

# Individual plots
plot_trajectory_1d(result, output_dir)
plot_trajectory_2d(result, output_dir)
plot_mountain_car_landscape(result, output_dir)
plot_diagnostics(result.diagnostics, output_dir)

# Animations
animate_trajectory_1d(result, output_dir, fps=10)
animate_trajectory_2d(result, output_dir, fps=10)
```

### Customization

```julia
# Custom titles
plot_trajectory_2d(result, output_dir, 
                   title="My Custom Title")

# Custom frame rate
animate_trajectory_2d(result, output_dir, fps=20)

# Disable for performance
save_simulation_outputs(
    result, output_dir, goal_state,
    generate_visualizations=false,
    generate_animations=false
)
```

---

## API Reference

### Agent Interface

All agents must implement:

```julia
step!(agent, observation, action)   # Run inference
get_action(agent) -> action          # Get next action
get_predictions(agent) -> states     # Get predictions
slide!(agent)                        # Slide planning window
reset!(agent)                        # Reset to initial state
```

### Environment Interface

All environments must implement:

```julia
step!(env, action) -> observation    # Execute action
reset!(env) -> observation           # Reset to initial
get_state(env) -> state             # Get current state
get_observation_model_params(env)    # Provide params for agent
```

### Simulation

```julia
# Configure
config = SimulationConfig(
    max_steps = 100,
    enable_diagnostics = true,
    enable_logging = true,
    verbose = true,
    log_interval = 10
)

# Run
result = run_simulation(agent, env, config)

# Save outputs
save_simulation_outputs(
    result, output_dir, goal_state,
    generate_visualizations=true,
    generate_animations=true
)
```

---

## Development

### Adding a New Agent

1. **Define RxInfer model** at top level:

```julia
@model function my_agent_model(...)
    # Generative model
end
```

2. **Create agent struct:**

```julia
mutable struct MyAgent <: AbstractActiveInferenceAgent{S,A,O}
    horizon::Int
    # ... other fields
end
```

3. **Implement interface:**

```julia
function step!(agent::MyAgent, obs, action)
    # Run RxInfer inference
end

function get_action(agent::MyAgent)
    # Extract action from posteriors
end

# ... other required methods
```

4. **Add to factory** in `src/config.jl`

### Adding a New Environment

1. **Create environment struct:**

```julia
mutable struct MyEnv <: AbstractEnvironment{S,A,O}
    current_state::Ref{StateVector{S}}
    # ... other fields
end
```

2. **Implement interface:**

```julia
function step!(env::MyEnv, action::ActionVector{A})
    # Update physics
    # Return observation
end

function reset!(env::MyEnv)
    # Reset state
    # Return initial observation
end

# ... other required methods
```

3. **Add to factory** in `src/config.jl`

---

## Troubleshooting

### "Visualization generation failed"

**Cause:** Missing Plots.jl

**Solution:**
```bash
julia --project=. -e 'using Pkg; Pkg.add("Plots")'
```

### "using expression not at top level"

**Cause:** `using` statement inside function

**Solution:** Move to module level or use qualified calls

### Empty plots/animations directories

**Cause:** Visualization not imported

**Solution:** Ensure `using Plots` at top level

### Type mismatch errors

**Cause:** Agent and environment dimensions don't match

**Solution:** Verify `{S,A,O}` parameters match

---

## Documentation Map

| Document | Purpose | Audience |
|----------|---------|----------|
| [README.md](../README.md) | Framework overview | All |
| [QUICKSTART.md](QUICKSTART.md) | 5-minute guide | New |
| [index.md](index.md) | API reference | All |
| [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) | Plotting guide | All |
| [COMPREHENSIVE_SUMMARY.md](COMPREHENSIVE_SUMMARY.md) | Complete overview | All |
| [ENHANCEMENTS_SUMMARY.md](ENHANCEMENTS_SUMMARY.md) | v0.1.1 changes | Existing |
| [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) | Implementation | Developers |
| [WORKING_STATUS.md](WORKING_STATUS.md) | Current status | All |
| [VISUALIZATION_FIX.md](VISUALIZATION_FIX.md) | Troubleshooting | All |
| [NAVIGATION.md](NAVIGATION.md) | Doc navigation | All |

---

## Best Practices

### For Research

1. **Use timestamped outputs** - Each run in separate directory
2. **Enable diagnostics** - Track performance and memory
3. **Generate visualizations** - Visual debugging is powerful
4. **Save complete outputs** - You'll want the data later

### For Development

1. **Use type system** - Catch errors at compile time
2. **Write tests** - Test each component
3. **Document code** - Docstrings for public functions
4. **Follow patterns** - Use existing agents/envs as templates

### For Performance

1. **Disable viz for batch** - Run many experiments without plots
2. **Reduce horizon** - Shorter planning = faster inference
3. **Profile first** - Measure before optimizing
4. **Use diagnostics** - Track what's slow

---

## Examples

### Minimal Example

```julia
# Load framework
include("src/simulation.jl")
using .Main: StateVector

# Create simple setup
env = SimpleNavEnv()
agent = SimpleNavAgent(10, StateVector{1}([1.0]), ...)

# Run short simulation
config = SimulationConfig(max_steps=10)
result = run_simulation(agent, env, config)

# Quick plot
using Plots
plot([s[1] for s in result.states])
```

### Full Example

```julia
# Complete workflow
using Dates

# Setup
env = MountainCarEnv(initial_position=-0.5)
goal_state = StateVector{2}([0.5, 0.0])
agent = MountainCarAgent(20, goal_state, ...)

# Configure
config = SimulationConfig(
    max_steps=100,
    enable_diagnostics=true,
    verbose=true
)

# Run
result = run_simulation(agent, env, config)

# Save everything
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
output_dir = "outputs/my_experiment_$timestamp"
save_simulation_outputs(result, output_dir, goal_state)

# Analyze
using CSV
trajectory = CSV.read("$output_dir/data/trajectory.csv", DataFrame)
plot(trajectory.step, trajectory.position)
```

---

## Summary

The Generic Agent-Environment Framework is:

✅ **Type-Safe** - Compile-time checks  
✅ **Real AI** - Actual Active Inference  
✅ **Visual** - Automatic plots and animations  
✅ **Complete** - Full output management  
✅ **Modular** - Easy to extend  
✅ **Tested** - Comprehensive test suite  
✅ **Documented** - Complete guides  
✅ **Production-Ready** - Use for research now  

---

**Ready to use: `julia --project=. run.jl simulate`** 🚀

---

**Framework Version:** 0.1.1  
**Document Version:** 1.0  
**Last Updated:** October 2, 2025

