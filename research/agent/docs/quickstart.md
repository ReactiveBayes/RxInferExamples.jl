# Quick Start Guide

Get started with the Generic Agent-Environment Framework in 5 minutes.

## 1. Setup

```bash
cd research/agent
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## 2. Run Config-Driven Simulation

```bash
julia run.jl simulate
```

This runs the configuration specified in `config.toml` (default: Mountain Car).

## 3. Run Explicit Examples

```bash
# Mountain car example
julia examples/mountain_car.jl

# Simple 1D navigation example
julia examples/simple_nav.jl
```

## 4. Run Tests

```bash
julia --project test/runtests.jl
```

## 5. Try Different Configurations

Edit `config.toml`:

```toml
[agent]
type = "SimpleNavAgent"  # Switch to simple navigation

[environment]
type = "SimpleNavEnv"

[simulation]
max_steps = 30
```

Then run again:

```bash
julia run.jl simulate
```

## Understanding the Framework

### Type System

The framework uses strongly-typed vectors:

```julia
StateVector{2}([0.0, 0.0])        # 2D state
ActionVector{1}([0.5])             # 1D action
ObservationVector{2}([0.1, 0.2])  # 2D observation
```

Type parameters ensure agents and environments have matching dimensions.

### Agent-Environment Loop

```julia
# Create environment
env = MountainCarEnv(initial_position = -0.5)

# Create agent
env_params = get_observation_model_params(env)
agent = MountainCarAgent(
    horizon = 20,
    goal_state = StateVector{2}([0.5, 0.0]),
    initial_state = StateVector{2}([-0.5, 0.0]),
    env_params
)

# Run simulation
config = SimulationConfig(max_steps = 50)
result = run_simulation(agent, env, config)
```

### Adding Custom Agent

1. Create new file in `src/agents/my_agent.jl`
2. Define `@model` at top level
3. Implement agent struct and interface methods
4. Add to factory in `src/config.jl`

### Adding Custom Environment

1. Create new file in `src/environments/my_env.jl`
2. Implement environment struct and interface methods
3. Add to factory in `src/config.jl`

## Common Tasks

### Change Planning Horizon

```toml
[agent]
horizon = 30  # Increase from 20 to 30
```

### Enable Verbose Logging

```toml
[simulation]
verbose = true
log_interval = 5  # Log every 5 steps
```

### Disable Diagnostics (Faster)

```toml
[simulation]
enable_diagnostics = false
```

### View Configuration

```bash
julia run.jl config
```

## Next Steps

- Read [README.md](README.md) for architecture details
- Study `examples/mountain_car.jl` for complete implementation
- Review test suite for usage patterns
- Create your own agent-environment pair!

## Troubleshooting

### Package Installation Issues

```bash
rm Manifest.toml
julia --project -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

### Type Mismatch Errors

Ensure agent and environment have matching dimensions:
- Mountain Car: `{2,1,2}` (2D state, 1D action, 2D obs)
- Simple Nav: `{1,1,1}` (1D state, 1D action, 1D obs)

### RxInfer Model Errors

Models must be defined at top level (not inside functions).
See `src/agents/mountain_car_agent.jl` for example.

---

**You're ready to build Active Inference agents!** 🚀

