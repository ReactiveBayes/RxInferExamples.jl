# Generic Agent-Environment Framework Documentation

## Quick Links

- [Main README](../README.md)
- [Quick Start Guide](../QUICKSTART.md)
- [Implementation Summary](../IMPLEMENTATION_SUMMARY.md)

## Core Components

### Type System
- **[types.jl](../src/types.jl)**: `StateVector`, `ActionVector`, `ObservationVector` with compile-time dimension checking

### Abstract Interfaces
- **[abstract_agent.jl](../src/agents/abstract_agent.jl)**: `AbstractActiveInferenceAgent{S,A,O}` interface
- **[abstract_environment.jl](../src/environments/abstract_environment.jl)**: `AbstractEnvironment{S,A,O}` interface

### Agents
- **[mountain_car_agent.jl](../src/agents/mountain_car_agent.jl)**: Mountain Car Active Inference agent with RxInfer
- **[simple_nav_agent.jl](../src/agents/simple_nav_agent.jl)**: Simple 1D navigation agent

### Environments
- **[mountain_car_env.jl](../src/environments/mountain_car_env.jl)**: Mountain Car physics environment
- **[simple_nav_env.jl](../src/environments/simple_nav_env.jl)**: Simple 1D navigation environment

### Infrastructure
- **[simulation.jl](../src/simulation.jl)**: Generic simulation runner with comprehensive output saving
- **[config.jl](../src/config.jl)**: Configuration loading and factory functions
- **[diagnostics.jl](../src/diagnostics.jl)**: Comprehensive diagnostics tracking
- **[logging.jl](../src/logging.jl)**: Multi-format logging system
- **[visualization.jl](../src/visualization.jl)**: Plotting and animation generation

## Usage Guides

### Running Simulations

#### Config-Driven Approach
```bash
cd research/agent
julia run.jl simulate
```

#### Explicit Examples
```bash
julia examples/mountain_car.jl
julia examples/simple_nav.jl
```

### Configuration

Edit `config.toml` to customize:
- Agent type and parameters
- Environment setup
- Simulation settings
- Output directories

### Testing

```bash
julia --project test/runtests.jl
```

### Visualization

All simulations automatically generate visualizations and animations:

**Static Plots:**
- Trajectory plots (position, velocity, actions over time)
- Phase space plots (state space visualization)
- Mountain car landscape (terrain with trajectory overlay)
- Diagnostics plots (memory, inference time, uncertainty)

**Animations:**
- Animated GIF trajectories showing real-time evolution
- Multi-panel animations with state, action, and phase space

**Outputs are saved in run-specific directories:**
```
outputs/mountaincar_20251002_140530/
├── plots/              # PNG visualizations
├── animations/         # GIF animations
├── data/              # CSV data files
├── diagnostics/       # JSON diagnostics
├── results/           # Summary statistics
└── REPORT.md          # Comprehensive markdown report
```

See [VISUALIZATION_GUIDE.md](../VISUALIZATION_GUIDE.md) for details.

## API Reference

### Agent Interface

All agents must implement:

```julia
step!(agent, observation, action)  # Update beliefs via RxInfer inference
get_action(agent)                  # Get next action based on beliefs
get_predictions(agent)             # Get predicted future states
slide!(agent)                      # Slide planning horizon forward
reset!(agent)                      # Reset to initial state
```

### Environment Interface

All environments must implement:

```julia
step!(env, action)                # Execute action, return observation
reset!(env)                       # Reset to initial state
get_state(env)                    # Get current internal state
get_observation_model_params(env) # Provide parameters for agent's model
```

## Architecture Diagrams

### Agent-Environment Loop

```
┌─────────────────────────────────────────────────────┐
│                 Simulation Loop                      │
│                                                      │
│  ┌──────────┐         ┌────────────┐               │
│  │          │ action  │            │               │
│  │  Agent   ├────────>│Environment │               │
│  │          │         │            │               │
│  │          │<────────┤            │               │
│  └──────────┘  obs    └────────────┘               │
│       │                                              │
│       │ RxInfer                                      │
│       │ Inference                                    │
│       v                                              │
│  ┌──────────┐                                       │
│  │ Beliefs  │                                       │
│  │ (states) │                                       │
│  └──────────┘                                       │
└─────────────────────────────────────────────────────┘
```

### Type Safety

```
AbstractActiveInferenceAgent{S,A,O}
                  ↓
         MountainCarAgent{2,1,2}
                  ↓
           Compatible with
                  ↓
          MountainCarEnv{2,1,2}
                  ↓
      AbstractEnvironment{S,A,O}
```

Type parameters ensure dimension compatibility at compile time.

## Adding Components

### Creating a New Agent

1. Define RxInfer `@model` at top level:
```julia
@model function my_agent_model(...)
    # Generative model for agent
end
```

2. Create agent struct:
```julia
mutable struct MyAgent <: AbstractActiveInferenceAgent{S,A,O}
    # Agent fields
end
```

3. Implement required interface methods
4. Add factory function to `config.jl`

### Creating a New Environment

1. Create environment struct:
```julia
mutable struct MyEnv <: AbstractEnvironment{S,A,O}
    current_state::Ref{StateVector{S}}
    # Other fields
end
```

2. Implement required interface methods
3. Add factory function to `config.jl`

## Performance Considerations

- **StaticArrays**: Type system uses `SVector` for efficiency
- **Diagnostics**: Can be disabled for faster execution
- **Logging**: Multiple levels available (disable verbose for speed)
- **Planning Horizon**: Longer horizons = more computation

## Troubleshooting

### Common Issues

**Type Mismatch**
- Ensure agent and environment have matching `{S,A,O}` dimensions

**RxInfer Errors**
- Models must be defined at top level
- Check that observation model parameters are correctly passed

**Performance**
- Disable diagnostics with `enable_diagnostics = false`
- Reduce planning horizon
- Use shorter simulations for testing

### Debug Mode

Enable verbose logging:
```toml
[simulation]
verbose = true
log_interval = 1
```

## References

- **RxInfer.jl**: [https://github.com/biaslab/RxInfer.jl](https://github.com/biaslab/RxInfer.jl)
- **StaticArrays.jl**: [https://github.com/JuliaArrays/StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
- **Active Inference**: Friston, K. et al. "Active Inference: A Process Theory"

## Contributing

See [README.md](../README.md) for contribution guidelines.

## License

Part of RxInferExamples.jl - same license applies.

