# Quick Start Guide

Get started with the Generic Active Inference Agent Framework in 5 minutes.

## 1. Setup

```bash
cd research/car_agent
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## 2. Run the Example

```bash
julia run.jl example
```

This will:
- Create a mountain car Active Inference agent
- Run a 100-step simulation
- Generate comprehensive diagnostics
- Create a visualization
- Save outputs to `outputs/`

## 3. Run the Tests

```bash
julia run.jl test
```

This runs the comprehensive test suite covering:
- Configuration validation
- Agent creation and operation
- Diagnostics and logging
- Integration scenarios
- Edge cases

## 4. View Configuration

```bash
julia run.jl config
```

## 5. Create Your Own Agent

### Minimal Example

```julia
using LinearAlgebra
include("config.jl")
include("src/agent.jl")
using .Agent

# Define your dynamics
transition = (s::AbstractVector) -> [1.0 0.1; 0.0 0.9] * s
control = (u::AbstractVector) -> [0.0; 0.1] * u[1]

# Create agent
agent = GenericActiveInferenceAgent(
    10,  # planning horizon
    2,   # state dimension
    1,   # action dimension
    transition,
    control;
    goal_state = [1.0, 0.0]
)

# Run Active Inference
for t in 1:50
    action = get_action(agent)
    observation = your_environment(action)
    step!(agent, observation, action)
    slide!(agent)
end

# Get results
print_status(agent)
```

### With Full Diagnostics

```julia
include("src/diagnostics.jl")
using .Diagnostics

# Create diagnostics collector
diagnostics = DiagnosticsCollector()

# In your loop
for t in 1:max_steps
    # Time inference
    start_timer!(diagnostics.performance_profiler, "inference")
    step!(agent, observation, action)
    stop_timer!(diagnostics.performance_profiler, "inference")
    
    # Record metrics
    predictions = get_predictions(agent)
    record_belief!(diagnostics.belief_tracker, t,
                  agent.state.state_mean, agent.state.state_cov)
    record_predictions!(diagnostics.prediction_tracker, 
                       predictions, observation)
    
    if t % 10 == 0
        trace_memory!(diagnostics.memory_tracer)
    end
    
    slide!(agent)
end

# Print comprehensive report
print_diagnostics_report(diagnostics)
```

### With Logging

```julia
include("src/logging.jl")
using .LoggingUtils

# Setup logging
loggers = setup_logging(
    verbose = true,
    structured = true,
    performance = true
)

# In your loop
for t in 1:max_steps
    action = get_action(agent)
    observation = your_environment(action)
    
    step!(agent, observation, action)
    
    # Log step
    log_agent_step(t, action, observation,
                   free_energy = agent.state.last_free_energy)
    
    slide!(agent)
end

# Cleanup
close_logging(loggers)
```

## 6. Next Steps

- Read the [full README](README.md) for detailed documentation
- Check [ARCHITECTURE.md](docs/ARCHITECTURE.md) for design details
- Explore [mountain_car_example.jl](examples/mountain_car_example.jl) for a complete example
- Review the [test suite](test/runtests.jl) for usage patterns

## Common Tasks

### Change Planning Horizon

Edit `config.jl`:
```julia
const AGENT = (
    planning_horizon = 30,  # Increase to 30
    # ...
)
```

### Disable Diagnostics

```julia
const DIAGNOSTICS = (
    track_beliefs = false,
    track_predictions = false,
    # ...
)
```

### Add Custom Logging

```julia
log_event("custom_event", Dict(
    "metric" => value,
    "data" => [1, 2, 3]
))
```

### Export Results

```julia
using JSON

results = Dict(
    "states" => states,
    "actions" => actions,
    "diagnostics" => get_comprehensive_summary(diagnostics)
)

open("outputs/results.json", "w") do f
    JSON.print(f, results, 2)
end
```

## Troubleshooting

### Package Installation Issues

```bash
rm -rf ~/.julia/registries
julia -e 'using Pkg; Pkg.update()'
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Inference Errors

Check that:
- Transition and control functions return correct dimensions
- State and action dimensions match your problem
- Initial state is valid

### Memory Issues

- Reduce planning horizon
- Disable diagnostics
- Clear histories periodically

### Slow Performance

- Reduce inference iterations (edit `config.jl`)
- Use smaller planning horizon
- Disable free energy tracking

## Getting Help

1. Check the [README](README.md)
2. Review the [examples](examples/)
3. Look at the [tests](test/)
4. Read module docstrings in [src/](src/)

---

**Ready to build Active Inference agents!** 🚀

