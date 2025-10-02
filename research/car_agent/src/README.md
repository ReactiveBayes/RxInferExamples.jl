# Source Code Documentation

Core implementation modules for the Generic Active Inference Agent Framework.

## Overview

This directory contains the three main modules that comprise the framework:

1. **agent.jl** - Core Active Inference agent implementation
2. **diagnostics.jl** - Diagnostic tracking and monitoring
3. **logging.jl** - Structured logging system

## Module Architecture

```
src/
├── agent.jl           # Core agent implementation
├── diagnostics.jl     # Diagnostics and monitoring
├── logging.jl         # Logging utilities
├── README.md          # This file
└── AGENTS.md          # Agent architecture guide
```

## agent.jl - Core Agent

The heart of the framework, implementing generic Active Inference agents.

### Key Components

**Structures:**
- `AgentState`: Internal agent state (beliefs, priors, history)
- `GenericActiveInferenceAgent`: Main agent class

**Functions:**
- `step!(agent, observation, action)`: Perform inference
- `get_action(agent)`: Get next optimal action
- `get_predictions(agent)`: Get predicted future states
- `slide!(agent)`: Update planning window
- `reset!(agent)`: Reset to initial state

### Architecture

```
Agent
  ├── State (beliefs, priors, history)
  ├── Generative Model (transitions, control)
  ├── Inference Engine (placeholder for RxInfer)
  └── Planning Horizon (sliding window)
```

### Usage

```julia
include("src/agent.jl")
using .Agent

# Create agent
agent = GenericActiveInferenceAgent(
    horizon = 20,
    state_dim = 2,
    action_dim = 1,
    transition_func,
    control_func;
    goal_state = [1.0, 0.0]
)

# Active Inference loop
for t in 1:max_steps
    action = get_action(agent)
    observation = environment(action)
    step!(agent, observation, action)
    slide!(agent)
end
```

### Extension Points

- **Custom Transition Functions**: Define your system dynamics
- **Custom Control Functions**: Define how actions affect state
- **Custom Inference**: Override inference strategy
- **Custom Priors**: Adjust belief initialization

## diagnostics.jl - Diagnostics System

Comprehensive diagnostic tracking for monitoring agent performance.

### Key Components

**Trackers:**
- `MemoryTracer`: Track memory usage over time
- `PerformanceProfiler`: Time operation execution
- `BeliefTracker`: Monitor belief evolution
- `PredictionTracker`: Measure prediction accuracy
- `FreeEnergyTracker`: Monitor variational free energy
- `DiagnosticsCollector`: Unified interface

### Architecture

```
Diagnostics
  ├── Memory Tracing (heap usage, GC)
  ├── Performance Profiling (operation timing)
  ├── Belief Tracking (mean, covariance evolution)
  ├── Prediction Tracking (accuracy metrics)
  └── Free Energy Tracking (convergence monitoring)
```

### Usage

```julia
include("src/diagnostics.jl")
using .Diagnostics

# Create collector
diagnostics = DiagnosticsCollector()

# In simulation loop
start_timer!(diagnostics.performance_profiler, "inference")
step!(agent, observation, action)
stop_timer!(diagnostics.performance_profiler, "inference")

record_belief!(diagnostics.belief_tracker, t,
              agent.state.state_mean, agent.state.state_cov)

# Get summary
summary = get_comprehensive_summary(diagnostics)
print_diagnostics_report(diagnostics)
```

### Tracked Metrics

- **Memory**: Peak, average, GC time
- **Performance**: Min/max/avg operation times
- **Beliefs**: Evolution, changes, uncertainty
- **Predictions**: Errors by horizon
- **Free Energy**: History, reduction, convergence

## logging.jl - Logging System

Multi-format structured logging for analysis and debugging.

### Key Components

**Utilities:**
- `ProgressBar`: Visual progress indication
- `log_event()`: Structured event logging
- `log_agent_step()`: Log agent steps
- `start_operation()` / `finish_operation()`: Operation logging

### Architecture

```
Logging
  ├── Console Logging (human-readable)
  ├── File Logging (persistent)
  ├── Structured JSON (machine-readable)
  ├── Performance CSV (metrics)
  └── Progress Bars (visual feedback)
```

### Usage

```julia
include("src/logging.jl")
using .LoggingUtils

# Setup logging
loggers = setup_logging(
    verbose = true,
    structured = true,
    performance = true
)

# Log events
log_event("simulation_start", Dict(
    "agent_type" => "mountain_car",
    "horizon" => 20
))

# Progress bar
pb = ProgressBar(max_steps)
for t in 1:max_steps
    # ... simulation ...
    update!(pb, t)
end
finish!(pb)

# Cleanup
close_logging(loggers)
```

### Log Formats

- **Console**: `[INFO] Event: simulation_start`
- **JSON**: `{"timestamp": "...", "event": "simulation_start", ...}`
- **CSV**: Structured performance metrics

## Design Principles

### Modularity

Each module is self-contained and can be used independently:

```julia
# Use only agent
include("src/agent.jl")

# Use only diagnostics
include("src/diagnostics.jl")

# Use all modules
include("src/agent.jl")
include("src/diagnostics.jl")
include("src/logging.jl")
```

### Extensibility

All modules designed for extension:

- **Agent**: Custom inference strategies
- **Diagnostics**: Custom trackers
- **Logging**: Custom log formats

### Performance

Optimized for minimal overhead:

- Diagnostics: < 5% overhead when enabled
- Logging: Asynchronous where possible
- Agent: Efficient state management

## API Documentation

### Agent Module API

```julia
# Creation
GenericActiveInferenceAgent(horizon, state_dim, action_dim, g, h; kwargs...)

# Core operations
step!(agent, observation, action)
get_action(agent)
get_predictions(agent)
slide!(agent)
reset!(agent; kwargs...)

# Diagnostics
get_diagnostics(agent)
print_status(agent)
```

### Diagnostics Module API

```julia
# Memory tracking
trace_memory!(tracer)
get_memory_summary(tracer)

# Performance profiling
start_timer!(profiler, operation)
stop_timer!(profiler, operation)
get_performance_summary(profiler)

# Belief tracking
record_belief!(tracker, step, mean, cov)
get_belief_summary(tracker)

# Comprehensive
get_comprehensive_summary(collector)
print_diagnostics_report(collector)
```

### Logging Module API

```julia
# Progress bars
pb = ProgressBar(total)
update!(pb, current)
finish!(pb)

# Event logging
log_event(event_name, data)
log_agent_step(step, action, observation; kwargs...)

# Operation logging
start_operation(name)
finish_operation(name, steps)
```

## Error Handling

### Graceful Degradation

All modules handle errors gracefully:

```julia
# Agent continues with warnings if inference fails
step!(agent, observation, action)  # Warns but doesn't crash

# Diagnostics can be disabled
diagnostics = DiagnosticsCollector(enabled=false)

# Logging failures don't stop simulation
log_event("event", data)  # Warns but continues
```

### Validation

Input validation at entry points:

```julia
# Agent validates dimensions
@assert length(observation) == state_dim

# Diagnostics validate types
@assert mean isa AbstractVector
@assert cov isa AbstractMatrix
```

## Performance Optimization

### Memory Management

- Preallocate arrays where possible
- Clear histories periodically
- Use views instead of copies

### Computation

- Cache expensive computations
- Use matrix operations (BLAS)
- Avoid allocations in hot loops

### Profiling

```julia
# Profile agent
@time step!(agent, observation, action)

# Profile diagnostics
@time record_belief!(tracker, t, mean, cov)

# Find bottlenecks
using Profile
@profile begin
    for t in 1:1000
        step!(agent, randn(2), [0.0])
    end
end
Profile.print()
```

## Future Enhancements

Planned improvements:

- [ ] GPU acceleration for large-scale problems
- [ ] Distributed agent execution
- [ ] Advanced inference strategies
- [ ] Real-time visualization
- [ ] Model parameter learning
- [ ] Hierarchical agents
- [ ] Multi-agent coordination

## Contributing

When modifying source code:

1. Follow Julia style conventions
2. Add comprehensive docstrings
3. Include type annotations
4. Write tests for new features
5. Update this README
6. Benchmark performance impact

## Support

For implementation questions:

1. Check module docstrings
2. Review AGENTS.md
3. See examples directory
4. Consult tests for usage patterns

---

**Clean, modular, well-documented implementation for production use.**

