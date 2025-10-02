# Generic Active Inference Agent Framework - Architecture

This document describes the architecture and design principles of the generic Active Inference agent framework.

## Design Philosophy

The framework is built on several key principles:

1. **Modularity**: Each component has a single responsibility
2. **Extensibility**: Easy to add new features without modifying core code
3. **Clarity**: Code should be self-documenting and easy to understand
4. **Production-Ready**: Robust error handling and comprehensive testing
5. **Domain-Agnostic**: Works for any problem that can be framed as Active Inference

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application                          │
│  (mountain_car_example.jl, your_problem.jl, etc.)          │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Framework                            │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Agent.jl    │  │Diagnostics.jl│  │  Logging.jl  │     │
│  │              │  │              │  │              │     │
│  │ - State      │  │ - Memory     │  │ - Console    │     │
│  │ - Inference  │  │ - Performance│  │ - Structured │     │
│  │ - Planning   │  │ - Beliefs    │  │ - CSV        │     │
│  │ - Actions    │  │ - Predictions│  │ - Progress   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    RxInfer.jl                               │
│  (Message Passing, Variational Inference, Factor Graphs)    │
└─────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### 1. Agent Module (`src/agent.jl`)

**Purpose**: Core Active Inference agent implementation

**Key Components**:
- `AbstractActiveInferenceAgent`: Abstract interface for all agents
- `GenericActiveInferenceAgent`: Concrete implementation
- `AgentState`: State management structure
- `build_generic_model()`: RxInfer model construction

**Responsibilities**:
- Maintain agent state (beliefs, priors, history)
- Perform inference using RxInfer
- Compute optimal actions
- Predict future states
- Manage planning horizon

**Design Patterns**:
- Strategy pattern for different inference algorithms
- State pattern for agent lifecycle management
- Factory pattern for model construction

### 2. Diagnostics Module (`src/diagnostics.jl`)

**Purpose**: Comprehensive diagnostic tracking and analysis

**Key Components**:
- `MemoryTracer`: Track memory usage
- `PerformanceProfiler`: Time operations
- `BeliefTracker`: Monitor belief evolution
- `PredictionTracker`: Measure prediction accuracy
- `FreeEnergyTracker`: Track variational free energy
- `DiagnosticsCollector`: Unified interface

**Responsibilities**:
- Real-time monitoring of agent performance
- Memory leak detection
- Performance profiling
- Convergence analysis
- Data collection for post-hoc analysis

**Design Patterns**:
- Observer pattern for event tracking
- Composite pattern for diagnostics collector
- Singleton pattern for global tracers

### 3. Logging Module (`src/logging.jl`)

**Purpose**: Multi-format structured logging

**Key Components**:
- `StructuredLogger`: JSON event logging
- `PerformanceLogger`: CSV metrics logging
- `ProgressBar`: Visual progress feedback
- Setup and teardown utilities

**Responsibilities**:
- Console output for human readability
- File logging for persistence
- Structured JSON for machine parsing
- Performance CSV for analysis
- Progress visualization

**Design Patterns**:
- Decorator pattern for multiple loggers
- Facade pattern for logging setup
- Builder pattern for log message construction

### 4. Configuration Module (`config.jl`)

**Purpose**: Centralized parameter management

**Key Components**:
- Agent parameters (AGENT)
- Simulation parameters (SIMULATION)
- Logging parameters (LOGGING)
- Diagnostics parameters (DIAGNOSTICS)
- Validation utilities

**Responsibilities**:
- Single source of truth for all parameters
- Parameter validation
- Custom configuration creation
- Documentation of defaults

**Design Patterns**:
- Singleton pattern for configuration
- Validation pattern for parameter checking

## Core Workflows

### Active Inference Loop

```julia
# Initialize
agent = GenericActiveInferenceAgent(...)
diagnostics = DiagnosticsCollector()

# Main loop
for t in 1:max_steps
    # 1. ACT: Get action from agent
    action = get_action(agent)
    
    # 2. EXECUTE: Apply action in environment
    observation = environment(action)
    
    # 3. INFER: Update beliefs
    step!(agent, observation, action)
    
    # 4. SLIDE: Move planning window
    slide!(agent)
    
    # 5. DIAGNOSE: Record metrics
    record_diagnostics!(diagnostics, agent, t)
end

# Report results
print_diagnostics_report(diagnostics)
```

### Inference Process

```julia
function step!(agent, observation, action)
    # 1. Clamp current action and observation
    agent.state.control_means[1] = action
    agent.state.goal_means[1] = observation
    
    # 2. Build generative model
    model = build_generic_model(...)
    
    # 3. Run RxInfer variational inference
    result = infer(model=model, data=data, ...)
    
    # 4. Extract posteriors
    agent.state.inference_result = result
    
    # 5. Update diagnostics
    update_diagnostics!(agent, result)
end
```

### Sliding Window

```julia
function slide!(agent)
    # 1. Extract updated state belief from inference
    (m_new, V_new) = extract_updated_belief(result)
    
    # 2. Shift control priors forward
    agent.state.control_means = circshift(..., -1)
    agent.state.control_means[end] = default
    
    # 3. Shift goal priors forward
    agent.state.goal_means = circshift(..., -1)
    agent.state.goal_means[end] = goal_state
    
    # 4. Update state belief
    agent.state.state_mean = m_new
    agent.state.state_cov = V_new
end
```

## Data Flow

### Inference Data Flow

```
Input:
  - Current observation
  - Previous action
  - Agent state (beliefs, priors)
  
Processing:
  1. Clamp known values
  2. Build factor graph
  3. Run message passing
  4. Extract posteriors
  
Output:
  - Updated state belief
  - Optimal action distribution
  - Predicted future states
  - Free energy value
```

### Diagnostic Data Flow

```
Runtime:
  - Memory measurements (periodic)
  - Performance timings (per operation)
  - Belief states (every step)
  - Predictions (every step)
  - Free energy (every step)

Storage:
  - In-memory buffers
  - CSV files (metrics)
  - JSON files (structured events)
  - Log files (human-readable)

Analysis:
  - Summary statistics
  - Convergence analysis
  - Performance reports
  - Visualization data
```

## Extension Points

### Adding New Agent Types

1. Inherit from `AbstractActiveInferenceAgent`
2. Implement required methods:
   - `step!(agent, observation, action)`
   - `get_action(agent)`
   - `get_predictions(agent)`
   - `slide!(agent)`
   - `reset!(agent)`

### Adding New Diagnostics

1. Create new tracker struct
2. Implement recording methods
3. Implement summary method
4. Add to `DiagnosticsCollector`

### Adding New Inference Algorithms

1. Modify `step!()` to support algorithm selection
2. Implement algorithm-specific inference logic
3. Update model building as needed

### Adding Problem Domains

1. Define transition function `g(s)`
2. Define control function `h(u)`
3. Optionally define inverse `h_inv(Δs)`
4. Create agent with your functions
5. Write domain-specific environment

## Performance Considerations

### Optimization Strategies

1. **Planning Horizon**: Smaller = faster but less optimal
2. **Inference Iterations**: Fewer = faster but less accurate
3. **Precision Matrices**: Sparse when possible
4. **Caching**: Reuse inference results when applicable
5. **Diagnostics**: Disable for production speedup

### Memory Management

- State history grows linearly with steps
- Diagnostics can be memory-intensive
- Use periodic garbage collection
- Clear histories when not needed

### Scalability

- Linear in planning horizon
- Linear in state/action dimensions
- Quadratic in inference iterations
- Logarithmic in diagnostic overhead

## Testing Strategy

### Test Levels

1. **Unit Tests**: Individual functions and modules
2. **Integration Tests**: Cross-module interactions
3. **System Tests**: Full Active Inference loops
4. **Performance Tests**: Speed and memory benchmarks
5. **Edge Case Tests**: Boundary conditions

### Test Coverage Goals

- Code coverage: >95%
- Branch coverage: >90%
- All public APIs tested
- All error paths tested
- Performance regression testing

## Error Handling

### Error Recovery

1. **Graceful Degradation**: Continue with reduced functionality
2. **State Preservation**: Maintain valid state on errors
3. **Logging**: Record all errors for debugging
4. **User Feedback**: Clear error messages

### Common Error Cases

- Invalid parameters (caught at creation)
- Inference failures (fallback to previous)
- Memory issues (periodic cleanup)
- Numerical instability (clipping, validation)

## Future Enhancements

Potential areas for expansion:

1. **Multi-Agent Systems**: Coordinate multiple agents
2. **Hierarchical Planning**: Multi-scale planning horizons
3. **Online Learning**: Adapt model parameters
4. **Distributed Computing**: Parallel inference
5. **GPU Acceleration**: For large-scale problems
6. **Visualization**: Real-time plotting and animation
7. **Benchmarking Suite**: Standard test problems

---

For implementation details, see the source code and inline documentation.

