# Active Inference Agent Architecture

Comprehensive guide to the Active Inference agent implementation.

## Table of Contents

1. [Conceptual Overview](#conceptual-overview)
2. [Agent Architecture](#agent-architecture)
3. [Generative Model](#generative-model)
4. [Inference Process](#inference-process)
5. [Planning Horizon](#planning-horizon)
6. [Implementation Details](#implementation-details)
7. [Customization Guide](#customization-guide)

## Conceptual Overview

### What is Active Inference?

Active Inference is a principle from theoretical neuroscience that explains how agents:

1. **Form Beliefs**: Maintain probabilistic beliefs about hidden states
2. **Predict Observations**: Generate predictions about future sensory data
3. **Minimize Surprise**: Take actions that reduce prediction error
4. **Achieve Goals**: By making goal states unsurprising

### Key Concepts

**Free Energy**: A variational bound on surprise. Agents minimize free energy by:
- **Perception**: Updating beliefs to fit observations
- **Action**: Selecting actions that bring expected observations closer to goals

**Generative Model**: The agent's internal model of how the world works:
```
p(observations | states, actions)
```

**Planning Horizon**: How far ahead the agent plans (temporal depth of inference).

## Agent Architecture

### High-Level Structure

```
GenericActiveInferenceAgent
├── State (current beliefs and history)
├── Generative Model (world dynamics)
├── Inference Engine (belief updates)
└── Planning System (action selection)
```

### Agent State

The `AgentState` structure tracks all agent information:

```julia
mutable struct AgentState
    # Dimensions
    horizon::Int              # Planning horizon (T steps)
    state_dim::Int           # State space dimensionality  
    action_dim::Int          # Action space dimensionality
    
    # Current beliefs
    state_mean::Vector{Float64}      # Current state estimate
    state_cov::Matrix{Float64}       # Current state uncertainty
    
    # Planning priors
    control_means::Vector{Vector{Float64}}  # Action priors
    control_covs::Vector{Matrix{Float64}}    # Action uncertainties
    goal_means::Vector{Vector{Float64}}      # Goal priors
    goal_covs::Vector{Matrix{Float64}}       # Goal uncertainties
    
    # Inference results
    inference_result::Union{Nothing, Any}    # RxInfer result
    last_free_energy::Union{Nothing, Float64}  # Last free energy value
    
    # History
    step_count::Int          # Total steps taken
    total_inference_time::Float64  # Cumulative inference time
    
    # Diagnostics
    memory_usage::Vector{Float64}  # Memory over time
    peak_memory_mb::Float64       # Peak memory usage
end
```

### Agent Interface

Public API for agent interactions:

```julia
# Core operations
step!(agent, observation, action)  # Update beliefs
get_action(agent)                  # Get next action
get_predictions(agent)             # Get predicted states
slide!(agent)                      # Slide planning window
reset!(agent; kwargs...)           # Reset agent state

# Introspection
get_diagnostics(agent)             # Get diagnostic info
print_status(agent)                # Print current status
```

## Generative Model

### Model Structure

The agent's generative model defines how it believes the world works:

```
State transition: s_t = g(s_{t-1}) + h(u_t) + w_t
Observation:      x_t = s_t + v_t
Goal prior:       x_t ~ N(goal, precision)
```

Where:
- `g()`: Transition function (autonomous dynamics)
- `h()`: Control function (action effects)
- `w_t`: Process noise
- `v_t`: Observation noise

### Mathematical Formulation

For a planning horizon T:

```
p(s_{1:T}, u_{1:T}, x_{1:T}) = 
    p(s_0) ∏_{t=1}^T p(s_t | s_{t-1}, u_t) p(x_t | s_t) p(x_t | goal)
```

Where:
- `p(s_t | s_{t-1}, u_t)`: State transition model
- `p(x_t | s_t)`: Observation model
- `p(x_t | goal)`: Goal prior (encodes preferences)

### Precision Matrices

Control belief precision:

```julia
Gamma  # Transition precision (inverse process noise)
Theta  # Observation precision (inverse observation noise)  
Sigma  # Goal prior precision (how strongly to enforce goals)
```

Higher precision = stronger belief = lower variance.

## Inference Process

### Variational Message Passing

The agent performs approximate Bayesian inference using variational message passing (implemented in RxInfer):

1. **Initialize**: Set initial beliefs about states and actions
2. **Forward Pass**: Propagate beliefs forward in time
3. **Backward Pass**: Backpropagate goal information
4. **Iterate**: Alternate until convergence
5. **Extract**: Get posterior beliefs (means and covariances)

### Message Passing Graph

```
s_0 → s_1 → s_2 → ... → s_T
      ↓     ↓           ↓
      x_1   x_2         x_T
      ↑     ↑           ↑
    goal  goal        goal
```

Messages flow bidirectionally to compute posteriors.

### Free Energy Minimization

The agent minimizes variational free energy:

```
F = ⟨log q(s, u) - log p(x, s, u)⟩_q
```

This is equivalent to maximizing model evidence (accuracy) while minimizing complexity.

### Inference Result

After inference, the agent has:
- Posterior beliefs over states: `q(s_{1:T})`
- Posterior beliefs over actions: `q(u_{1:T})`
- Predicted observations: `q(x_{1:T})`

## Planning Horizon

### Sliding Window

The agent maintains a fixed planning horizon T that slides forward:

```
Step 1:  [t, t+1, t+2, ..., t+T]
Step 2:     [t+1, t+2, ..., t+T+1]
Step 3:        [t+2, ..., t+T+2]
```

### Sliding Mechanism

```julia
function slide!(agent)
    # Shift control priors (drop first, add zero at end)
    agent.state.control_means = [
        agent.state.control_means[2:end]...,
        zeros(agent.state.action_dim)
    ]
    
    # Shift goal priors (goal stays at end)
    agent.state.goal_means = [
        agent.state.goal_means[2:end]...,
        agent.goal_state
    ]
    
    # Update state belief from first prediction
    # ...
end
```

### Horizon Selection

Choosing the planning horizon:

- **Short horizon (T < 10)**: Fast inference, myopic behavior
- **Medium horizon (T = 10-30)**: Balanced, most common
- **Long horizon (T > 30)**: Far-sighted, slower inference

## Implementation Details

### Agent Creation

```julia
agent = GenericActiveInferenceAgent(
    horizon::Int,                        # Planning horizon
    state_dim::Int,                      # State dimensionality
    action_dim::Int,                     # Action dimensionality
    transition_function::Function,       # g(s)
    control_function::Function;          # h(u)
    
    # Optional parameters
    control_inverse::Union{Nothing, Function} = nothing,
    transition_precision::Float64 = 1e4,
    observation_precision::Float64 = 1e4,
    control_prior_precision::Float64 = 1e-6,
    goal_prior_precision::Float64 = 1e4,
    initial_state_precision::Float64 = 1e6,
    goal_state::Vector{Float64} = zeros(state_dim),
    initial_state_mean::Vector{Float64} = zeros(state_dim),
    inference_iterations::Int = 10,
    track_free_energy::Bool = true,
    verbose::Bool = false
)
```

### Step Function

```julia
function step!(agent, observation, action)
    # Record history
    push!(agent.belief_history, (agent.state.state_mean, agent.state.state_cov))
    push!(agent.action_history, action)
    
    # Set priors based on observation
    agent.state.control_means[1] = action
    agent.state.goal_means[1] = observation
    
    # Perform inference (placeholder in generic version)
    # In real implementation, this calls RxInfer
    
    # Update diagnostics
    agent.state.step_count += 1
    agent.state.total_inference_time += elapsed_time
end
```

### Action Selection

```julia
function get_action(agent)
    if agent.state.inference_result === nothing
        return zeros(agent.state.action_dim)
    end
    
    # Extract action from inference result
    return agent.state.control_means[1]
end
```

## Customization Guide

### Custom Transition Functions

Define how your system evolves:

```julia
# Linear system
g_linear = (s::AbstractVector) -> A * s

# Nonlinear system  
g_nonlinear = (s::AbstractVector) -> begin
    x, v = s
    [v, -sin(x)]  # Pendulum dynamics
end

# With parameters
g_parametric = (s::AbstractVector) -> begin
    # Access global parameters
    A * s + B * external_input
end
```

### Custom Control Functions

Define how actions affect the system:

```julia
# Simple additive control
h_additive = (u::AbstractVector) -> B * u

# Nonlinear control
h_nonlinear = (u::AbstractVector) -> begin
    # Control saturates
    sign.(u) .* min.(abs.(u), max_control)
end

# State-dependent control (closure)
h_state_dependent = let current_state = nothing
    (u::AbstractVector) -> begin
        if current_state !== nothing
            return f(current_state, u)
        end
        return zeros(state_dim)
    end
end
```

### Custom Precision Matrices

Tune belief strengths:

```julia
# Precise transitions, noisy observations
transition_precision = 1e6
observation_precision = 1e2

# Noisy transitions, precise observations
transition_precision = 1e2
observation_precision = 1e6

# Strong goal enforcement
goal_prior_precision = 1e8

# Weak goal enforcement (exploration)
goal_prior_precision = 1e2
```

### Multiple Goals

Handle multiple objectives:

```julia
# Goal changes over time
function set_goal_sequence!(agent, goals)
    for (t, goal) in enumerate(goals[1:agent.state.horizon])
        agent.state.goal_means[t] = goal
    end
end

# Weighted goals
function set_weighted_goals!(agent, goals, weights)
    combined_goal = sum(w * g for (w, g) in zip(weights, goals))
    agent.goal_state = combined_goal
end
```

### Custom State Representations

Use different state representations:

```julia
# Cartesian coordinates
state = [x, y, vx, vy]

# Polar coordinates
state = [r, theta, vr, vtheta]

# Augmented state (with derivatives)
state = [x, dx, ddx]

# Multi-modal state
state = [physical_state..., cognitive_state...]
```

## Advanced Topics

### Hierarchical Agents

Build multi-level agents:

```julia
struct HierarchicalAgent
    high_level::GenericActiveInferenceAgent   # Plans goals
    low_level::GenericActiveInferenceAgent    # Plans actions
end
```

### Multi-Agent Systems

Coordinate multiple agents:

```julia
struct MultiAgentSystem
    agents::Vector{GenericActiveInferenceAgent}
    communication_graph::Matrix{Bool}
end
```

### Online Learning

Adapt model parameters:

```julia
function adapt_precision!(agent, prediction_error)
    if prediction_error > threshold
        agent.transition_precision *= 0.9
    else
        agent.transition_precision *= 1.1
    end
end
```

### Model Selection

Compare different models:

```julia
function select_best_model(models, observations)
    free_energies = [compute_free_energy(m, obs) for m in models]
    return argmin(free_energies)
end
```

## Best Practices

1. **Start Simple**: Begin with linear models, then add complexity
2. **Tune Precision**: Adjust precision matrices for your problem
3. **Choose Horizon**: Balance planning depth with computational cost
4. **Monitor Free Energy**: Use as convergence and performance metric
5. **Validate Model**: Ensure transition and control functions are correct
6. **Test Incrementally**: Add one feature at a time
7. **Profile Performance**: Identify bottlenecks early

## Common Pitfalls

- **Too Long Horizon**: Slow inference, diminishing returns
- **Wrong Precision**: Poor belief updates, unstable behavior
- **Invalid Functions**: Non-differentiable or discontinuous
- **Dimension Mismatch**: Ensure consistent dimensionality
- **Memory Leaks**: Clear histories periodically for long runs

---

**Deep understanding enables powerful Active Inference agents.**

