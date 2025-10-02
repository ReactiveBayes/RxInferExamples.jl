# Generic Agent Interface and Composability

**Framework:** Generic Agent-Environment Framework for Active Inference  
**Version:** 0.1.1  
**Date:** October 2, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [Abstract Interface Pattern](#abstract-interface-pattern)
4. [Type-Level Composability](#type-level-composability)
5. [Technical Implementation](#technical-implementation)
6. [Usage Patterns](#usage-patterns)
7. [Extension Guide](#extension-guide)
8. [Benefits and Trade-offs](#benefits-and-trade-offs)

---

## Overview

The Generic Agent-Environment Framework is built on a foundation of **abstract interfaces** and **composable type systems** that enable any agent to work with any environment, as long as their dimensions match. This design provides flexibility, type safety, and extensibility without sacrificing performance or correctness.

### Core Concept

```julia
# Any agent can work with any environment
# if their type parameters match
AbstractActiveInferenceAgent{S,A,O}  ↔  AbstractEnvironment{S,A,O}
                                    ↓
                            run_simulation(agent, env, config)
```

**Key Insight:** By defining clear interfaces and using Julia's type system, we achieve **compile-time guarantees** that agents and environments are compatible, while maintaining complete **runtime flexibility** in which implementations are used.

---

## Design Philosophy

### 1. Separation of Concerns

The framework cleanly separates three concerns:

**Agents (Intelligence):**
- Perform Active Inference
- Plan future actions
- Maintain beliefs about states
- **Don't know** about environment physics

**Environments (Physics):**
- Simulate world dynamics
- Execute actions
- Generate observations
- **Don't know** about agent inference

**Simulation Infrastructure:**
- Connects agents and environments
- Manages the agent-environment loop
- Tracks results and diagnostics
- **Doesn't care** what specific agent/environment is used

### 2. Interface-Driven Design

```julia
# Define what, not how
abstract type AbstractEnvironment{S,A,O} end
abstract type AbstractActiveInferenceAgent{S,A,O} end

# Implementations provide the how
struct MountainCarEnv <: AbstractEnvironment{2,1,2}
struct MountainCarAgent <: AbstractActiveInferenceAgent{2,1,2}
struct SimpleNavAgent <: AbstractActiveInferenceAgent{1,1,1}
```

### 3. Compile-Time Safety, Runtime Flexibility

**Compile-time:** Type parameters ensure dimensional compatibility
```julia
# This compiles: dimensions match
agent::AbstractActiveInferenceAgent{2,1,2}
env::AbstractEnvironment{2,1,2}
run_simulation(agent, env, config)  # ✓ Safe

# This doesn't compile: dimension mismatch
agent::AbstractActiveInferenceAgent{1,1,1}
env::AbstractEnvironment{2,1,2}
run_simulation(agent, env, config)  # ✗ Type error
```

**Runtime:** Factory pattern allows dynamic selection
```julia
# Runtime selection via configuration
agent = create_agent_from_config(config)  # Could be any agent
env = create_environment_from_config(config)  # Could be any environment
run_simulation(agent, env, sim_config)  # Works if dimensions match
```

---

## Abstract Interface Pattern

### The Agent Interface

Every agent must implement five methods:

```julia
"""
AbstractActiveInferenceAgent{S,A,O}

Type Parameters:
- S: State dimension
- A: Action dimension
- O: Observation dimension
"""
abstract type AbstractActiveInferenceAgent{S,A,O} end

# Required interface methods:
step!(agent, observation, action)      # Run inference
get_action(agent) -> action            # Get next action
get_predictions(agent) -> states       # Get predictions
slide!(agent)                          # Slide planning window
reset!(agent)                          # Reset to initial state
```

**Key Insight:** The interface defines **what** agents must do, not **how** they do it. Different agents can use completely different inference strategies (variational, sampling, analytical) as long as they implement these five methods.

### The Environment Interface

Every environment must implement four methods:

```julia
"""
AbstractEnvironment{S,A,O}

Type Parameters:
- S: State dimension (internal state)
- A: Action dimension (control inputs)
- O: Observation dimension (what agent sees)
"""
abstract type AbstractEnvironment{S,A,O} end

# Required interface methods:
step!(env, action) -> observation           # Execute action
reset!(env) -> observation                  # Reset to initial
get_state(env) -> state                     # Get current state
get_observation_model_params(env) -> params # Provide params for agent
```

**Key Insight:** Environments can implement arbitrary physics (linear, nonlinear, stochastic, deterministic) as long as they follow the interface contract.

### The Simulation Contract

The generic simulation runner relies on these interfaces:

```julia
function run_simulation(
    agent::AbstractActiveInferenceAgent{S,A,O},
    env::AbstractEnvironment{S,A,O},  # Types must match!
    config::SimulationConfig
) where {S,A,O}
    # Initialize
    obs = reset!(env)
    reset!(agent)
    
    # Agent-environment loop
    for t in 1:config.max_steps
        action = get_action(agent)      # Agent interface
        obs = step!(env, action)         # Environment interface
        step!(agent, obs, action)        # Agent interface
        slide!(agent)                    # Agent interface
    end
end
```

**Key Insight:** `run_simulation` works with **any** agent-environment pair that satisfies the interface, without knowing implementation details.

---

## Type-Level Composability

### The Type Parameter System

```julia
StateVector{N}       # N-dimensional state
ActionVector{M}      # M-dimensional action
ObservationVector{K} # K-dimensional observation
```

These strongly-typed wrappers provide:
1. **Compile-time dimension checking**
2. **Zero-cost abstractions** (no runtime overhead)
3. **Clear documentation** of dimensionality
4. **Type-driven dispatch** for different state dimensions

### Dimensional Compatibility

The framework enforces that agents and environments have compatible dimensions:

```julia
# Mountain Car: 2D state, 1D action, 2D observation
env::AbstractEnvironment{2,1,2}
agent::AbstractActiveInferenceAgent{2,1,2}
# ✓ Compatible: S=2, A=1, O=2 match

# Dimension mismatch caught at compile time
env::AbstractEnvironment{2,1,2}
agent::AbstractActiveInferenceAgent{1,1,1}
# ✗ Type error: {2,1,2} ≠ {1,1,1}
```

### Generic Algorithms

Type parameters enable generic algorithms that work for any dimensionality:

```julia
# Works for 1D, 2D, 3D, ..., N-D states
function save_trajectory(states::Vector{StateVector{N}}) where N
    # Automatically adapts to dimensionality
    if N == 1
        # 1D-specific formatting
    elseif N == 2
        # 2D-specific formatting
    else
        # Generic N-D formatting
    end
end
```

---

## Technical Implementation

### Agent Implementation Pattern

**Step 1: Define RxInfer Model (Top-Level)**

```julia
# Must be at module top-level for RxInfer
@model function my_agent_model(m_u, V_u, m_x, V_x, m_s_t_min, V_s_t_min, T, params...)
    # Generative model for agent's beliefs
    s_t_min ~ MvNormal(mean = m_s_t_min, cov = V_s_t_min)
    
    for k in 1:T
        u[k] ~ MvNormal(mean = m_u[k], cov = V_u[k])  # Control prior
        x[k] ~ MvNormal(mean = m_x[k], cov = V_x[k])  # Goal prior
        # ... model-specific dynamics
    end
    
    return (s,)
end
```

**Step 2: Create Agent Struct**

```julia
mutable struct MyAgent <: AbstractActiveInferenceAgent{S,A,O}
    # Type parameters S, A, O specify dimensions
    
    # Planning horizon
    horizon::Int
    
    # Goal specification
    goal_state::StateVector{S}
    
    # Belief state
    state_belief::Ref{Tuple{Vector{Float64}, Matrix{Float64}}}
    
    # Control priors (over planning horizon)
    m_u::Vector{Vector{Float64}}
    V_u::Vector{Matrix{Float64}}
    
    # Goal priors (over planning horizon)
    m_x::Vector{Vector{Float64}}
    V_x::Vector{Matrix{Float64}}
    
    # Environment parameters (from get_observation_model_params)
    env_params::NamedTuple
    
    # Inference results
    result_ref::Ref{Union{Nothing, Any}}
    
    # Constructor
    function MyAgent(
        horizon::Int,
        goal_state::StateVector{S},
        initial_state::StateVector{S},
        env_params;
        initial_state_precision::Float64 = 1e6
    ) where {S}
        # Initialize fields...
        new{S,A,O}(...)  # Specify all type parameters
    end
end
```

**Step 3: Implement Interface Methods**

```julia
# 1. Perform inference
function step!(agent::MyAgent, 
               observation::ObservationVector{O},
               action::ActionVector{A}) where {O,A}
    # Clamp current observation and action
    agent.m_x[1] = Vector(observation)
    agent.V_x[1] = tiny * diagm(ones(O))
    agent.m_u[1] = Vector(action)
    agent.V_u[1] = tiny * diagm(ones(A))
    
    # Prepare data for RxInfer
    data = Dict(
        :m_u => agent.m_u,
        :V_u => agent.V_u,
        :m_x => agent.m_x,
        :V_x => agent.V_x,
        :m_s_t_min => agent.state_belief[][1],
        :V_s_t_min => agent.state_belief[][2]
    )
    
    # Run variational inference
    model = my_agent_model(T = agent.horizon, ...)
    agent.result_ref[] = infer(model = model, data = data)
end

# 2. Extract action from inference results
function get_action(agent::MyAgent)::ActionVector{A} where {A}
    if agent.result_ref[] === nothing
        return ActionVector{A}(zeros(A))
    end
    
    # Extract posterior mean of next action
    posteriors = agent.result_ref[].posteriors[:u]
    action_mean = mode(posteriors[2])  # Second action (first future action)
    return ActionVector{A}(action_mean)
end

# 3. Get predicted states
function get_predictions(agent::MyAgent)::Vector{StateVector{S}} where {S}
    if agent.result_ref[] === nothing
        return [StateVector{S}(zeros(S)) for _ in 1:agent.horizon]
    end
    
    posteriors = agent.result_ref[].posteriors[:s]
    return [StateVector{S}(mode(p)) for p in posteriors]
end

# 4. Slide planning window
function slide!(agent::MyAgent)
    # Extract updated belief from inference results
    # ... (see mountain_car_agent.jl for details)
    
    # Shift priors forward in time
    agent.m_u = circshift(agent.m_u, -1)
    agent.V_u = circshift(agent.V_u, -1)
    agent.m_x = circshift(agent.m_x, -1)
    agent.V_x = circshift(agent.V_x, -1)
    
    # Set new priors at horizon end
    agent.m_u[end] = zeros(A)
    agent.m_x[end] = Vector(agent.goal_state)
    # ...
end

# 5. Reset to initial state
function reset!(agent::MyAgent)
    agent.state_belief[] = (zeros(S), tiny * diagm(ones(S)))
    agent.result_ref[] = nothing
    # Reset all priors...
end
```

### Environment Implementation Pattern

**Step 1: Create Environment Struct**

```julia
mutable struct MyEnv <: AbstractEnvironment{S,A,O}
    # Current state
    current_state::Ref{StateVector{S}}
    
    # Initial conditions
    initial_state::StateVector{S}
    
    # Physics parameters
    physics_params::NamedTuple
    
    # Observation model parameters
    observation_precision::Float64
    observation_noise_std::Float64
    
    # Constructor
    function MyEnv(;
        initial_state::Vector{Float64},
        physics_params...,
        observation_precision::Float64 = 1e4,
        observation_noise_std::Float64 = 0.01
    )
        state_ref = Ref(StateVector{S}(initial_state))
        new{S,A,O}(state_ref, ...)
    end
end
```

**Step 2: Implement Interface Methods**

```julia
# 1. Execute action
function step!(env::MyEnv, action::ActionVector{A})::ObservationVector{O} where {A,O}
    current = env.current_state[]
    
    # Apply physics
    new_state = physics_update(current, action, env.physics_params)
    env.current_state[] = new_state
    
    # Generate observation (state + noise)
    noise = env.observation_noise_std * randn(O)
    observation = ObservationVector{O}(Vector(new_state) .+ noise)
    
    return observation
end

# 2. Reset environment
function reset!(env::MyEnv)::ObservationVector{O} where {O}
    env.current_state[] = env.initial_state
    return ObservationVector{O}(Vector(env.initial_state))
end

# 3. Get current state
function get_state(env::MyEnv)::StateVector{S} where {S}
    return env.current_state[]
end

# 4. Provide observation model parameters to agents
function get_observation_model_params(env::MyEnv)
    return (
        observation_precision = env.observation_precision,
        observation_noise_std = env.observation_noise_std,
        # Include any physics parameters agents need for their models
        physics_params = env.physics_params
    )
end
```

---

## Usage Patterns

### Pattern 1: Explicit Composition

```julia
# Create environment
env = MountainCarEnv(
    initial_position = -0.5,
    initial_velocity = 0.0,
    engine_force_limit = 0.04
)

# Get parameters for agent
params = get_observation_model_params(env)

# Create compatible agent
agent = MountainCarAgent(
    20,  # horizon
    StateVector{2}([0.5, 0.0]),  # goal
    StateVector{2}([-0.5, 0.0]), # initial state
    params  # environment parameters
)

# Run - type system ensures compatibility
result = run_simulation(agent, env, config)
```

### Pattern 2: Factory-Based Composition

```julia
# Load configuration
config = load_config("config.toml")

# Create components via factories
env = create_environment_from_config(config["environment"])
agent = create_agent_from_config(
    config["agent"],
    config["environment"],
    env
)

# Run - types matched by factory
result = run_simulation(agent, env, sim_config)
```

### Pattern 3: Mix-and-Match

```julia
# Any agent with any compatible environment
agents = [
    MountainCarAgent(20, goal, initial, params1),
    MountainCarAgent(30, goal, initial, params1),  # Different horizon
    SimpleNavAgent(15, StateVector{1}([1.0]), ...)  # Different agent
]

environments = [
    MountainCarEnv(initial_position=-0.5),
    MountainCarEnv(initial_position=-0.3),  # Different initial conditions
    SimpleNavEnv(initial_position=0.0)       # Different environment
]

# Run all compatible combinations
for agent in agents
    for env in environments
        if dimensions_match(agent, env)  # Type system checks this
            result = run_simulation(agent, env, config)
            save_outputs(result, ...)
        end
    end
end
```

---

## Extension Guide

### Adding a New Agent

**Step 1:** Define generative model
```julia
@model function new_agent_model(...)
    # Your model here
end
```

**Step 2:** Create agent struct
```julia
mutable struct NewAgent <: AbstractActiveInferenceAgent{S,A,O}
    # Your fields here
end
```

**Step 3:** Implement five interface methods
- `step!` - Run inference
- `get_action` - Extract action
- `get_predictions` - Extract predictions
- `slide!` - Update planning window
- `reset!` - Reset state

**Step 4:** Add to factory (optional)
```julia
# In src/config.jl
elseif agent_type == "NewAgent"
    return NewAgent(...)
```

### Adding a New Environment

**Step 1:** Create environment struct
```julia
mutable struct NewEnv <: AbstractEnvironment{S,A,O}
    current_state::Ref{StateVector{S}}
    # Your fields here
end
```

**Step 2:** Implement four interface methods
- `step!` - Execute action, return observation
- `reset!` - Reset to initial state
- `get_state` - Return current state
- `get_observation_model_params` - Provide params

**Step 3:** Add to factory (optional)
```julia
# In src/config.jl
elseif env_type == "NewEnv"
    return NewEnv(...)
```

### The agent and environment automatically work together if dimensions match!

---

## Benefits and Trade-offs

### Benefits

**1. Type Safety**
- Compile-time dimension checking
- Impossible to mix incompatible agents/environments
- Catches errors before runtime

**2. Flexibility**
- Add new agents without modifying environments
- Add new environments without modifying agents
- Mix and match any compatible pair

**3. Maintainability**
- Clear interface contracts
- Separation of concerns
- Easy to understand system boundaries

**4. Extensibility**
- New implementations don't affect existing code
- No central registry needed
- Plug-and-play architecture

**5. Performance**
- Zero-cost abstractions (Julia compiles specialized code)
- No virtual dispatch overhead
- Type-stable code throughout

**6. Testability**
- Test agents and environments independently
- Mock implementations easy to create
- Integration tests straightforward

### Trade-offs

**1. Learning Curve**
- Need to understand abstract interfaces
- Must implement all required methods
- Type parameters can be intimidating initially

**2. Boilerplate**
- Each new agent/environment requires full interface implementation
- Cannot skip methods (compiler enforces completeness)

**3. Rigidity**
- Interface changes affect all implementations
- Adding required methods breaks existing code
- Versioning becomes important

**4. Complexity**
- More abstract than direct implementation
- Requires thinking about contracts and composition
- May be overkill for simple use cases

### When This Pattern Excels

✅ **Good fit:**
- Multiple agents with different algorithms
- Multiple environments with different physics
- Research requiring many experiments
- Need for reproducibility and modularity
- Long-term codebase maintenance

❌ **Overkill:**
- Single agent, single environment
- Prototype/throwaway code
- Very simple systems
- Performance-critical inner loops (though Julia optimizes this well)

---

## Real-World Examples

### Example 1: Same Agent, Different Environments

```julia
# Train mountain car agent in different terrains
agent = MountainCarAgent(20, goal, initial, params)

environments = [
    MountainCarEnv(friction_coefficient=0.05),  # Low friction
    MountainCarEnv(friction_coefficient=0.15),  # High friction
    MountainCarEnv(engine_force_limit=0.02),    # Weak engine
    MountainCarEnv(engine_force_limit=0.08),    # Strong engine
]

for (i, env) in enumerate(environments)
    result = run_simulation(agent, env, config)
    save_outputs(result, "env_variation_$i")
end
```

### Example 2: Different Agents, Same Environment

```julia
# Compare different agents on same task
env = MountainCarEnv(initial_position=-0.5)

agents = [
    MountainCarAgent(10, goal, initial, params),  # Short horizon
    MountainCarAgent(20, goal, initial, params),  # Medium horizon
    MountainCarAgent(40, goal, initial, params),  # Long horizon
]

for (i, agent) in enumerate(agents)
    result = run_simulation(agent, env, config)
    save_outputs(result, "agent_variation_$i")
end
```

### Example 3: Research Across Dimensions

```julia
# Study different state space dimensionalities
experiments = [
    (SimpleNavAgent{1,1,1}(...), SimpleNavEnv{1,1,1}(...)),     # 1D
    (MountainCarAgent{2,1,2}(...), MountainCarEnv{2,1,2}(...)), # 2D
    (CartPoleAgent{4,1,4}(...), CartPoleEnv{4,1,4}(...)),       # 4D
]

for (name, (agent, env)) in experiments
    result = run_simulation(agent, env, config)
    analyze_results(result, name)
end
```

---

## Summary

The Generic Agent-Environment Framework achieves **composability** through:

1. **Abstract Interfaces** - Clear contracts for agents and environments
2. **Type Parameters** - Compile-time dimension checking
3. **Generic Algorithms** - Simulation code works with any implementation
4. **Factory Pattern** - Runtime selection of implementations
5. **Separation of Concerns** - Agents, environments, and infrastructure decoupled

This design enables **flexible, type-safe, and maintainable** Active Inference research, where new agents and environments can be added without modifying existing code, and any compatible pair can be composed together automatically.

---

## Further Reading

- [complete_guide.md](complete_guide.md) - Complete framework guide
- [comprehensive_summary.md](comprehensive_summary.md) - Framework overview
- [index.md](index.md) - API reference with detailed examples
- [quickstart.md](quickstart.md) - Get started quickly

---

**Framework Version:** 0.1.1  
**Document Version:** 1.0  
**Last Updated:** October 2, 2025

