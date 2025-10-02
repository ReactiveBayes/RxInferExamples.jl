# Examples Directory

Complete working examples demonstrating the Generic Active Inference Agent Framework.

## Overview

This directory contains full implementations showing how to use the framework for specific problems.

## Available Examples

### mountain_car_example.jl

Classic Mountain Car reinforcement learning problem solved with Active Inference.

**Problem**: An underpowered car must build momentum to escape a valley by oscillating back and forth.

**Features**:
- Complete physics implementation
- Custom generative model
- Full diagnostics integration
- Visualization of results

**Run**:
```bash
julia --project=. examples/mountain_car_example.jl
```

## Creating New Examples

### Template

```julia
# 1. Setup
using Pkg
Pkg.activate(".")

include("../config.jl")
include("../src/agent.jl")
include("../src/diagnostics.jl")
include("../src/logging.jl")

# 2. Define Problem-Specific Functions
function transition_func(s::AbstractVector)
    # Your system dynamics
end

function control_func(u::AbstractVector)
    # Your control effects
end

# 3. Create Agent
agent = GenericActiveInferenceAgent(
    horizon, state_dim, action_dim,
    transition_func, control_func;
    goal_state = your_goal
)

# 4. Run Simulation
diagnostics = DiagnosticsCollector()

for t in 1:max_steps
    action = get_action(agent)
    observation = simulate_environment(action)
    
    step!(agent, observation, action)
    record_diagnostics!(diagnostics, t, agent)
    slide!(agent)
end

# 5. Visualize Results
plot_results(agent, diagnostics)
```

### Guidelines

1. **Self-Contained**: Each example should run independently
2. **Well-Documented**: Include comments explaining key concepts
3. **Realistic**: Use actual physics/dynamics, not toy problems
4. **Complete**: Show full workflow from setup to visualization
5. **Tested**: Verify example runs successfully

## Example Domains

Ideas for future examples:

- **Cartpole Balancing**: Classic control problem
- **Drone Navigation**: 3D trajectory planning
- **Robotic Arm**: Multi-joint control
- **Pursuit-Evasion**: Multi-agent coordination
- **Economic Agent**: Resource allocation
- **Cognitive Model**: Decision-making simulation

## Support

For help with examples:

1. Review existing examples
2. Check `src/AGENTS.md` for architecture details
3. See test suite for usage patterns
4. Consult main README

---

**Learn by example - clear, complete, realistic demonstrations.**

