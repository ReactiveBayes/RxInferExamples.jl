# Agent Examples Guide

Guide to understanding and creating Active Inference agent examples.

## Mountain Car Example

### Problem Description

An underpowered car in a 1D valley must reach a goal at the top of the right hill. The car's engine is too weak to drive straight up, so it must build momentum by oscillating.

**State**: `[position, velocity]`
**Action**: `[force]` ∈ [-1, 1]
**Goal**: Position > 0.5 (top of right hill)

### Physics

```julia
function mountain_car_physics(state, action)
    x, v = state
    force = action[1]
    
    # Acceleration from force and gravity
    a = force - 0.0025 * cos(3*x)
    
    # Update velocity and position
    v_new = clamp(v + a, -0.07, 0.07)
    x_new = clamp(x + v_new, -1.2, 0.6)
    
    return [x_new, v_new]
end
```

### Agent Configuration

```julia
agent = GenericActiveInferenceAgent(
    horizon = 20,              # Plan 20 steps ahead
    state_dim = 2,             # [position, velocity]
    action_dim = 1,            # [force]
    transition_func,           # Natural oscillation
    control_func;              # Force effects
    goal_state = [0.5, 0.0],  # Top of hill, stationary
    transition_precision = 1e4,
    goal_prior_precision = 1e4
)
```

### Key Insights

1. **Oscillation Strategy**: Agent learns to rock back and forth
2. **Momentum Building**: Uses gravity to build velocity
3. **Planning Horizon**: Needs ~15-20 steps to see solution
4. **Goal Prior**: Strong goal prior encourages reaching target

### Adaptations

Try modifying:
- Gravity strength: Change `0.0025` coefficient
- Speed limits: Adjust velocity clamping
- Hill steepness: Modify `cos(3*x)` frequency
- Goal location: Set different target positions

## Creating Problem-Specific Examples

### 1. Define State Space

What information does the agent need?

```julia
# Minimal state (position only)
state = [x]

# Full state (position and derivatives)
state = [x, dx, ddx]

# Augmented state (physics + cognition)
state = [physical_vars..., cognitive_vars...]
```

### 2. Define Action Space

What can the agent control?

```julia
# Continuous control
action = [force] # ∈ ℝ

# Multi-dimensional control
action = [force_x, force_y, torque]

# Bounded control
action = clamp.(raw_action, lower, upper)
```

### 3. Define Dynamics

How does the system evolve?

```julia
# First-order dynamics
transition_func = (s) -> s + dt * f(s)

# Second-order dynamics  
transition_func = (s) -> [s[2], f(s)]

# Stochastic dynamics
transition_func = (s) -> g(s) + randn() * noise_std
```

### 4. Define Control

How do actions affect the system?

```julia
# Additive control
control_func = (u) -> B * u

# Multiplicative control
control_func = (u) -> diagm(u) * state_effect

# Nonlinear control
control_func = (u) -> nonlinear_transformation(u)
```

### 5. Set Goal and Precision

What is the agent trying to achieve?

```julia
# Point goal
goal_state = [target_x, target_v]

# Region goal (use lower precision)
goal_state = [approx_target, 0.0]
goal_prior_precision = 1e2  # Allows tolerance

# Time-varying goal
for t in 1:horizon
    agent.state.goal_means[t] = trajectory[t]
end
```

### 6. Choose Horizon

How far should the agent plan?

```julia
# Short horizon: Fast, reactive
horizon = 5

# Medium horizon: Balanced
horizon = 20

# Long horizon: Strategic
horizon = 50
```

Rule of thumb: Horizon ≥ (steps to goal / 2)

### 7. Run Simulation

```julia
for t in 1:max_steps
    # Get action
    action = get_action(agent)
    
    # Apply to environment
    observation = physics(current_state, action)
    current_state = observation
    
    # Update agent
    step!(agent, observation, action)
    slide!(agent)
    
    # Check goal
    if reached_goal(current_state, goal)
        break
    end
end
```

## Example Patterns

### Pattern 1: Target Reaching

Agent navigates to fixed goal:
- Clear goal state
- High goal prior precision
- Medium horizon

### Pattern 2: Trajectory Following

Agent follows predefined path:
- Time-varying goals
- Moderate goal precision
- Long horizon

### Pattern 3: Obstacle Avoidance

Agent avoids obstacles while reaching goal:
- Multi-objective (attraction + repulsion)
- Adaptive precision
- Medium horizon, frequent replanning

### Pattern 4: Resource Management

Agent balances competing objectives:
- Multiple soft constraints
- Lower goal precision
- Short horizon, greedy

### Pattern 5: Exploration

Agent explores uncertain environment:
- Weak priors
- High process noise
- Long horizon

## Visualization

### State Trajectory

```julia
using Plots

plot(state_history, label=["x" "v"],
     xlabel="Time", ylabel="State",
     title="State Evolution")
```

### Action Sequence

```julia
plot(action_history, label="Force",
     xlabel="Time", ylabel="Action",
     title="Control Sequence")
```

### Free Energy

```julia
plot(free_energy_history,
     xlabel="Time", ylabel="Free Energy",
     title="Inference Convergence")
```

### Phase Space

```julia
scatter(state_history[:,1], state_history[:,2],
        xlabel="Position", ylabel="Velocity",
        title="Phase Space Trajectory")
```

## Debugging Examples

### Agent Doesn't Reach Goal

**Possible causes**:
1. Horizon too short → Increase horizon
2. Goal precision too weak → Increase goal_prior_precision
3. Physics incorrect → Verify transition_func
4. Actions too weak → Check control_func scaling

### Agent Behavior Unstable

**Possible causes**:
1. Precision too high → Reduce precision values
2. Actions unbounded → Add clamping
3. Numerical issues → Check for NaN/Inf
4. Learning rate wrong → Adjust inference iterations

### Inference Too Slow

**Possible causes**:
1. Horizon too long → Reduce horizon
2. Too many iterations → Reduce inference_iterations
3. Expensive functions → Optimize transition_func/control_func

### Memory Issues

**Possible causes**:
1. History growing unbounded → Clear periodically
2. Large state dimensions → Reduce if possible
3. Too many diagnostics → Disable some trackers

## Best Practices

1. **Start Minimal**: Simplest version first
2. **Validate Physics**: Test dynamics independently
3. **Tune Incrementally**: One parameter at a time
4. **Visualize Often**: Plot everything
5. **Compare Baselines**: Against known solutions
6. **Document Assumptions**: Explain design choices

---

**Clear examples accelerate understanding and adoption.**

