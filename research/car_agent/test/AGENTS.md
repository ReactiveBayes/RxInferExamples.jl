# Agent Testing Guide

Comprehensive guide for testing Active Inference agents in the Generic Active Inference Agent Framework.

## Overview

This document describes how to write tests for Active Inference agents, including test strategies, patterns, and best practices specific to probabilistic agents.

## Agent Test Philosophy

### Test What Matters

For Active Inference agents, focus on:

1. **Belief Updates**: Verify beliefs change appropriately with observations
2. **Action Selection**: Ensure actions move toward goals
3. **Planning Horizon**: Test sliding window mechanics
4. **Convergence**: Verify inference converges to solutions
5. **Robustness**: Test behavior with noisy/unexpected inputs

### Don't Test Implementation Details

Avoid testing:
- Internal RxInfer message passing (trust the library)
- Exact numerical values (use tolerances)
- Specific inference algorithms (test outcomes, not methods)

## Test Patterns

### Pattern 1: Agent Creation Test

```julia
@testset "Agent Creation" begin
    agent = GenericActiveInferenceAgent(
        horizon = 10,
        state_dim = 2,
        action_dim = 1,
        transition_func,
        control_func;
        goal_state = [1.0, 0.0]
    )
    
    @test agent isa GenericActiveInferenceAgent
    @test agent.state.horizon == 10
    @test agent.state.state_dim == 2
    @test agent.goal_state == [1.0, 0.0]
end
```

### Pattern 2: Belief Update Test

```julia
@testset "Belief Updates" begin
    agent = create_test_agent()
    
    # Record initial belief
    initial_mean = copy(agent.state.state_mean)
    
    # Perform inference
    step!(agent, observation, action)
    
    # Belief should change
    @test agent.state.state_mean != initial_mean
    @test agent.state.step_count == 1
end
```

### Pattern 3: Action Selection Test

```julia
@testset "Action Selection" begin
    agent = create_test_agent()
    
    # Agent should produce valid actions
    action = get_action(agent)
    @test action isa Vector{Float64}
    @test length(action) == agent.state.action_dim
    @test all(isfinite.(action))
end
```

### Pattern 4: Goal-Directed Behavior Test

```julia
@testset "Goal-Directed Behavior" begin
    agent = create_test_agent()
    goal = [1.0, 0.0]
    agent.goal_state = goal
    
    # Run multiple steps
    for t in 1:50
        observation = get_current_state()
        action = get_action(agent)
        step!(agent, observation, action)
        slide!(agent)
    end
    
    # State should move toward goal
    final_state = agent.state.state_mean
    distance_to_goal = norm(final_state - goal)
    @test distance_to_goal < initial_distance
end
```

### Pattern 5: Planning Horizon Test

```julia
@testset "Planning Horizon" begin
    agent = create_test_agent(horizon=5)
    
    # Predictions should span horizon
    predictions = get_predictions(agent)
    @test length(predictions) == 5
    
    # Slide window
    slide!(agent)
    
    # Horizon should remain constant
    predictions = get_predictions(agent)
    @test length(predictions) == 5
end
```

## Testing Different Agent Types

### Linear Agents

For agents with linear dynamics:

```julia
function create_linear_test_agent()
    A = [1.0 0.1; 0.0 0.9]  # Stable dynamics
    B = [0.0; 0.1]
    
    transition_func = (s::AbstractVector) -> A * s
    control_func = (u::AbstractVector) -> B * u[1]
    
    return GenericActiveInferenceAgent(
        10, 2, 1,
        transition_func,
        control_func;
        goal_state = [1.0, 0.0]
    )
end
```

Test characteristics:
- Predictable behavior
- Fast convergence
- Analytical solutions available for validation

### Nonlinear Agents

For agents with nonlinear dynamics:

```julia
function create_nonlinear_test_agent()
    # Pendulum-like dynamics
    transition_func = (s::AbstractVector) -> begin
        theta, omega = s
        [omega, -sin(theta)]
    end
    
    control_func = (u::AbstractVector) -> [0.0, u[1]]
    
    return GenericActiveInferenceAgent(
        20, 2, 1,
        transition_func,
        control_func;
        goal_state = [0.0, 0.0]
    )
end
```

Test characteristics:
- More complex behavior
- Slower convergence
- Need larger tolerances
- Test stability regions

## Testing Agent Properties

### Stability

Test that the agent remains stable:

```julia
@testset "Agent Stability" begin
    agent = create_test_agent()
    
    for t in 1:100
        observation = randn(2)  # Random observations
        action = get_action(agent)
        
        # Actions should remain bounded
        @test all(isfinite.(action))
        @test norm(action) < 100.0  # Reasonable bound
        
        step!(agent, observation, action)
        slide!(agent)
    end
end
```

### Convergence

Test that inference converges:

```julia
@testset "Inference Convergence" begin
    agent = create_test_agent()
    
    if agent.track_free_energy
        fe_history = Float64[]
        
        for t in 1:50
            step!(agent, observation, action)
            if agent.state.last_free_energy !== nothing
                push!(fe_history, agent.state.last_free_energy)
            end
            slide!(agent)
        end
        
        # Free energy should generally decrease
        @test fe_history[end] <= fe_history[1]
    end
end
```

### Robustness

Test behavior with various inputs:

```julia
@testset "Robustness to Noise" begin
    agent = create_test_agent()
    
    # Test with noisy observations
    for t in 1:20
        observation = true_state + 0.1 * randn(2)
        action = get_action(agent)
        step!(agent, observation, action)
        slide!(agent)
    end
    
    # Agent should still produce reasonable actions
    final_action = get_action(agent)
    @test all(isfinite.(final_action))
    @test norm(final_action) < 10.0
end
```

## Edge Case Testing

### Extreme Horizons

```julia
@testset "Extreme Horizons" begin
    # Very short horizon
    agent_short = create_test_agent(horizon=1)
    @test_nowarn step!(agent_short, [0.0, 0.0], [0.0])
    
    # Very long horizon
    agent_long = create_test_agent(horizon=100)
    @test_nowarn step!(agent_long, [0.0, 0.0], [0.0])
end
```

### Dimension Variations

```julia
@testset "Different Dimensions" begin
    # High-dimensional state
    agent_hd = create_test_agent(state_dim=10, action_dim=3)
    @test agent_hd.state.state_dim == 10
    @test agent_hd.state.action_dim == 3
end
```

### Extreme Values

```julia
@testset "Extreme Values" begin
    agent = create_test_agent()
    
    # Very large observations
    @test_nowarn step!(agent, [1000.0, 1000.0], [0.0])
    
    # Very small observations
    @test_nowarn step!(agent, [1e-10, 1e-10], [0.0])
end
```

## Performance Testing

### Inference Speed

```julia
@testset "Inference Performance" begin
    agent = create_test_agent(horizon=20)
    
    times = Float64[]
    for _ in 1:100
        t_start = time()
        step!(agent, randn(2), [0.0])
        t_end = time()
        push!(times, t_end - t_start)
        slide!(agent)
    end
    
    avg_time = mean(times)
    @test avg_time < 0.1  # Should be fast
end
```

### Memory Usage

```julia
@testset "Memory Usage" begin
    agent = create_test_agent()
    
    initial_mem = agent.state.peak_memory_mb
    
    for t in 1:1000
        step!(agent, randn(2), [0.0])
        slide!(agent)
    end
    
    final_mem = agent.state.peak_memory_mb
    
    # Memory shouldn't grow unbounded
    @test final_mem - initial_mem < 100.0  # MB
end
```

## Integration Testing

### Full Simulation Test

```julia
@testset "Complete Simulation" begin
    agent = create_test_agent()
    diagnostics = DiagnosticsCollector()
    
    for t in 1:100
        # Get action
        action = get_action(agent)
        
        # Simulate environment
        observation = simulate_environment(action)
        
        # Update agent
        start_timer!(diagnostics.performance_profiler, "inference")
        step!(agent, observation, action)
        stop_timer!(diagnostics.performance_profiler, "inference")
        
        # Record diagnostics
        record_belief!(diagnostics.belief_tracker, t,
                      agent.state.state_mean, agent.state.state_cov)
        
        slide!(agent)
    end
    
    # Verify simulation completed successfully
    @test agent.state.step_count == 100
    @test length(agent.belief_history) == 100
    
    # Check diagnostics
    summary = get_comprehensive_summary(diagnostics)
    @test summary["beliefs"]["measurements"] == 100
end
```

## Test Utilities

### Test Helpers

```julia
# Create standard test agent
function create_test_agent(;kwargs...)
    # Default test configuration
    ...
end

# Simulate simple environment
function simulate_environment(action)
    # Simple physics
    ...
end

# Compare states with tolerance
function states_approximately_equal(s1, s2; atol=1e-6)
    return norm(s1 - s2) < atol
end
```

## Debugging Failed Tests

### Check Agent State

```julia
# Print agent diagnostics
print_status(agent)

# Get detailed diagnostics
diag = get_diagnostics(agent)
@show diag
```

### Visualize Behavior

```julia
# Plot belief evolution
plot(belief_history)

# Plot action sequence
plot(action_history)
```

### Isolate Problem

```julia
# Test individual components
@test_nowarn transition_func(state)
@test_nowarn control_func(action)
```

## Best Practices

1. **Use Tolerances**: Floating-point comparisons need tolerances
2. **Test Probabilistic Behavior**: Run multiple trials for stochastic tests
3. **Mock Complex Dependencies**: Use simple physics for unit tests
4. **Test Incrementally**: Start with simple cases, add complexity
5. **Document Assumptions**: Note what behavior is expected and why

---

**Comprehensive testing ensures robust, reliable Active Inference agents.**

