#!/usr/bin/env julia

# Test integration between modules
# Tests cross-module functionality and end-to-end workflows

module TestIntegration

using Test

# Include main modules to access Config, Physics, World, and Agent
include("../config.jl")
include("../src/physics.jl")
include("../src/world.jl")
include("../src/agent.jl")
include("../src/visualization.jl")
include("../src/utils.jl")

# Import functions from modules
using .Config: PHYSICS, SIMULATION
using .Physics: create_physics
using .World: create_world, simulate_trajectory
using .Agent: create_agent

@doc """
Test integration between modules.
"""
function test_integration()
    @testset "Integration Tests" begin
        @info "Testing module integration..."

        # Test complete workflow
        Fa, Ff, Fg, height = create_physics()

        # Create world
        execute, observe, reset, get_state, set_state = create_world(
            Fg = Fg, Ff = Ff, Fa = Fa
        )

        # Create agent
        compute, act, slide, future, reset_agent = create_agent(
            T = 5,
            Fa = Fa, Ff = Ff, Fg = Fg
        )

        # Test initial state
        initial_state = observe()
        @test length(initial_state) == 2
        @test typeof(initial_state) == Vector{Float64}

        # Test initial action
        action = act()
        @test typeof(action) == Float64

        # Execute action and observe result
        execute(action)
        new_state = observe()
        @test length(new_state) == 2
        @test typeof(new_state) == Vector{Float64}
        # Note: Initial action might be 0.0, so state might not change immediately
        # This is expected behavior for the agent starting without inference results

        # Test predictions
        predictions = future()
        @test length(predictions) == 5
        @test typeof(predictions) == Vector{Float64}

        # Test inference cycle
        compute(action, new_state)
        inferred_action = act()
        @test typeof(inferred_action) == Float64

        # Test sliding window
        slide()
        slide_action = act()
        @test typeof(slide_action) == Float64

        # Test multiple steps of integration
        for i in 1:3
            execute(inferred_action)
            current_state = observe()
            compute(inferred_action, current_state)
            new_action = act()
            @test typeof(new_action) == Float64
            inferred_action = new_action
        end

        # Test trajectory simulation with physics
        test_actions = [0.0, 0.1, -0.1, 0.05]
        test_states = simulate_trajectory(test_actions, 0.0, 0.0, Fa, Ff, Fg)
        @test length(test_states) == 5  # initial + 4 actions
        @test all(length(s) == 2 for s in test_states)

        # States should change with different actions
        @test test_states[1] != test_states[2]  # Action should cause change
        @test test_states[2] != test_states[3]  # Different actions should lead to different states

        # Test agent-world integration
        set_state(-0.5, 0.0)  # Reset to initial state
        integration_state = observe()

        # Run integrated inference cycle
        for step in 1:3
            current_state = observe()
            agent_action = act()
            execute(agent_action)
            next_state = observe()
            compute(agent_action, next_state)

            @test typeof(agent_action) == Float64
            @test length(current_state) == 2
            @test length(next_state) == 2
        end

        # Test that physics constraints are respected
        extreme_state = observe()
        @test abs(extreme_state[1]) < 10.0  # Should not fly off to infinity
        @test abs(extreme_state[2]) < 5.0   # Velocity should be reasonable

        @info "Integration tests passed."
    end
end

# Export test function
export test_integration

end # module TestIntegration
