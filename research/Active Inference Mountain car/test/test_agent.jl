#!/usr/bin/env julia

# Test agent module functionality
# Tests active inference agent behavior and planning

module TestAgent

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
using .World: create_world
using .Agent: create_agent

@doc """
Test agent module functionality.
"""
function test_agent()
    @testset "Agent Module" begin
        @info "Testing agent module..."

        # Create physics functions
        Fa, Ff, Fg, height = create_physics()

        # Create agent with short horizon for testing
        compute, act, slide, future, reset = create_agent(
            T = 5,  # Short horizon for testing
            Fa = Fa, Ff = Ff, Fg = Fg
        )

        # Test initial action (should be 0.0 without inference)
        initial_action = act()
        @test typeof(initial_action) == Float64

        # Test future predictions (should return zeros initially)
        initial_predictions = future()
        @test length(initial_predictions) == 5
        @test all(p ≈ 0.0 for p in initial_predictions)

        # Test reset functionality
        reset()
        reset_action = act()
        @test typeof(reset_action) == Float64

        # Test with a simple inference scenario
        # Create a world to interact with
        execute, observe, world_reset, get_state, set_state = create_world(
            Fg = Fg, Ff = Ff, Fa = Fa
        )

        # Set a specific state
        set_state(0.0, 0.0)
        current_state = observe()

        # Perform inference
        compute(0.1, current_state)

        # Get action after inference
        inferred_action = act()
        @test typeof(inferred_action) == Float64

        # Get predictions
        predictions = future()
        @test length(predictions) == 5
        @test typeof(predictions) == Vector{Float64}

        # Test sliding window
        slide()
        slide_action = act()
        @test typeof(slide_action) == Float64

        # Test multiple inference steps
        for i in 1:3
            execute(inferred_action)
            new_state = observe()
            compute(inferred_action, new_state)
            new_action = act()
            @test typeof(new_action) == Float64
            inferred_action = new_action
        end

        # Test agent creation with custom parameters
        custom_compute, custom_act, custom_slide, custom_future, custom_reset = create_agent(
            T = 3,
            Fa = Fa, Ff = Ff, Fg = Fg,
            engine_force_limit = PHYSICS.engine_force_limit,
            x_target = [0.5, 0.0],
            initial_position = -0.5,
            initial_velocity = 0.0
        )

        custom_action = custom_act()
        @test typeof(custom_action) == Float64

        @info "Agent tests passed."
    end
end

# Export test function
export test_agent

end # module TestAgent
