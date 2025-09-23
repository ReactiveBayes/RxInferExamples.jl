#!/usr/bin/env julia

# Test world module functionality
# Tests environment state management and trajectory simulation

module TestWorld

using Test

# Include main modules to access Config, Physics, and World
include("../config.jl")
include("../src/physics.jl")
include("../src/world.jl")
include("../src/agent.jl")
include("../src/visualization.jl")
include("../src/utils.jl")

# Import functions from modules
using .Config: PHYSICS, WORLD, TARGET
using .Physics: create_physics, next_state
using .World: create_world, simulate_trajectory

@doc """
Test world module functionality.
"""
function test_world()
    @testset "World Module" begin
        @info "Testing world module..."

        # Create physics functions
        Fa, Ff, Fg, height = create_physics()

        # Create world
        execute, observe, reset, get_state, set_state = create_world(
            Fg = Fg, Ff = Ff, Fa = Fa
        )

        # Test initial state
        initial_state = observe()
        @test initial_state[1] ≈ WORLD.initial_position
        @test initial_state[2] ≈ WORLD.initial_velocity
        @test length(initial_state) == 2
        @test typeof(initial_state) == Vector{Float64}

        # Test state transitions
        execute(0.1)
        new_state = observe()
        @test new_state[1] ≠ initial_state[1] || new_state[2] ≠ initial_state[2]
        @test length(new_state) == 2
        @test typeof(new_state) == Vector{Float64}

        # Test reset functionality
        reset()
        reset_state = observe()
        @test reset_state[1] ≈ WORLD.initial_position
        @test reset_state[2] ≈ WORLD.initial_velocity

        # Test get_state functionality
        state_copy = get_state()
        @test state_copy[1] ≈ reset_state[1]
        @test state_copy[2] ≈ reset_state[2]
        @test state_copy !== reset_state  # Should be different objects

        # Test set_state functionality
        set_state(1.0, 2.0)
        modified_state = observe()
        @test modified_state[1] ≈ 1.0
        @test modified_state[2] ≈ 2.0

        # Reset to original for trajectory test
        reset()

        # Test trajectory simulation
        actions = [0.0, 0.1, -0.1]
        states = simulate_trajectory(actions, 0.0, 0.0, Fa, Ff, Fg)
        @test length(states) == 4  # initial + 3 actions
        @test all(length(s) == 2 for s in states)
        @test all(typeof(s) == Vector{Float64} for s in states)

        # Test that states change with actions
        @test states[1] != states[2]  # Initial and first action should differ
        @test states[2] != states[3]  # Different actions should lead to different states
        @test states[3] != states[4]  # Third action should change state

        # Test trajectory with zero actions
        zero_actions = [0.0, 0.0, 0.0]
        zero_states = simulate_trajectory(zero_actions, 0.0, 0.0, Fa, Ff, Fg)
        @test length(zero_states) == 4
        @test all(length(s) == 2 for s in zero_states)

        # States should still change due to physics (gravity/friction)
        @test zero_states[1] != zero_states[4]

        # Test world creation with custom initial conditions
        custom_execute, custom_observe, custom_reset, custom_get_state, custom_set_state = create_world(
            Fg = Fg, Ff = Ff, Fa = Fa,
            initial_position = 1.0,
            initial_velocity = 0.5
        )

        custom_state = custom_observe()
        @test custom_state[1] ≈ 1.0
        @test custom_state[2] ≈ 0.5

        @info "World tests passed."
    end
end

# Export test function
export test_world

end # module TestWorld
