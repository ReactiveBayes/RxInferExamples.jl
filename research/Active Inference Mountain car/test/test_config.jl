#!/usr/bin/env julia

# Test configuration module functionality
# Tests parameter validation and structure verification

module TestConfig

using Test

# Include main modules to access Config
include("../config.jl")
include("../src/physics.jl")
include("../src/world.jl")
include("../src/agent.jl")
include("../src/visualization.jl")
include("../src/utils.jl")

# Import functions from modules
using .Config: PHYSICS, WORLD, TARGET, SIMULATION, AGENT, VISUALIZATION, OUTPUTS, NUMERICAL

@doc """
Test configuration module functionality.
"""
function test_configuration()
    @testset "Configuration Module" begin
        @info "Testing configuration module..."

        # Test that all configuration sections exist
        @test haskey(PHYSICS, :engine_force_limit)
        @test haskey(PHYSICS, :friction_coefficient)
        @test haskey(WORLD, :initial_position)
        @test haskey(WORLD, :initial_velocity)
        @test haskey(WORLD, :target_position)
        @test haskey(WORLD, :target_velocity)
        @test haskey(TARGET, :position)
        @test haskey(TARGET, :velocity)
        @test haskey(SIMULATION, :planning_horizon)
        @test haskey(SIMULATION, :time_steps_naive)
        @test haskey(SIMULATION, :time_steps_ai)
        @test haskey(SIMULATION, :naive_action)
        @test haskey(AGENT, :transition_precision)
        @test haskey(AGENT, :observation_variance)
        @test haskey(AGENT, :control_prior_variance)
        @test haskey(AGENT, :goal_prior_variance)
        @test haskey(AGENT, :initial_state_variance)
        @test haskey(VISUALIZATION, :landscape_points)
        @test haskey(VISUALIZATION, :landscape_range)
        @test haskey(VISUALIZATION, :animation_fps)
        @test haskey(VISUALIZATION, :plot_size)
        @test haskey(VISUALIZATION, :engine_force_limits)
        @test haskey(OUTPUTS, :output_dir)
        @test haskey(OUTPUTS, :naive_animation)
        @test haskey(OUTPUTS, :ai_animation)
        @test haskey(OUTPUTS, :log_file)
        @test haskey(NUMERICAL, :epsilon)
        @test haskey(NUMERICAL, :tolerance)

        # Test that values are reasonable
        @test PHYSICS.engine_force_limit > 0
        @test PHYSICS.friction_coefficient > 0
        @test WORLD.initial_position < TARGET.position
        @test SIMULATION.planning_horizon > 0
        @test SIMULATION.time_steps_naive > 0
        @test SIMULATION.time_steps_ai > 0
        @test AGENT.transition_precision > 0
        @test AGENT.observation_variance > 0
        @test AGENT.control_prior_variance > 0
        @test AGENT.goal_prior_variance > 0
        @test AGENT.initial_state_variance > 0
        @test VISUALIZATION.landscape_points > 0
        @test length(VISUALIZATION.landscape_range) == 2
        @test VISUALIZATION.landscape_range[1] < VISUALIZATION.landscape_range[2]
        @test VISUALIZATION.animation_fps > 0
        @test VISUALIZATION.plot_size[1] > 0
        @test VISUALIZATION.plot_size[2] > 0
        @test VISUALIZATION.engine_force_limits[1] < VISUALIZATION.engine_force_limits[2]
        @test NUMERICAL.epsilon > 0
        @test NUMERICAL.tolerance > 0

        # Test derived configuration values
        @test TARGET.position == WORLD.target_position
        @test TARGET.velocity == WORLD.target_velocity

        @info "Configuration tests passed."
    end
end

# Export test function
export test_configuration

end # module TestConfig
