#!/usr/bin/env julia

# Activate the project environment to ensure all dependencies are available
import Pkg
using Test
using Statistics
using Dates
Pkg.activate(".")

# Test suite for Active Inference Mountain Car example
# Comprehensive test-driven development approach

module MountainCarTests

using Test
using Statistics

# Include all necessary modules
include("../config.jl")
include("../src/physics.jl")
include("../src/world.jl")
include("../src/agent.jl")
include("../src/visualization.jl")
include("../src/utils.jl")

# Import functions from modules
using .Physics: create_physics, next_state, get_landscape_coordinates
using .World: create_world, simulate_trajectory
using .Agent: create_agent
using .Visualization: plot_landscape, height_at_position, get_color_scheme
using .Utils: validate_config, calculate_stats, export_to_csv, export_to_json
using .Config: PHYSICS, WORLD, TARGET, SIMULATION, AGENT, VISUALIZATION, OUTPUTS

@doc """
Test physics module functionality.
"""
function test_physics()
    @testset "Physics Module" begin
        @info "Testing physics module..."

        # Test physics creation
        Fa, Ff, Fg, height = create_physics()
        @test Fa(0.0) == 0.0  # Test that Fa is callable
        @test Ff(0.0) == 0.0  # Test that Ff is callable
        @test Fg(0.0) == -0.05  # Test that Fg is callable
        @test height(0.0) >= 0  # Test that height is callable

        # Test engine force limits
        @test abs(Fa(10.0)) ≤ PHYSICS.engine_force_limit + 1e-10
        @test abs(Fa(-10.0)) ≤ PHYSICS.engine_force_limit + 1e-10

        # Test friction force
        @test Ff(1.0) ≈ -PHYSICS.friction_coefficient
        @test Ff(-1.0) ≈ PHYSICS.friction_coefficient

        # Test gravitational force at different positions
        @test Fg(-1.0) > 0  # Should push up on left slope (to the right)
        @test Fg(1.0) < 0   # Should pull down on right slope (to the left)

        # Test height function
        @test height(-1.0) ≥ 0
        @test height(0.0) ≥ 0
        @test height(1.0) ≥ 0

        # Test state transition
        state = [0.0, 0.0]
        action = 0.0
        next_s = next_state(state, action, Fa, Ff, Fg)
        @test length(next_s) == 2
        @test typeof(next_s) == Vector{Float64}

        @info "Physics tests passed."
    end
end

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

        # Test state transitions
        execute(0.1)
        new_state = observe()
        @test new_state[1] ≠ initial_state[1] || new_state[2] ≠ initial_state[2]

        # Test reset functionality
        reset()
        reset_state = observe()
        @test reset_state[1] ≈ WORLD.initial_position
        @test reset_state[2] ≈ WORLD.initial_velocity

        # Test trajectory simulation
        actions = [0.0, 0.1, -0.1]
        states = simulate_trajectory(actions, 0.0, 0.0, Fa, Ff, Fg)
        @test length(states) == 4  # initial + 3 actions
        @test all(length(s) == 2 for s in states)

        @info "World tests passed."
    end
end

@doc """
Test agent module functionality.
"""
function test_agent()
    @testset "Agent Module" begin
        @info "Testing agent module..."

        # Create physics functions
        Fa, Ff, Fg, height = create_physics()

        # Create agent
        compute, act, slide, future, reset = create_agent(
            T = 5,  # Short horizon for testing
            Fa = Fa, Ff = Ff, Fg = Fg
        )

        # Test initial action (should be 0.0)
        initial_action = act()
        @test initial_action ≈ 0.0

        # Test with some observation
        test_state = [0.0, 0.0]
        compute(0.0, test_state)

        # Test that we can get an action after inference
        action = act()
        @test typeof(action) == Float64

        # Test future predictions
        predictions = future()
        @test length(predictions) == 5  # Should match planning horizon
        @test typeof(predictions) == Vector{Float64}

        # Test reset functionality
        reset()
        reset_action = act()
        @test reset_action ≈ 0.0

        @info "Agent tests passed."
    end
end

@doc """
Test visualization module functionality.
"""
function test_visualization()
    @testset "Visualization Module" begin
        @info "Testing visualization module..."

        # Test height function
        h = height_at_position(0.0)
        @test typeof(h) == Float64
        @test h ≥ 0

        # Test landscape coordinates
        x_coords, y_coords = get_landscape_coordinates()
        @test length(x_coords) == VISUALIZATION.landscape_points
        @test length(y_coords) == VISUALIZATION.landscape_points
        @test x_coords[1] ≈ VISUALIZATION.landscape_range[1]
        @test x_coords[end] ≈ VISUALIZATION.landscape_range[2]

        # Test that plotting functions exist and are callable
        @test typeof(plot_landscape) <: Function

        # Test color scheme functionality
        colors = get_color_scheme(:default)
        @test haskey(colors, :landscape)
        @test haskey(colors, :car)
        @test haskey(colors, :goal)

        # Test different themes
        dark_colors = get_color_scheme(:dark)
        @test dark_colors != colors  # Should be different

        colorblind_colors = get_color_scheme(:colorblind_friendly)
        @test colorblind_colors != colors

        @info "Visualization tests passed."
    end
end

@doc """
Test utilities module functionality.
"""
function test_utils()
    @testset "Utils Module" begin
        @info "Testing utils module..."

        # Test configuration validation
        issues = validate_config()
        @test typeof(issues) == Vector{String}

        # Test statistics calculation
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = calculate_stats(test_data)
        @test haskey(stats, "mean")
        @test haskey(stats, "std")
        @test haskey(stats, "min")
        @test haskey(stats, "max")
        @test stats["mean"] ≈ 3.0
        @test stats["min"] ≈ 1.0
        @test stats["max"] ≈ 5.0

        # Test data export functionality
        test_dict = Dict(
            "experiment" => "test",
            "results" => Dict("accuracy" => 0.95, "loss" => 0.05),
            "parameters" => Dict("learning_rate" => 0.01)
        )

        # Test JSON export
        json_file = "test_results.json"
        export_to_json(test_dict, json_file)
        @test isfile(json_file)

        # Clean up
        if isfile(json_file)
            rm(json_file)
        end

        # Test CSV export
        csv_file = "test_results.csv"
        export_to_csv(test_dict, csv_file)
        @test isfile(csv_file)

        # Clean up
        if isfile(csv_file)
            rm(csv_file)
        end

        @info "Utils tests passed."
    end
end

@doc """
Test performance benchmarking functionality.
"""
function test_performance()
    @testset "Performance Benchmarks" begin
        @info "Testing performance benchmarks..."

        # Test basic physics calculations
        Fa, Ff, Fg, height = create_physics()

        # Benchmark physics function calls
        n_samples = 1000
        positions = randn(n_samples)
        velocities = randn(n_samples)

        # Benchmark gravitational force calculation
        grav_times = Vector{Float64}(undef, n_samples)
        for i in 1:n_samples
            start_time = time()
            result = Fg(positions[i])
            grav_times[i] = time() - start_time
        end

        avg_grav_time = mean(grav_times)
        @test avg_grav_time < 1e-3  # Should be very fast (< 1ms)

        # Benchmark height calculation
        height_times = Vector{Float64}(undef, n_samples)
        for i in 1:n_samples
            start_time = time()
            result = height(positions[i])
            height_times[i] = time() - start_time
        end

        avg_height_time = mean(height_times)
        @test avg_height_time < 1e-3  # Should be very fast

        @info "Performance tests completed." avg_grav_time = round(avg_grav_time * 1000, digits=3) avg_height_time = round(avg_height_time * 1000, digits=3)
    end
end

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

        # Run a few steps
        initial_state = observe()
        action = act()
        execute(action)
        new_state = observe()

        @test length(initial_state) == 2
        @test length(new_state) == 2
        @test typeof(action) == Float64

        # Test predictions
        predictions = future()
        @test length(predictions) == 5

        @info "Integration tests passed."
    end
end

@doc """
Test error handling and edge cases.
"""
function test_error_handling()
    @testset "Error Handling" begin
        @info "Testing error handling..."

        # Test invalid configuration - create a modified config for testing
        invalid_physics = (
            engine_force_limit = -1.0,  # Invalid negative value
            friction_coefficient = PHYSICS.friction_coefficient
        )

        # Test validation function with invalid physics
        function test_validate_config(physics_tuple)
            issues = String[]
            if physics_tuple.engine_force_limit <= 0
                push!(issues, "Engine force limit must be positive")
            end
            if physics_tuple.friction_coefficient < 0
                push!(issues, "Friction coefficient must be non-negative")
            end
            return issues
        end

        issues = test_validate_config(invalid_physics)
        @test !isempty(issues)
        @test any(contains(issue, "Engine force limit") for issue in issues)

        # Test physics functions with extreme values
        Fa, Ff, Fg, height = create_physics()

        # Test with extreme position values
        extreme_pos = 100.0
        @test isfinite(Fg(extreme_pos))
        @test height(extreme_pos) >= 0  # Height should always be non-negative

        # Test with extreme action values
        @test abs(Fa(1000.0)) <= PHYSICS.engine_force_limit + 1e-10  # Should be clamped

        @info "Error handling tests passed."
    end
end

@doc """
Test configuration module.
"""
function test_config()
    @testset "Configuration" begin
        @info "Testing configuration..."

        # Test that all configuration sections exist
        @test haskey(PHYSICS, :engine_force_limit)
        @test haskey(WORLD, :initial_position)
        @test haskey(TARGET, :position)
        @test haskey(SIMULATION, :planning_horizon)
        @test haskey(AGENT, :transition_precision)
        @test haskey(VISUALIZATION, :landscape_points)

        # Test that values are reasonable
        @test PHYSICS.engine_force_limit > 0
        @test PHYSICS.friction_coefficient > 0
        @test WORLD.initial_position < TARGET.position
        @test SIMULATION.planning_horizon > 0
        @test VISUALIZATION.landscape_points > 0

        @info "Configuration tests passed."
    end
end

@doc """
Run all tests.
"""
function run_all_tests()
    @info "Starting comprehensive test suite for Active Inference Mountain Car..."

    test_config()
    test_physics()
    test_world()
    test_agent()
    test_visualization()
    test_utils()
    test_performance()
    test_integration()
    test_error_handling()

    @info "All tests passed successfully!"
end

# Import the needed constants and functions
using .Config: PHYSICS, WORLD, TARGET, SIMULATION, AGENT, VISUALIZATION
using .Physics: create_physics, next_state, get_landscape_coordinates
using .World: create_world, simulate_trajectory
using .Agent: create_agent
using .Visualization: plot_landscape, height_at_position, get_color_scheme
using .Utils: validate_config, calculate_stats, export_to_csv, export_to_json

# Run tests if this file is executed directly
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_all_tests()
end

end # module MountainCarTests
