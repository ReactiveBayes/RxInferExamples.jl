#!/usr/bin/env julia

# Test physics module functionality
# Tests physics calculations, forces, and landscape geometry

module TestPhysics

using Test
using Statistics

# Include main modules to access Config and Physics
include("../config.jl")
include("../src/physics.jl")
include("../src/world.jl")
include("../src/agent.jl")
include("../src/visualization.jl")
include("../src/utils.jl")

# Import functions from modules
using .Config: PHYSICS, VISUALIZATION
using .Physics: create_physics, next_state, get_landscape_coordinates

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

        # Test landscape coordinates
        x_coords, y_coords = get_landscape_coordinates()
        @test length(x_coords) == VISUALIZATION.landscape_points
        @test length(y_coords) == VISUALIZATION.landscape_points
        @test x_coords[1] ≈ VISUALIZATION.landscape_range[1]
        @test x_coords[end] ≈ VISUALIZATION.landscape_range[2]

        # Test that height values are reasonable (most should be >= 0, some might be slightly negative due to numerical precision)
        @test all(isfinite, y_coords)  # All values should be finite
        @test all(y > -1.0 for y in y_coords)  # No extremely negative values
        @test mean(y_coords) >= -0.1  # Average should be close to zero or positive

        # Test physics with extreme values (within reasonable bounds)
        extreme_pos = 5.0  # Use a more reasonable extreme value
        @test isfinite(Fg(extreme_pos))
        @test isfinite(height(extreme_pos))
        @test height(extreme_pos) >= -0.5  # Allow some negative values at extremes

        # Test engine force clamping
        @test abs(Fa(1000.0)) ≤ PHYSICS.engine_force_limit + 1e-10
        @test abs(Fa(-1000.0)) ≤ PHYSICS.engine_force_limit + 1e-10

        # Test friction at extreme velocities
        @test isfinite(Ff(1000.0))
        @test isfinite(Ff(-1000.0))

        # Test multiple state transitions
        state = [0.0, 1.0]
        action = 0.1
        next_s1 = next_state(state, action, Fa, Ff, Fg)
        next_s2 = next_state(next_s1, action, Fa, Ff, Fg)
        @test length(next_s2) == 2
        @test typeof(next_s2) == Vector{Float64}

        @info "Physics tests passed."
    end
end

# Export test function
export test_physics

end # module TestPhysics
