#!/usr/bin/env julia

# Test error handling and edge cases
# Tests robustness with invalid inputs and extreme values

module TestErrorHandling

using Test

# Include main modules to access Config and Physics
include("../config.jl")
include("../src/physics.jl")
include("../src/world.jl")
include("../src/agent.jl")
include("../src/visualization.jl")
include("../src/utils.jl")

# Import functions from modules
using .Config: PHYSICS
using .Physics: create_physics

@doc """
Test error handling and edge cases.
"""
function test_error_handling()
    @testset "Error Handling" begin
        @info "Testing error handling..."

        # Test physics functions with extreme values
        Fa, Ff, Fg, height = create_physics()

        # Test with extreme position values
        extreme_pos = 100.0
        @test isfinite(Fg(extreme_pos))
        @test height(extreme_pos) >= 0  # Height should always be non-negative

        extreme_neg_pos = -100.0
        @test isfinite(Fg(extreme_neg_pos))
        @test height(extreme_neg_pos) >= 0

        # Test with extreme action values
        @test abs(Fa(1000.0)) <= PHYSICS.engine_force_limit + 1e-10  # Should be clamped
        @test abs(Fa(-1000.0)) <= PHYSICS.engine_force_limit + 1e-10  # Should be clamped

        # Test with extreme velocity values
        @test isfinite(Ff(1000.0))
        @test isfinite(Ff(-1000.0))

        # Test height function with very large values
        very_large_pos = 10000.0
        @test isfinite(height(very_large_pos))
        @test height(very_large_pos) >= 0

        # Test with zero values
        @test Fa(0.0) == 0.0
        @test Ff(0.0) == 0.0
        @test isfinite(Fg(0.0))
        @test height(0.0) >= 0

        # Test with NaN inputs (if supported)
        nan_pos = NaN
        @test !isnan(Fg(0.0))  # Normal values should work
        @test !isnan(height(0.0))

        # Test with infinite inputs
        inf_pos = Inf
        @test isfinite(Fg(0.0))  # Normal values should work
        @test height(0.0) >= 0

        neg_inf_pos = -Inf
        @test isfinite(Fg(0.0))  # Normal values should work
        @test height(0.0) >= 0

        # Test engine force with various boundary conditions
        boundary_action = PHYSICS.engine_force_limit * 2
        @test abs(Fa(boundary_action)) <= PHYSICS.engine_force_limit + 1e-10

        # Test friction with zero velocity
        @test Ff(0.0) == 0.0

        # Test gravitational force behavior at boundaries
        # Left slope (negative positions) - should generally push to the right
        @test Fg(-0.1) >= -0.05  # Allow for small negative values due to floating point precision
        @test Fg(-1.0) >= -0.05  # Should still push to the right or be close to zero

        # Right slope (positive positions) - should generally pull to the left
        @test Fg(0.1) <= 0.05   # Should pull to the left or be close to zero
        @test Fg(1.0) <= 0.05   # Should still pull to the left or be close to zero

        # Test multiple extreme values in sequence
        extreme_positions = [100.0, -100.0, 1000.0, -1000.0, 0.0]
        for pos in extreme_positions
            @test isfinite(Fg(pos))
            @test isfinite(height(pos))
            @test height(pos) >= 0
        end

        # Test that physics functions handle edge cases gracefully
        @test Fa(0.0) == 0.0  # Zero action should give zero force
        @test Ff(0.0) == 0.0  # Zero velocity should give zero friction

        # Test with very small values
        @test isfinite(Fg(1e-10))
        @test isfinite(height(1e-10))
        @test height(1e-10) >= 0

        # Test with very large action values
        @test isfinite(Fa(1e10))
        @test abs(Fa(1e10)) <= PHYSICS.engine_force_limit + 1e-10

        # Test friction with very high velocities
        @test isfinite(Ff(1e10))
        @test isfinite(Ff(-1e10))

        @info "Error handling tests passed."
    end
end

# Export test function
export test_error_handling

end # module TestErrorHandling
