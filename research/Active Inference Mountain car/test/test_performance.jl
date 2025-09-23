#!/usr/bin/env julia

# Test performance benchmarking functionality
# Tests computational efficiency and performance characteristics

module TestPerformance

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
using .Config: PHYSICS
using .Physics: create_physics

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
        actions = randn(n_samples) * 0.1

        # Benchmark gravitational force calculation
        grav_times = Vector{Float64}(undef, n_samples)
        grav_results = Vector{Float64}(undef, n_samples)
        for i in 1:n_samples
            start_time = time()
            result = Fg(positions[i])
            grav_times[i] = time() - start_time
            grav_results[i] = result
        end

        avg_grav_time = mean(grav_times)
        @test avg_grav_time < 1e-3  # Should be very fast (< 1ms)
        @test all(isfinite, grav_results)  # All results should be finite

        # Benchmark height calculation
        height_times = Vector{Float64}(undef, n_samples)
        height_results = Vector{Float64}(undef, n_samples)
        for i in 1:n_samples
            start_time = time()
            result = height(positions[i])
            height_times[i] = time() - start_time
            height_results[i] = result
        end

        avg_height_time = mean(height_times)
        @test avg_height_time < 1e-3  # Should be very fast
        @test all(isfinite, height_results)  # All results should be finite
        @test all(r >= -1.0 for r in height_results)  # Allow some slightly negative values at boundaries

        # Benchmark engine force calculation
        engine_times = Vector{Float64}(undef, n_samples)
        engine_results = Vector{Float64}(undef, n_samples)
        for i in 1:n_samples
            start_time = time()
            result = Fa(actions[i])
            engine_times[i] = time() - start_time
            engine_results[i] = result
        end

        avg_engine_time = mean(engine_times)
        @test avg_engine_time < 1e-3  # Should be very fast
        @test all(isfinite, engine_results)  # All results should be finite
        @test all(abs(r) <= PHYSICS.engine_force_limit + 1e-10 for r in engine_results)  # Should respect limits

        # Benchmark friction force calculation
        friction_times = Vector{Float64}(undef, n_samples)
        friction_results = Vector{Float64}(undef, n_samples)
        for i in 1:n_samples
            start_time = time()
            result = Ff(velocities[i])
            friction_times[i] = time() - start_time
            friction_results[i] = result
        end

        avg_friction_time = mean(friction_times)
        @test avg_friction_time < 1e-3  # Should be very fast
        @test all(isfinite, friction_results)  # All results should be finite

        # Test performance with extreme values
        extreme_positions = [100.0, -100.0, 1000.0, -1000.0]
        extreme_times = Vector{Float64}(undef, length(extreme_positions))
        extreme_height_results = Vector{Float64}(undef, length(extreme_positions))

        for (i, pos) in enumerate(extreme_positions)
            start_time = time()
            result = height(pos)
            extreme_times[i] = time() - start_time
            extreme_height_results[i] = result
        end

        @test all(t < 1e-3 for t in extreme_times)  # Should still be fast with extreme values
        @test all(isfinite, extreme_height_results)  # Should handle extremes gracefully
        @test all(r >= 0 for r in extreme_height_results)  # Heights should remain non-negative

        # Performance statistics
        all_times = vcat(grav_times, height_times, engine_times, friction_times)
        avg_time = mean(all_times)
        max_time = maximum(all_times)
        min_time = minimum(all_times)

        @test avg_time < 1e-3  # Average should be very fast
        @test max_time < 1e-2  # Maximum should still be reasonable
        @test min_time >= 0    # No negative times

        @info "Performance tests completed." avg_grav_time = round(avg_grav_time * 1000, digits=3) avg_height_time = round(avg_height_time * 1000, digits=3) avg_engine_time = round(avg_engine_time * 1000, digits=3) avg_friction_time = round(avg_friction_time * 1000, digits=3)
    end
end

# Export test function
export test_performance

end # module TestPerformance
