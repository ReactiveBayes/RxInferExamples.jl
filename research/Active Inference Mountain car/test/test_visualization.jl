#!/usr/bin/env julia

# Test visualization module functionality
# Tests plotting functions, themes, and visualization capabilities

module TestVisualization

using Test
using Statistics

# Include main modules to access Config, Physics, and Visualization
include("../config.jl")
include("../src/physics.jl")
include("../src/world.jl")
include("../src/agent.jl")
include("../src/visualization.jl")
include("../src/utils.jl")

# Import functions from modules
using .Config: VISUALIZATION, PHYSICS, TARGET, WORLD
using .Physics: create_physics, get_landscape_coordinates
using .Visualization: plot_landscape, height_at_position, get_color_scheme

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

        # Test height at various positions
        @test height_at_position(-1.0) ≥ 0
        @test height_at_position(0.0) ≥ 0
        @test height_at_position(1.0) ≥ 0
        @test height_at_position(10.0) ≥ 0  # Extreme position should still be non-negative

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

        # Test that plotting functions exist and are callable
        @test typeof(plot_landscape) <: Function

        # Test color scheme functionality
        colors = get_color_scheme(:default)
        @test haskey(colors, :landscape)
        @test haskey(colors, :car)
        @test haskey(colors, :goal)
        @test haskey(colors, :trajectory)
        @test haskey(colors, :predictions)

        # Test different themes
        dark_colors = get_color_scheme(:dark)
        @test dark_colors != colors  # Should be different

        colorblind_colors = get_color_scheme(:colorblind_friendly)
        @test colorblind_colors != colors
        @test colorblind_colors != dark_colors

        # Test that color schemes are consistent (same keys)
        @test keys(colors) == keys(dark_colors)
        @test keys(colors) == keys(colorblind_colors)

        # Test invalid theme (should fallback to default)
        fallback_colors = get_color_scheme(:invalid_theme)
        @test fallback_colors == colors

        # Test landscape coordinate range
        @test x_coords[1] < x_coords[end]
        @test length(unique(x_coords)) == length(x_coords)  # All x coordinates should be unique

        # Test height function behavior at boundaries
        @test isfinite(height_at_position(x_coords[1]))
        @test isfinite(height_at_position(x_coords[end]))

        # Test that landscape is not flat (should have variation)
        height_range = maximum(y_coords) - minimum(y_coords)
        @test height_range > 0  # Should have some height variation

        # Test color scheme structure
        required_keys = [:landscape, :car, :goal, :trajectory, :predictions, :uncertainty, :goal_region]
        for key in required_keys
            @test haskey(colors, key)
            @test haskey(dark_colors, key)
            @test haskey(colorblind_colors, key)
        end

        @info "Visualization tests passed."
    end
end

# Export test function
export test_visualization

end # module TestVisualization
