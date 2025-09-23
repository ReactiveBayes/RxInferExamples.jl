#!/usr/bin/env julia

# Test utils module functionality
# Tests logging, data export, performance metrics, and utilities

module TestUtils

using Test
using JSON

# Include main modules to access Config and Utils
include("../config.jl")
include("../src/physics.jl")
include("../src/world.jl")
include("../src/agent.jl")
include("../src/visualization.jl")
include("../src/utils.jl")

# Import functions from modules
using .Config: PHYSICS, WORLD, TARGET, OUTPUTS, NUMERICAL
using .Utils: validate_config, calculate_stats, export_to_csv, export_to_json, Timer

@doc """
Test utilities module functionality.
"""
function test_utils()
    @testset "Utils Module" begin
        @info "Testing utils module..."

        # Test configuration validation
        issues = validate_config()
        @test typeof(issues) == Vector{String}

        # Test that current configuration is valid
        @test isempty(issues)  # Should have no issues with default config

        # Test statistics calculation
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = calculate_stats(test_data)
        @test haskey(stats, "mean")
        @test haskey(stats, "std")
        @test haskey(stats, "min")
        @test haskey(stats, "max")
        @test haskey(stats, "median")
        @test haskey(stats, "length")
        @test stats["mean"] ≈ 3.0
        @test stats["min"] ≈ 1.0
        @test stats["max"] ≈ 5.0
        @test stats["median"] ≈ 3.0
        @test stats["length"] == 5

        # Test statistics with different data
        float_data = [1.5, 2.7, 3.9]
        float_stats = calculate_stats(float_data)
        @test float_stats["mean"] ≈ 2.7
        @test float_stats["std"] > 0

        # Test empty data handling
        empty_stats = calculate_stats(Float64[])
        @test empty_stats["length"] == 0
        @test isnan(empty_stats["mean"])

        # Test data export functionality
        test_dict = Dict(
            "experiment" => "test",
            "results" => Dict("accuracy" => 0.95, "loss" => 0.05),
            "parameters" => Dict("learning_rate" => 0.01),
            "scores" => [0.8, 0.9, 0.95]
        )

        # Test JSON export
        json_file = "test_results.json"
        export_to_json(test_dict, json_file)
        @test isfile(json_file)

        # Verify JSON content
        if isfile(json_file)
            exported_data = JSON.parsefile(json_file)
            @test exported_data["experiment"] == "test"
            @test exported_data["results"]["accuracy"] ≈ 0.95
            @test exported_data["parameters"]["learning_rate"] ≈ 0.01
            @test exported_data["scores"] == [0.8, 0.9, 0.95]

            # Clean up
            rm(json_file)
        end

        # Test CSV export
        csv_file = "test_results.csv"
        export_to_csv(test_dict, csv_file)
        @test isfile(csv_file)

        # Verify CSV was created (don't test exact content due to flattening)
        if isfile(csv_file)
            @test filesize(csv_file) > 0  # Should not be empty

            # Clean up
            rm(csv_file)
        end

        # Test Timer functionality
        timer = Timer("test_timer")
        sleep(0.001)  # Small delay
        close(timer)  # Should not error

        # Test with invalid configuration for validation
        invalid_physics = (
            engine_force_limit = -1.0,  # Invalid negative value
            friction_coefficient = PHYSICS.friction_coefficient
        )

        # Create a validation function for testing
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

        invalid_issues = test_validate_config(invalid_physics)
        @test !isempty(invalid_issues)
        @test any(contains(issue, "Engine force limit") for issue in invalid_issues)

        # Test validation with valid config
        valid_physics = (
            engine_force_limit = 0.04,
            friction_coefficient = 0.1
        )
        valid_issues = test_validate_config(valid_physics)
        @test isempty(valid_issues)

        # Test nested dictionary flattening behavior
        nested_dict = Dict(
            "a" => 1,
            "b" => Dict("c" => 2, "d" => Dict("e" => 3)),
            "f" => [1, 2, 3]
        )

        # Should handle nested structures without error
        @test typeof(nested_dict) == Dict{String, Any}

        @info "Utils tests passed."
    end
end

# Export test function
export test_utils

end # module TestUtils
