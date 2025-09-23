#!/usr/bin/env julia

# Activate the project environment to ensure all dependencies are available
import Pkg
using Test
using Statistics
using Dates
Pkg.activate(".")

# Modular test suite for Active Inference Mountain Car example
# Orchestrates individual test modules for comprehensive testing

@doc """
Modular test suite orchestrator for Active Inference Mountain Car.

This module imports and runs individual test modules to provide
comprehensive testing coverage while maintaining modularity and
maintainability.
"""

# Include main modules first for the test modules to import from
include("../config.jl")
include("../src/physics.jl")
include("../src/world.jl")
include("../src/agent.jl")
include("../src/visualization.jl")
include("../src/utils.jl")

# Include all test modules
include("test_config.jl")
include("test_physics.jl")
include("test_world.jl")
include("test_agent.jl")
include("test_visualization.jl")
include("test_utils.jl")
include("test_performance.jl")
include("test_integration.jl")
include("test_error_handling.jl")

# Import test functions from all modules
using .TestConfig: test_configuration
using .TestPhysics: test_physics
using .TestWorld: test_world
using .TestAgent: test_agent
using .TestVisualization: test_visualization
using .TestUtils: test_utils
using .TestPerformance: test_performance
using .TestIntegration: test_integration
using .TestErrorHandling: test_error_handling

@doc """
Run all modular tests.

This function orchestrates the execution of all individual test modules
and provides comprehensive reporting of test results.
"""
function run_all_tests()
    @info "Starting modular test suite for Active Inference Mountain Car..."

    # Run all test modules
    test_configuration()
    test_physics()
    test_world()
    test_agent()
    test_visualization()
    test_utils()
    test_performance()
    test_integration()
    test_error_handling()

    @info "🎉 All modular tests passed successfully!"
end

@doc """
Run individual test module.

Args:
- module_name: Symbol name of test module to run (:config, :physics, :world, :agent, :visualization, :utils, :performance, :integration, :error_handling)
"""
function run_test_module(module_name::Symbol)
    @info "Running individual test module: $module_name"

    if module_name == :config
        test_configuration()
    elseif module_name == :physics
        test_physics()
    elseif module_name == :world
        test_world()
    elseif module_name == :agent
        test_agent()
    elseif module_name == :visualization
        test_visualization()
    elseif module_name == :utils
        test_utils()
    elseif module_name == :performance
        test_performance()
    elseif module_name == :integration
        test_integration()
    elseif module_name == :error_handling
        test_error_handling()
    else
        @error "Unknown test module: $module_name"
        throw(ArgumentError("Unknown test module: $module_name"))
    end

    @info "✅ Test module $module_name completed successfully!"
end

@doc """
List available test modules.
"""
function list_test_modules()
    println("Available test modules:")
    modules = [:config, :physics, :world, :agent, :visualization, :utils, :performance, :integration, :error_handling]
    for module_name in modules
        println("  - $module_name")
    end
end

# Command line interface
if length(ARGS) > 0
    if ARGS[1] == "--list"
        list_test_modules()
    elseif ARGS[1] == "--help"
        println("""
        Modular Test Suite for Active Inference Mountain Car

        Usage:
          julia runtests.jl                    # Run all tests
          julia runtests.jl --list            # List available test modules
          julia runtests.jl --help            # Show this help

        Individual test modules:
          config          - Configuration validation
          physics         - Physics calculations and forces
          world           - Environment state management
          agent           - Active inference agent functionality
          visualization   - Plotting and visualization features
          utils           - Utilities and data export
          performance     - Performance benchmarking
          integration     - Cross-module integration
          error_handling  - Error handling and edge cases
        """)
    else
        # Try to run specific module
        try
            run_test_module(Symbol(ARGS[1]))
        catch e
            @error "Failed to run test module: $(ARGS[1])"
            @error "Use --list to see available modules"
            rethrow(e)
        end
    end
else
    # Run all tests by default
    run_all_tests()
end

# Import functions from modules for the main test runner
using .Physics: create_physics, next_state, get_landscape_coordinates
using .World: create_world, simulate_trajectory
using .Agent: create_agent
using .Visualization: plot_landscape, height_at_position, get_color_scheme
using .Utils: validate_config, calculate_stats, export_to_csv, export_to_json
using .Config: PHYSICS, WORLD, TARGET, SIMULATION, AGENT, VISUALIZATION, OUTPUTS

# Run tests if this file is executed directly
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_all_tests()
end
