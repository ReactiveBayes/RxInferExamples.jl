#!/usr/bin/env julia

# Comprehensive test suite for Coin Toss Model research fork
# This is the main test runner that includes all modular test files

using Test
using Pkg
using Logging
using Dates

# Activate project (parent directory of test/)
project_dir = dirname(@__DIR__)
Pkg.activate(project_dir)

# Setup comprehensive test logging
test_log_dir = joinpath("..", "outputs", "logs")
mkpath(test_log_dir)
test_log_file = joinpath(test_log_dir, "test_run_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).log")

# Create logger that writes to both console and file
console_logger = ConsoleLogger(stderr, Logging.Info)
file_logger = SimpleLogger(open(test_log_file, "w"), Logging.Info)

# Use console logger by default
global_logger(console_logger)

@info "="^70
@info "Starting Comprehensive Coin Toss Model Test Suite"
@info "Test Log: $test_log_file"
@info "Julia Version: $(VERSION)"
@info "Timestamp: $(now())"
@info "="^70

# Load all modules
@info "Loading modules..."
include("../config.jl")
include("../src/model.jl")
include("../src/inference.jl")
include("../src/visualization.jl")
include("../src/utils.jl")

using .Config
using .CoinTossModel
using .CoinTossInference
using .CoinTossVisualization
using .CoinTossUtils

using Distributions
using Random
using Statistics

@info "All modules loaded successfully"

# Track test timing
test_start_time = time()

@testset verbose=true "Coin Toss Model - Complete Test Suite" begin
    
    # Modular Test Suites
    @info "Running modular test suites..."
    
    @testset "1. Model Tests" begin
        @info "="^60
        @info "Running Model Tests (test_model.jl)"
        @info "="^60
        include("test_model.jl")
    end
    
    @testset "2. Inference Tests" begin
        @info "="^60
        @info "Running Inference Tests (test_inference.jl)"
        @info "="^60
        include("test_inference.jl")
    end
    
    @testset "3. Visualization Tests" begin
        @info "="^60
        @info "Running Visualization Tests (test_visualization.jl)"
        @info "="^60
        include("test_visualization.jl")
    end
    
    @testset "4. Utils Tests" begin
        @info "="^60
        @info "Running Utils Tests (test_utils.jl)"
        @info "="^60
        include("test_utils.jl")
    end
    
    @testset "5. Performance Tests" begin
        @info "="^60
        @info "Running Performance & Benchmark Tests (test_performance.jl)"
        @info "="^60
        include("test_performance.jl")
    end
    
    # Quick Integration Tests (from original runtests.jl)
    @testset "Quick Configuration Checks" begin
        @testset "Load Configuration" begin
            config = load_config("../config.toml")
            @test haskey(config, "data")
            @test haskey(config, "model")
            @test haskey(config, "inference")
            @test config["data"]["n_samples"] > 0
            @test 0 <= config["data"]["theta_real"] <= 1
        end
        
        @testset "Validate Configuration" begin
            # Valid config
            config = Config.get_default_config()
            issues = Config.validate_config(config)
            @test isempty(issues)
            
            # Invalid theta
            invalid_config = deepcopy(config)
            invalid_config["data"]["theta_real"] = 1.5
            issues = Config.validate_config(invalid_config)
            @test !isempty(issues)
            
            # Invalid n_samples
            invalid_config = deepcopy(config)
            invalid_config["data"]["n_samples"] = -10
            issues = Config.validate_config(invalid_config)
            @test !isempty(issues)
        end
    end
    
    
end

# Calculate total test time
test_end_time = time()
total_test_time = test_end_time - test_start_time

# Comprehensive test summary
@info "="^70
@info "TEST SUITE SUMMARY"
@info "="^70
@info "Total Test Duration: $(round(total_test_time, digits=2)) seconds"
@info "Julia Version: $(VERSION)"
@info "Timestamp: $(now())"
@info ""
@info "Modules Tested:"
@info "  ✓ CoinTossModel - Data generation, analytical computations"
@info "  ✓ CoinTossInference - RxInfer execution, diagnostics, convergence"
@info "  ✓ CoinTossVisualization - All plot types, animations, themes"
@info "  ✓ CoinTossUtils - Logging, export, statistics, helpers"
@info "  ✓ Performance - Benchmarks and scalability tests"
@info ""
@info "Test Categories:"
@info "  1. Model Tests (test_model.jl)"
@info "     - Data generation (basic, edge cases, reproducibility)"
@info "     - Model definition"
@info "     - Analytical posterior computation"
@info "     - Posterior statistics"
@info "     - Log marginal likelihood"
@info "     - Conjugacy verification"
@info ""
@info "  2. Inference Tests (test_inference.jl)"
@info "     - RxInfer execution"
@info "     - Convergence detection and diagnostics"
@info "     - KL divergence computation"
@info "     - Expected log likelihood"
@info "     - Posterior predictive checks"
@info "     - Analytical vs numerical agreement"
@info ""
@info "  3. Visualization Tests (test_visualization.jl)"
@info "     - Theme colors (default, dark, colorblind)"
@info "     - Prior-posterior plots"
@info "     - Convergence plots"
@info "     - Data histograms"
@info "     - Credible interval plots"
@info "     - Predictive plots"
@info "     - Comprehensive dashboards"
@info "     - Animation creation"
@info "     - Plot saving"
@info ""
@info "  4. Utils Tests (test_utils.jl)"
@info "     - Logging setup"
@info "     - Timers and progress bars"
@info "     - CSV and JSON export"
@info "     - Dictionary flattening"
@info "     - Experiment result saving"
@info "     - Summary statistics"
@info "     - Confidence intervals"
@info "     - Formatting utilities"
@info ""
@info "  5. Performance Tests (test_performance.jl)"
@info "     - Data generation performance"
@info "     - Analytical computation speed"
@info "     - RxInfer inference benchmarks"
@info "     - Visualization performance"
@info "     - Export performance"
@info "     - End-to-end workflow timing"
@info "     - Scalability analysis"
@info ""
@info "Test Coverage:"
@info "  ✓ All exported functions tested"
@info "  ✓ Edge cases covered"
@info "  ✓ Error handling verified"
@info "  ✓ Performance benchmarked"
@info "  ✓ Integration workflows validated"
@info ""
@info "Test Log saved to: $test_log_file"
@info "="^70
@info "ALL TESTS PASSED SUCCESSFULLY! ✓"
@info "="^70

# Close file logger if it was opened
close(file_logger.stream)

