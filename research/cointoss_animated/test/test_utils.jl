#!/usr/bin/env julia
# Comprehensive tests for CoinTossUtils module

using Test
using DataFrames
using CSV
using JSON
using Statistics
using Logging

# Setup test logging
test_logger = ConsoleLogger(stderr, Logging.Info)
global_logger(test_logger)

@info "Starting CoinTossUtils tests"

@testset "CoinTossUtils Module Tests" begin
    
    @testset "Logging Setup" begin
        @info "Testing logging setup"
        
        # Test basic setup
        @test_nowarn setup_logging(verbose=true, structured=false, performance=false)
        @test_nowarn setup_logging(verbose=false, structured=false, performance=false)
        
        # Test with log file
        temp_dir = mktempdir()
        log_file = joinpath(temp_dir, "test.log")
        @test_nowarn setup_logging(verbose=true, log_file=log_file)
        
        # Cleanup
        rm(temp_dir, recursive=true)
        
        @info "Logging setup tests passed"
    end
    
    @testset "Timer - Basic" begin
        @info "Testing Timer basic functionality"
        
        timer = CoinTossUtils.Timer("test_operation")
        @test timer isa CoinTossUtils.Timer
        @test timer.name == "test_operation"
        @test timer.start_time > 0
        @test timer.end_time === nothing
        
        sleep(0.1)
        elapsed = close(timer)
        
        @test elapsed >= 0.1
        @test elapsed < 0.2
        @test timer.end_time !== nothing
        
        @info "Timer basic tests passed"
    end
    
    @testset "Timer - Elapsed Time" begin
        @info "Testing timer elapsed_time function"
        
        timer = CoinTossUtils.Timer("test")
        
        # Check elapsed time while running
        sleep(0.05)
        elapsed_running = elapsed_time(timer)
        @test elapsed_running >= 0.05
        @test elapsed_running < 0.1
        
        # Close timer
        close(timer)
        
        # Check elapsed time after closing
        elapsed_stopped = elapsed_time(timer)
        @test elapsed_stopped >= 0.05
        
        @info "Timer elapsed_time tests passed"
    end
    
    @testset "Timer - Multiple Timers" begin
        @info "Testing multiple concurrent timers"
        
        timer1 = CoinTossUtils.Timer("operation1")
        sleep(0.05)
        timer2 = CoinTossUtils.Timer("operation2")
        sleep(0.05)
        
        elapsed1 = close(timer1)
        elapsed2 = close(timer2)
        
        @test elapsed1 > elapsed2
        @test elapsed1 >= 0.1
        @test elapsed2 >= 0.05
        
        @info "Multiple timers tests passed"
    end
    
    @testset "ProgressBar - Basic" begin
        @info "Testing ProgressBar basic functionality"
        
        pb = ProgressBar(10; desc="Test Progress")
        @test pb isa ProgressBar
        
        # Update progress
        for i in 1:10
            update!(pb, i)
        end
        
        # Finish progress
        finish!(pb)
        
        @info "ProgressBar basic tests passed"
    end
    
    @testset "ProgressBar - Various Total Steps" begin
        @info "Testing ProgressBar with various totals"
        
        # Small total
        pb1 = ProgressBar(5; desc="Small")
        for i in 1:5
            update!(pb1, i)
        end
        finish!(pb1)
        @test true  # If no error, test passed
        
        # Large total
        pb2 = ProgressBar(1000; desc="Large")
        for i in 1:10:1000
            update!(pb2, i)
        end
        finish!(pb2)
        @test true
        
        @info "ProgressBar various totals tests passed"
    end
    
    @testset "Export to CSV - Basic" begin
        @info "Testing CSV export"
        
        data = Dict(
            "a" => 1,
            "b" => 2.5,
            "c" => "text",
            "d" => true
        )
        
        temp_dir = mktempdir()
        filepath = joinpath(temp_dir, "test_export.csv")
        
        export_to_csv(data, filepath)
        
        @test isfile(filepath)
        
        # Read back and verify
        df = CSV.read(filepath, DataFrame)
        @test "a" in df.key
        @test "b" in df.key
        
        # Cleanup
        rm(temp_dir, recursive=true)
        
        @info "CSV export basic tests passed"
    end
    
    @testset "Export to CSV - Nested Dictionaries" begin
        @info "Testing CSV export with nested dictionaries"
        
        nested_data = Dict(
            "level1" => Dict(
                "level2" => Dict(
                    "value" => 42
                ),
                "other" => 3.14
            ),
            "simple" => 100
        )
        
        temp_dir = mktempdir()
        filepath = joinpath(temp_dir, "nested_export.csv")
        
        export_to_csv(nested_data, filepath)
        
        @test isfile(filepath)
        
        # Read and check flattened keys
        df = CSV.read(filepath, DataFrame)
        @test "level1.level2.value" in df.key
        @test "level1.other" in df.key
        @test "simple" in df.key
        
        # Cleanup
        rm(temp_dir, recursive=true)
        
        @info "CSV nested export tests passed"
    end
    
    @testset "Export to JSON - Basic" begin
        @info "Testing JSON export"
        
        data = Dict(
            "number" => 42,
            "float" => 3.14,
            "string" => "hello",
            "bool" => true,
            "array" => [1, 2, 3]
        )
        
        temp_dir = mktempdir()
        filepath = joinpath(temp_dir, "test_export.json")
        
        export_to_json(data, filepath)
        
        @test isfile(filepath)
        
        # Read back and verify
        loaded = JSON.parsefile(filepath)
        @test loaded["number"] == 42
        @test loaded["string"] == "hello"
        
        # Cleanup
        rm(temp_dir, recursive=true)
        
        @info "JSON export basic tests passed"
    end
    
    @testset "Export to JSON - Nested Structures" begin
        @info "Testing JSON export with nested structures"
        
        nested_data = Dict(
            "metadata" => Dict(
                "version" => "1.0",
                "author" => "test"
            ),
            "results" => Dict(
                "values" => [1.1, 2.2, 3.3],
                "stats" => Dict(
                    "mean" => 2.2,
                    "std" => 0.9
                )
            )
        )
        
        temp_dir = mktempdir()
        filepath = joinpath(temp_dir, "nested_export.json")
        
        export_to_json(nested_data, filepath; indent=4)
        
        @test isfile(filepath)
        
        # Read back and verify structure preserved
        loaded = JSON.parsefile(filepath)
        @test loaded["metadata"]["version"] == "1.0"
        @test loaded["results"]["stats"]["mean"] == 2.2
        
        # Cleanup
        rm(temp_dir, recursive=true)
        
        @info "JSON nested export tests passed"
    end
    
    @testset "Dictionary Flattening - Basic" begin
        @info "Testing dictionary flattening"
        
        nested = Dict(
            "a" => 1,
            "b" => Dict(
                "c" => 2,
                "d" => 3
            )
        )
        
        flat = CoinTossUtils.flatten_dict(nested)
        
        @test haskey(flat, "a")
        @test haskey(flat, "b.c")
        @test haskey(flat, "b.d")
        @test flat["a"] == 1
        @test flat["b.c"] == 2
        @test flat["b.d"] == 3
        
        @info "Dictionary flattening basic tests passed"
    end
    
    @testset "Dictionary Flattening - Deep Nesting" begin
        @info "Testing deep nested dictionary flattening"
        
        deep_nested = Dict(
            "level1" => Dict(
                "level2" => Dict(
                    "level3" => Dict(
                        "value" => 42
                    ),
                    "other" => 10
                ),
                "simple" => 5
            )
        )
        
        flat = CoinTossUtils.flatten_dict(deep_nested)
        
        @test haskey(flat, "level1.level2.level3.value")
        @test haskey(flat, "level1.level2.other")
        @test haskey(flat, "level1.simple")
        @test flat["level1.level2.level3.value"] == 42
        
        @info "Deep nesting flattening tests passed"
    end
    
    @testset "Dictionary Flattening - Arrays" begin
        @info "Testing dictionary flattening with arrays"
        
        with_arrays = Dict(
            "simple" => 1,
            "items" => [
                Dict("id" => 1, "value" => "a"),
                Dict("id" => 2, "value" => "b")
            ]
        )
        
        flat = CoinTossUtils.flatten_dict(with_arrays)
        
        @test haskey(flat, "simple")
        @test haskey(flat, "items[1].id")
        @test haskey(flat, "items[1].value")
        @test haskey(flat, "items[2].id")
        @test flat["items[1].id"] == 1
        @test flat["items[2].value"] == "b"
        
        @info "Array flattening tests passed"
    end
    
    @testset "Save Experiment Results - Basic" begin
        @info "Testing save_experiment_results"
        
        results = Dict(
            "parameter" => 0.75,
            "score" => 42.5,
            "metrics" => Dict(
                "accuracy" => 0.95,
                "precision" => 0.92
            )
        )
        
        # Save to temp location
        original_dir = pwd()
        temp_dir = mktempdir()
        cd(temp_dir)
        mkpath("outputs/results")
        
        results_dir = save_experiment_results("test_experiment", results)
        
        @test isdir(results_dir)
        @test isfile(joinpath(results_dir, "results.json"))
        @test isfile(joinpath(results_dir, "results.csv"))
        @test isfile(joinpath(results_dir, "metadata.json"))
        
        # Verify metadata
        metadata = JSON.parsefile(joinpath(results_dir, "metadata.json"))
        @test metadata["experiment_name"] == "test_experiment"
        @test haskey(metadata, "timestamp")
        @test haskey(metadata, "julia_version")
        
        # Cleanup
        cd(original_dir)
        rm(temp_dir, recursive=true)
        
        @info "Save experiment results tests passed"
    end
    
    @testset "Ensure Directories" begin
        @info "Testing ensure_directories"
        
        temp_dir = mktempdir()
        
        config = Dict(
            "output" => Dict(
                "plots_dir" => joinpath(temp_dir, "plots"),
                "data_dir" => joinpath(temp_dir, "data"),
                "results_dir" => joinpath(temp_dir, "results")
            )
        )
        
        ensure_directories(config)
        
        @test isdir(joinpath(temp_dir, "plots"))
        @test isdir(joinpath(temp_dir, "data"))
        @test isdir(joinpath(temp_dir, "results"))
        
        # Test with no output config
        config_no_output = Dict("other" => "value")
        @test_logs (:warn, r"No output configuration") ensure_directories(config_no_output)
        
        # Cleanup
        rm(temp_dir, recursive=true)
        
        @info "Ensure directories tests passed"
    end
    
    @testset "Log Dictionary" begin
        @info "Testing log_dict function"
        
        test_dict = Dict(
            "a" => 1,
            "b" => Dict(
                "c" => 2,
                "d" => 3
            )
        )
        
        # Should log without error
        @test_nowarn log_dict(test_dict)
        @test_nowarn log_dict(test_dict; prefix="test")
        
        @info "Log dictionary tests passed"
    end
    
    @testset "Format Time - Various Durations" begin
        @info "Testing format_time"
        
        # Seconds
        @test format_time(45.5) == "45.50s"
        @test format_time(0.123) == "0.12s"
        
        # Minutes
        @test format_time(90.0) == "1m 30.00s"
        @test format_time(125.5) == "2m 5.50s"
        
        # Hours
        @test format_time(3661.0) == "1h 1m 1.00s"
        @test format_time(7325.5) == "2h 2m 5.50s"
        
        @info "Format time tests passed"
    end
    
    @testset "Format Bytes - Various Sizes" begin
        @info "Testing format_bytes"
        
        # Bytes
        @test format_bytes(512) == "512.00 B"
        @test format_bytes(1000) == "1000.00 B"
        
        # Kilobytes
        @test format_bytes(1024) == "1.00 KB"
        @test format_bytes(2048) == "2.00 KB"
        
        # Megabytes
        @test format_bytes(1024 * 1024) == "1.00 MB"
        @test format_bytes(5 * 1024 * 1024) == "5.00 MB"
        
        # Gigabytes
        @test format_bytes(1024 * 1024 * 1024) == "1.00 GB"
        @test format_bytes(3 * 1024 * 1024 * 1024) == "3.00 GB"
        
        @info "Format bytes tests passed"
    end
    
    @testset "Summary Statistics - Basic" begin
        @info "Testing compute_summary_statistics"
        
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_summary_statistics(data)
        
        @test stats["mean"] == 3.0
        @test stats["median"] == 3.0
        @test stats["min"] == 1.0
        @test stats["max"] == 5.0
        @test stats["n"] == 5
        @test haskey(stats, "std")
        @test haskey(stats, "var")
        @test haskey(stats, "q25")
        @test haskey(stats, "q75")
        
        @info "Summary statistics basic tests passed"
    end
    
    @testset "Summary Statistics - Edge Cases" begin
        @info "Testing summary statistics edge cases"
        
        # Single value
        data_single = [5.0]
        stats_single = compute_summary_statistics(data_single)
        @test stats_single["mean"] == 5.0
        @test stats_single["median"] == 5.0
        @test stats_single["min"] == 5.0
        @test stats_single["max"] == 5.0
        @test stats_single["n"] == 1
        
        # All same values
        data_same = [7.0, 7.0, 7.0, 7.0]
        stats_same = compute_summary_statistics(data_same)
        @test stats_same["mean"] == 7.0
        @test stats_same["std"] == 0.0
        @test stats_same["var"] == 0.0
        
        # Large dataset
        data_large = collect(1.0:1000.0)
        stats_large = compute_summary_statistics(data_large)
        @test stats_large["mean"] == 500.5
        @test stats_large["n"] == 1000
        
        @info "Summary statistics edge cases tests passed"
    end
    
    @testset "Bernoulli Confidence Interval - Basic" begin
        @info "Testing bernoulli_confidence_interval"
        
        # 70 successes out of 100 trials
        ci = bernoulli_confidence_interval(70, 100, confidence=0.95)
        
        @test haskey(ci, :lower)
        @test haskey(ci, :upper)
        @test haskey(ci, :estimate)
        
        @test 0 <= ci.lower < ci.upper <= 1
        @test isapprox(ci.estimate, 0.7, atol=1e-6)
        @test ci.lower < 0.7 < ci.upper
        
        @info "Bernoulli confidence interval basic tests passed"
    end
    
    @testset "Bernoulli Confidence Interval - Various Confidence Levels" begin
        @info "Testing confidence intervals with various levels"
        
        n_successes = 60
        n_trials = 100
        
        # 90% CI
        ci_90 = bernoulli_confidence_interval(n_successes, n_trials, confidence=0.90)
        
        # 95% CI
        ci_95 = bernoulli_confidence_interval(n_successes, n_trials, confidence=0.95)
        
        # 99% CI
        ci_99 = bernoulli_confidence_interval(n_successes, n_trials, confidence=0.99)
        
        # Higher confidence should give wider intervals
        width_90 = ci_90.upper - ci_90.lower
        width_95 = ci_95.upper - ci_95.lower
        width_99 = ci_99.upper - ci_99.lower
        
        @test width_90 < width_95 < width_99
        
        # All should contain the estimate
        @test ci_90.lower < ci_90.estimate < ci_90.upper
        @test ci_95.lower < ci_95.estimate < ci_95.upper
        @test ci_99.lower < ci_99.estimate < ci_99.upper
        
        @info "Various confidence level tests passed"
    end
    
    @testset "Bernoulli Confidence Interval - Edge Cases" begin
        @info "Testing confidence interval edge cases"
        
        # All successes
        ci_all = bernoulli_confidence_interval(100, 100, confidence=0.95)
        @test ci_all.estimate == 1.0
        @test ci_all.lower < 1.0
        @test ci_all.upper == 1.0
        
        # No successes
        ci_none = bernoulli_confidence_interval(0, 100, confidence=0.95)
        @test ci_none.estimate == 0.0
        @test ci_none.lower == 0.0
        @test ci_none.upper > 0.0
        
        # Small sample size
        ci_small = bernoulli_confidence_interval(5, 10, confidence=0.95)
        @test 0 <= ci_small.lower < ci_small.upper <= 1
        
        # Large sample size (should be narrower)
        ci_large = bernoulli_confidence_interval(500, 1000, confidence=0.95)
        width_small = ci_small.upper - ci_small.lower
        width_large = ci_large.upper - ci_large.lower
        @test width_large < width_small
        
        @info "Confidence interval edge cases tests passed"
    end
    
    @testset "Utility Integration" begin
        @info "Testing utility functions integration"
        
        temp_dir = mktempdir()
        original_dir = pwd()
        cd(temp_dir)
        
        # Create experiment
        timer = CoinTossUtils.Timer("full_experiment")
        
        # Generate some results
        results = Dict(
            "parameters" => Dict(
                "n" => 100,
                "theta" => 0.7
            ),
            "statistics" => compute_summary_statistics([0.65, 0.68, 0.72, 0.69, 0.71]),
            "confidence_interval" => bernoulli_confidence_interval(70, 100)
        )
        
        # Save results
        mkpath("outputs/results")
        results_dir = save_experiment_results("integration_test", results)
        
        elapsed = close(timer)
        
        @test isdir(results_dir)
        @test elapsed > 0
        
        # Cleanup
        cd(original_dir)
        rm(temp_dir, recursive=true)
        
        @info "Utility integration tests passed"
    end
    
end

@info "All CoinTossUtils tests completed successfully"

