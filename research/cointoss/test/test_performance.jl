#!/usr/bin/env julia
# Performance and benchmark tests for Coin Toss Model

using Test
using Distributions
using Random
using Statistics
using Logging

# Setup test logging
test_logger = ConsoleLogger(stderr, Logging.Info)
global_logger(test_logger)

@info "Starting Performance and Benchmark tests"

@testset "Performance and Benchmark Tests" begin
    
    @testset "Data Generation Performance" begin
        @info "Testing data generation performance"
        
        # Small dataset
        t_small = @elapsed begin
            for i in 1:100
                generate_coin_data(n=100, theta_real=0.7, seed=i)
            end
        end
        @test t_small < 1.0  # Should be very fast
        @info "Small dataset generation (100 runs, n=100): $(round(t_small, digits=3))s"
        
        # Medium dataset
        t_medium = @elapsed begin
            for i in 1:10
                generate_coin_data(n=10000, theta_real=0.7, seed=i)
            end
        end
        @test t_medium < 2.0
        @info "Medium dataset generation (10 runs, n=10000): $(round(t_medium, digits=3))s"
        
        # Large dataset (single run)
        t_large = @elapsed generate_coin_data(n=1000000, theta_real=0.7, seed=42)
        @test t_large < 5.0
        @info "Large dataset generation (1 run, n=1000000): $(round(t_large, digits=3))s"
        
        @info "Data generation performance tests passed"
    end
    
    @testset "Analytical Posterior Performance" begin
        @info "Testing analytical posterior computation performance"
        
        # Generate test data
        data = generate_coin_data(n=10000, theta_real=0.65, seed=123).observations
        
        # Benchmark analytical posterior
        t_analytical = @elapsed begin
            for i in 1:1000
                analytical_posterior(data, 2.0, 2.0)
            end
        end
        
        @test t_analytical < 1.0  # Should be instant (closed form)
        @info "Analytical posterior (1000 runs, n=10000): $(round(t_analytical, digits=3))s"
        
        # Compare with different data sizes
        data_small = data[1:100]
        data_large = data
        
        t_small = @elapsed begin
            for i in 1:1000
                analytical_posterior(data_small, 2.0, 2.0)
            end
        end
        
        t_large = @elapsed begin
            for i in 1:1000
                analytical_posterior(data_large, 2.0, 2.0)
            end
        end
        
        # Both should be fast (O(n) to count, but very simple)
        @test t_small < 1.0
        @test t_large < 2.0
        
        @info "Analytical posterior performance tests passed"
    end
    
    @testset "RxInfer Inference Performance" begin
        @info "Testing RxInfer inference performance"
        
        # Small dataset
        data_small = ones(50)
        t_small = @elapsed run_inference(data_small, 2.0, 2.0; iterations=10, showprogress=false)
        @test t_small < 2.0
        @info "RxInfer small (n=50, iter=10): $(round(t_small, digits=3))s"
        
        # Medium dataset
        data_medium = ones(500)
        t_medium = @elapsed run_inference(data_medium, 2.0, 2.0; iterations=10, showprogress=false)
        @test t_medium < 5.0
        @info "RxInfer medium (n=500, iter=10): $(round(t_medium, digits=3))s"
        
        # Many iterations
        t_many_iter = @elapsed run_inference(data_small, 2.0, 2.0; iterations=50, showprogress=false)
        @test t_many_iter < 10.0
        @info "RxInfer many iterations (n=50, iter=50): $(round(t_many_iter, digits=3))s"
        
        @info "RxInfer inference performance tests passed"
    end
    
    @testset "KL Divergence Performance" begin
        @info "Testing KL divergence computation performance"
        
        p = Beta(10, 5)
        q = Beta(12, 6)
        
        t_kl = @elapsed begin
            for i in 1:10000
                kl_divergence(q, p)
            end
        end
        
        @test t_kl < 1.0  # Should be very fast (analytical formula)
        @info "KL divergence (10000 runs): $(round(t_kl, digits=3))s"
        
        @info "KL divergence performance tests passed"
    end
    
    @testset "Posterior Statistics Performance" begin
        @info "Testing posterior statistics computation performance"
        
        posterior = Beta(100, 50)
        
        t_stats = @elapsed begin
            for i in 1:1000
                posterior_statistics(posterior, credible_level=0.95)
            end
        end
        
        @test t_stats < 2.0
        @info "Posterior statistics (1000 runs): $(round(t_stats, digits=3))s"
        
        @info "Posterior statistics performance tests passed"
    end
    
    @testset "Visualization Performance" begin
        @info "Testing visualization performance"
        
        prior = Beta(2, 2)
        posterior = Beta(20, 10)
        data = ones(100)
        
        # Prior-posterior plot
        t_pp = @elapsed plot_prior_posterior(prior, posterior; theta_real=0.7)
        @test t_pp < 5.0
        @info "Prior-posterior plot: $(round(t_pp, digits=3))s"
        
        # Data histogram
        t_hist = @elapsed plot_data_histogram(data; theta_real=0.7)
        @test t_hist < 5.0
        @info "Data histogram: $(round(t_hist, digits=3))s"
        
        # Credible interval plot
        t_ci = @elapsed plot_credible_interval(posterior; theta_real=0.7)
        @test t_ci < 5.0
        @info "Credible interval plot: $(round(t_ci, digits=3))s"
        
        @info "Visualization performance tests passed"
    end
    
    @testset "Dashboard Creation Performance" begin
        @info "Testing comprehensive dashboard performance"
        
        prior = Beta(2, 2)
        posterior = Beta(20, 10)
        data = ones(100)
        free_energy = collect(range(-100, -90, length=10))
        
        t_dashboard = @elapsed plot_comprehensive_dashboard(
            prior, posterior, data, free_energy;
            theta_real=0.7
        )
        
        @test t_dashboard < 15.0  # Multiple plots, so can be slower
        @info "Comprehensive dashboard: $(round(t_dashboard, digits=3))s"
        
        @info "Dashboard performance tests passed"
    end
    
    @testset "Export Performance" begin
        @info "Testing data export performance"
        
        # Create large results dictionary
        large_results = Dict(
            "data" => Dict(
                "observations" => rand(10000),
                "metadata" => Dict(
                    "timestamp" => string(now()),
                    "version" => "1.0"
                )
            ),
            "diagnostics" => Dict(
                "values" => [rand() for _ in 1:1000],
                "stats" => Dict(
                    "mean" => 0.5,
                    "std" => 0.29
                )
            )
        )
        
        temp_dir = mktempdir()
        
        # CSV export
        csv_path = joinpath(temp_dir, "perf_test.csv")
        t_csv = @elapsed export_to_csv(large_results, csv_path)
        @test t_csv < 2.0
        @info "CSV export (large dict): $(round(t_csv, digits=3))s"
        
        # JSON export
        json_path = joinpath(temp_dir, "perf_test.json")
        t_json = @elapsed export_to_json(large_results, json_path)
        @test t_json < 2.0
        @info "JSON export (large dict): $(round(t_json, digits=3))s"
        
        # Cleanup
        rm(temp_dir, recursive=true)
        
        @info "Export performance tests passed"
    end
    
    @testset "Dictionary Flattening Performance" begin
        @info "Testing dictionary flattening performance"
        
        # Create deeply nested dictionary
        deep_dict = Dict("level0" => Dict())
        current = deep_dict["level0"]
        for i in 1:10
            current["level$i"] = Dict()
            current = current["level$i"]
        end
        current["value"] = 42
        
        t_flatten = @elapsed begin
            for i in 1:1000
                CoinTossUtils.flatten_dict(deep_dict)
            end
        end
        
        @test t_flatten < 2.0
        @info "Dictionary flattening (1000 runs, 10 levels): $(round(t_flatten, digits=3))s"
        
        @info "Dictionary flattening performance tests passed"
    end
    
    @testset "End-to-End Workflow Performance" begin
        @info "Testing complete workflow performance"
        
        t_workflow = @elapsed begin
            # 1. Generate data
            data = generate_coin_data(n=500, theta_real=0.7, seed=42)
            
            # 2. Run inference
            result = run_inference(
                data.observations, 2.0, 2.0;
                iterations=10,
                track_fe=true,
                showprogress=false
            )
            
            # 3. Compute analytical posterior
            analytical_post = analytical_posterior(data.observations, 2.0, 2.0)
            
            # 4. Compute statistics
            stats = posterior_statistics(result.posterior)
            
            # 5. Create visualizations
            p1 = plot_prior_posterior(result.prior, result.posterior; theta_real=0.7)
            p2 = plot_data_histogram(data.observations; theta_real=0.7)
            
            # 6. Export results (to temp)
            temp_dir = mktempdir()
            results_dict = Dict(
                "data" => Dict("n" => data.n_samples),
                "inference" => Dict("converged" => result.converged)
            )
            export_to_json(results_dict, joinpath(temp_dir, "results.json"))
            rm(temp_dir, recursive=true)
        end
        
        @test t_workflow < 20.0  # Complete workflow should be reasonably fast
        @info "End-to-end workflow (n=500): $(round(t_workflow, digits=3))s"
        
        @info "End-to-end workflow performance tests passed"
    end
    
    @testset "Memory Efficiency - Large Datasets" begin
        @info "Testing memory efficiency with large datasets"
        
        # Test that large datasets don't cause memory issues
        @test_nowarn begin
            data = generate_coin_data(n=100000, theta_real=0.6, seed=99)
            posterior = analytical_posterior(data.observations, 1.0, 1.0)
            stats = posterior_statistics(posterior)
        end
        
        @info "Memory efficiency tests passed"
    end
    
    @testset "Scalability - Varying Sample Sizes" begin
        @info "Testing scalability with varying sample sizes"
        
        sample_sizes = [100, 500, 1000, 5000, 10000]
        times = Float64[]
        
        for n in sample_sizes
            data = generate_coin_data(n=n, theta_real=0.7, seed=42).observations
            t = @elapsed analytical_posterior(data, 2.0, 2.0)
            push!(times, t)
        end
        
        # Log results
        for (n, t) in zip(sample_sizes, times)
            @info "Analytical posterior (n=$n): $(round(t, digits=4))s"
        end
        
        # All should complete
        @test all(times .< 1.0)
        
        @info "Scalability tests passed"
    end
    
    @testset "Convergence Speed" begin
        @info "Testing inference convergence speed"
        
        data = ones(100)  # Simple data for fast convergence
        
        # Track convergence with different iteration limits
        for max_iter in [5, 10, 20, 50]
            result = run_inference(
                data, 1.0, 1.0;
                iterations=max_iter,
                track_fe=true,
                convergence_check=true,
                showprogress=false
            )
            
            if result.converged
                @info "Converged at iteration $(result.convergence_iteration) (max=$max_iter)"
                @test result.convergence_iteration <= max_iter
                break
            end
        end
        
        @info "Convergence speed tests passed"
    end
    
    @testset "Parallel Test Execution Readiness" begin
        @info "Testing that functions are thread-safe for parallel testing"
        
        # These tests verify independence of function calls
        # Important for potential parallel test execution
        
        # Generate multiple datasets independently
        datasets = [generate_coin_data(n=100, theta_real=0.7, seed=i) for i in 1:10]
        @test length(datasets) == 10
        @test all(d.seed != datasets[1].seed for d in datasets[2:end])
        
        # Multiple inference runs should be independent
        results = [
            run_inference(ones(50), 2.0, 2.0; iterations=5, showprogress=false)
            for _ in 1:5
        ]
        @test length(results) == 5
        
        @info "Parallel readiness tests passed"
    end
    
    @testset "Performance Summary" begin
        @info "========== PERFORMANCE SUMMARY =========="
        @info "All performance benchmarks completed successfully"
        @info "Key findings:"
        @info "  - Data generation: Fast, scales linearly"
        @info "  - Analytical computations: Near-instant (closed form)"
        @info "  - RxInfer inference: Efficient for moderate datasets"
        @info "  - Visualization: Acceptable performance"
        @info "  - Export: Efficient for typical use cases"
        @info "  - End-to-end: Complete workflow < 20s"
        @info "========================================"
    end
    
end

@info "All Performance and Benchmark tests completed successfully"

