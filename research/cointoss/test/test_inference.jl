#!/usr/bin/env julia
# Comprehensive tests for CoinTossInference module

using Test
using Distributions
using Random
using Statistics
using Logging

# Setup test logging
test_logger = ConsoleLogger(stderr, Logging.Info)
global_logger(test_logger)

@info "Starting CoinTossInference tests"

@testset "CoinTossInference Module Tests" begin
    
    @testset "InferenceResult Structure" begin
        @info "Testing InferenceResult structure"
        
        data = generate_coin_data(n=100, theta_real=0.7, seed=42).observations
        result = run_inference(data, 2.0, 2.0; iterations=10, showprogress=false)
        
        @test result isa InferenceResult
        @test result.posterior isa Beta
        @test result.prior isa Beta
        @test result.observations isa Vector{Float64}
        @test result.execution_time isa Float64
        @test result.iterations isa Int
        @test result.converged isa Bool
        @test result.diagnostics isa Dict
        
        @info "InferenceResult structure tests passed"
    end
    
    @testset "Basic Inference Execution" begin
        @info "Testing basic inference execution"
        
        data = generate_coin_data(n=100, theta_real=0.7, seed=42).observations
        
        result = run_inference(
            data, 2.0, 2.0;
            iterations=10,
            track_fe=true,
            showprogress=false
        )
        
        @test result.posterior isa Beta
        @test result.execution_time > 0
        @test result.iterations == 10
        @test result.free_energy !== nothing
        @test length(result.free_energy) > 0
        
        @info "Basic inference execution tests passed"
    end
    
    @testset "Inference with Different Iterations" begin
        @info "Testing inference with varying iteration counts"
        
        data = ones(50)
        
        # Few iterations
        result_5 = run_inference(data, 1.0, 1.0; iterations=5, showprogress=false)
        @test result_5.iterations == 5
        
        # Many iterations
        result_50 = run_inference(data, 1.0, 1.0; iterations=50, showprogress=false)
        @test result_50.iterations == 50
        
        # Single iteration
        result_1 = run_inference(data, 1.0, 1.0; iterations=1, showprogress=false)
        @test result_1.iterations == 1
        
        @info "Variable iteration tests passed"
    end
    
    @testset "Convergence Detection" begin
        @info "Testing convergence detection"
        
        # Simple data for fast convergence
        data = ones(50)
        
        result = run_inference(
            data, 1.0, 1.0;
            iterations=20,
            track_fe=true,
            convergence_check=true,
            convergence_tol=1e-6,
            showprogress=false
        )
        
        @test result.free_energy !== nothing
        @test length(result.free_energy) > 0
        
        # For conjugate model with sufficient data, should converge
        if result.converged
            @test result.convergence_iteration !== nothing
            @test result.convergence_iteration <= result.iterations
        end
        
        @info "Convergence detection tests passed"
    end
    
    @testset "Convergence Diagnostics Function" begin
        @info "Testing compute_convergence_diagnostics"
        
        # Create mock free energy trace
        fe_trace = [-100.0, -95.0, -92.0, -91.0, -90.5, -90.3, -90.2, -90.15, -90.1, -90.08]
        
        diagnostics = compute_convergence_diagnostics(fe_trace)
        
        @test haskey(diagnostics, "n_iterations")
        @test haskey(diagnostics, "initial_fe")
        @test haskey(diagnostics, "final_fe")
        @test haskey(diagnostics, "total_reduction")
        @test haskey(diagnostics, "mean_change")
        @test haskey(diagnostics, "max_change")
        @test haskey(diagnostics, "final_change")
        @test haskey(diagnostics, "monotonic_decrease")
        @test haskey(diagnostics, "fe_trace")
        
        @test diagnostics["n_iterations"] == 10
        @test diagnostics["initial_fe"] == -100.0
        @test diagnostics["final_fe"] == -90.08
        @test diagnostics["total_reduction"] ≈ 9.92
        @test diagnostics["monotonic_decrease"] == true  # Should decrease monotonically
        
        @info "Convergence diagnostics function tests passed"
    end
    
    @testset "Convergence Diagnostics - Edge Cases" begin
        @info "Testing convergence diagnostics edge cases"
        
        # Empty trace
        empty_diagnostics = compute_convergence_diagnostics(Float64[])
        @test haskey(empty_diagnostics, "error")
        
        # Single element
        single_diagnostics = compute_convergence_diagnostics([-100.0])
        @test haskey(single_diagnostics, "error")
        
        # Non-monotonic decrease
        non_monotonic = [-100.0, -95.0, -96.0, -94.0]
        nm_diagnostics = compute_convergence_diagnostics(non_monotonic)
        @test nm_diagnostics["monotonic_decrease"] == false
        
        @info "Convergence diagnostics edge cases tests passed"
    end
    
    @testset "KL Divergence Computation" begin
        @info "Testing KL divergence computation"
        
        # Test KL divergence calculation
        p = Beta(2, 2)
        q = Beta(3, 3)
        
        kl = kl_divergence(q, p)
        @test isfinite(kl)
        @test kl >= 0  # KL divergence is always non-negative
        
        # KL(p||p) should be 0
        kl_self = kl_divergence(p, p)
        @test isapprox(kl_self, 0.0, atol=1e-10)
        
        # KL is asymmetric: KL(q||p) ≠ KL(p||q)
        kl_qp = kl_divergence(q, p)
        kl_pq = kl_divergence(p, q)
        @test kl_qp != kl_pq
        
        @info "KL divergence computation tests passed"
    end
    
    @testset "KL Divergence - Various Distributions" begin
        @info "Testing KL divergence with various distributions"
        
        # Very different distributions
        p1 = Beta(1, 10)  # Concentrated near 0
        q1 = Beta(10, 1)  # Concentrated near 1
        kl1 = kl_divergence(q1, p1)
        @test kl1 > 1.0  # Should be large
        
        # Similar distributions
        p2 = Beta(5, 5)
        q2 = Beta(5.5, 5.5)
        kl2 = kl_divergence(q2, p2)
        @test kl2 < 0.1  # Should be small
        
        # Uniform vs informative
        p_uniform = Beta(1, 1)
        q_informative = Beta(100, 100)
        kl3 = kl_divergence(q_informative, p_uniform)
        @test isfinite(kl3)
        @test kl3 > 0
        
        @info "KL divergence various distributions tests passed"
    end
    
    @testset "Expected Log Likelihood" begin
        @info "Testing expected log likelihood computation"
        
        posterior = Beta(10, 5)
        data = ones(10)  # All heads
        
        ell = expected_log_likelihood(posterior, data)
        
        @test isfinite(ell)
        @test ell < 0  # Log likelihood should be negative
        
        # More heads should give higher ELL for high-θ posterior
        data_heads = ones(20)
        data_tails = zeros(20)
        
        ell_heads = expected_log_likelihood(posterior, data_heads)
        ell_tails = expected_log_likelihood(posterior, data_tails)
        
        # Since posterior is biased toward heads (α > β), ELL for heads should be higher
        @test ell_heads > ell_tails
        
        @info "Expected log likelihood tests passed"
    end
    
    @testset "Expected Log Likelihood - Edge Cases" begin
        @info "Testing expected log likelihood edge cases"
        
        # Empty data
        posterior = Beta(5, 5)
        data_empty = Float64[]
        ell_empty = expected_log_likelihood(posterior, data_empty)
        @test ell_empty == 0.0
        
        # Extreme posteriors
        posterior_extreme = Beta(100, 1)  # Very confident in heads
        data = ones(10)
        ell_extreme = expected_log_likelihood(posterior_extreme, data)
        @test isfinite(ell_extreme)
        
        @info "Expected log likelihood edge cases tests passed"
    end
    
    @testset "Inference Diagnostics Computation" begin
        @info "Testing compute_inference_diagnostics"
        
        posterior = Beta(10, 5)
        prior = Beta(2, 2)
        data = ones(8) # 8 heads, 0 tails
        free_energy = [-100.0, -95.0, -92.0, -91.0, -90.5]
        
        diagnostics = compute_inference_diagnostics(posterior, prior, data, free_energy)
        
        # Check all expected keys
        required_keys = [
            "n_observations", "n_heads", "n_tails", "empirical_rate",
            "kl_divergence", "information_gain", "expected_log_likelihood",
            "posterior_mean", "posterior_variance", "posterior_mode",
            "prior_mean", "prior_variance", "mean_shift", "variance_reduction",
            "final_free_energy", "initial_free_energy", "free_energy_reduction",
            "mean_fe_change", "max_fe_change", "final_fe_change"
        ]
        
        for key in required_keys
            @test haskey(diagnostics, key)
        end
        
        # Verify some calculations
        @test diagnostics["n_observations"] == 8
        @test diagnostics["n_heads"] == 8
        @test diagnostics["n_tails"] == 0
        @test diagnostics["empirical_rate"] == 1.0
        
        @info "Inference diagnostics computation tests passed"
    end
    
    @testset "Posterior Predictive Check - Basic" begin
        @info "Testing posterior predictive check"
        
        posterior = Beta(10, 5)
        pp = posterior_predictive_check(posterior, 1000, seed=123)
        
        @test haskey(pp, "theta_samples")
        @test haskey(pp, "predictive_samples")
        @test haskey(pp, "predictive_mean")
        @test haskey(pp, "predictive_variance")
        @test haskey(pp, "pp_prob_heads")
        @test haskey(pp, "n_samples")
        
        @test length(pp["theta_samples"]) == 1000
        @test length(pp["predictive_samples"]) == 1000
        @test pp["n_samples"] == 1000
        
        # PP probability should be close to posterior mean
        @test isapprox(pp["pp_prob_heads"], mean(posterior), atol=0.05)
        
        @info "Posterior predictive check basic tests passed"
    end
    
    @testset "Posterior Predictive Check - Reproducibility" begin
        @info "Testing posterior predictive reproducibility"
        
        posterior = Beta(10, 5)
        
        # Same seed should give same results
        pp1 = posterior_predictive_check(posterior, 1000, seed=456)
        pp2 = posterior_predictive_check(posterior, 1000, seed=456)
        
        @test pp1["theta_samples"] == pp2["theta_samples"]
        @test pp1["predictive_samples"] == pp2["predictive_samples"]
        
        # Different seed should give different results
        pp3 = posterior_predictive_check(posterior, 1000, seed=789)
        @test pp1["theta_samples"] != pp3["theta_samples"]
        
        @info "Posterior predictive reproducibility tests passed"
    end
    
    @testset "Posterior Predictive Check - Edge Cases" begin
        @info "Testing posterior predictive edge cases"
        
        # Extreme posteriors
        posterior_heads = Beta(100, 1)  # Strongly biased toward heads
        pp_heads = posterior_predictive_check(posterior_heads, 1000, seed=1)
        @test pp_heads["pp_prob_heads"] > 0.95
        
        posterior_tails = Beta(1, 100)  # Strongly biased toward tails
        pp_tails = posterior_predictive_check(posterior_tails, 1000, seed=2)
        @test pp_tails["pp_prob_heads"] < 0.05
        
        # Uniform posterior
        posterior_uniform = Beta(1, 1)
        pp_uniform = posterior_predictive_check(posterior_uniform, 10000, seed=3)
        @test isapprox(pp_uniform["pp_prob_heads"], 0.5, atol=0.05)
        
        @info "Posterior predictive edge cases tests passed"
    end
    
    @testset "Track Free Energy Function" begin
        @info "Testing track_free_energy function"
        
        using ..CoinTossModel: coin_model
        
        data = ones(20)
        model_fn() = coin_model(a=2.0, b=2.0)
        data_dict = (y=data,)
        
        fe_trace = track_free_energy(model_fn, data_dict, iterations=20)
        
        @test fe_trace isa Vector{Float64}
        @test length(fe_trace) > 0
        @test all(isfinite.(fe_trace))
        
        # Free energy should generally decrease or stabilize
        if length(fe_trace) > 1
            # Check that it doesn't increase dramatically
            fe_changes = diff(fe_trace)
            @test all(fe_changes .<= 1e-6)  # Allow small numerical increases
        end
        
        @info "Track free energy function tests passed"
    end
    
    @testset "Inference Agreement - Analytical vs Numerical" begin
        @info "Testing agreement between analytical and numerical inference"
        
        # For conjugate Beta-Bernoulli, analytical and RxInfer should agree
        data = generate_coin_data(n=100, theta_real=0.6, seed=42).observations
        prior_a = 2.0
        prior_b = 3.0
        
        # Analytical
        analytical_post = analytical_posterior(data, prior_a, prior_b)
        
        # Numerical (RxInfer)
        result = run_inference(data, prior_a, prior_b; iterations=10, showprogress=false)
        
        # Parameters should be very close
        α_analytical, β_analytical = params(analytical_post)
        α_numerical, β_numerical = params(result.posterior)
        
        @test isapprox(α_analytical, α_numerical, rtol=0.01)
        @test isapprox(β_analytical, β_numerical, rtol=0.01)
        
        # Means should be very close
        @test isapprox(mean(analytical_post), mean(result.posterior), rtol=0.01)
        
        @info "Analytical vs numerical agreement tests passed"
    end
    
    @testset "Inference with Various Priors" begin
        @info "Testing inference with various prior specifications"
        
        data = ones(20)  # All heads
        
        # Uniform prior
        result_uniform = run_inference(data, 1.0, 1.0; showprogress=false)
        @test mean(result_uniform.posterior) > 0.9  # Should be high
        
        # Informative prior (biased toward tails)
        result_informative = run_inference(data, 1.0, 10.0; showprogress=false)
        # Prior pulls toward tails, but data pushes toward heads
        @test 0.3 < mean(result_informative.posterior) < 0.8
        
        # Weak prior
        result_weak = run_inference(data, 0.5, 0.5; showprogress=false)
        @test mean(result_weak.posterior) > 0.9
        
        @info "Various priors inference tests passed"
    end
    
    @testset "Information Gain Analysis" begin
        @info "Testing information gain from data"
        
        data = generate_coin_data(n=100, theta_real=0.7, seed=999).observations
        
        # Weak prior
        result = run_inference(data, 1.0, 1.0; showprogress=false)
        
        # Information gain should be positive
        info_gain = result.diagnostics["information_gain"]
        @test info_gain > 0
        
        # More data should give more information gain
        data_short = data[1:20]
        data_long = data
        
        result_short = run_inference(data_short, 1.0, 1.0; showprogress=false)
        result_long = run_inference(data_long, 1.0, 1.0; showprogress=false)
        
        @test result_long.diagnostics["information_gain"] > result_short.diagnostics["information_gain"]
        
        @info "Information gain analysis tests passed"
    end
    
    @testset "Variance Reduction Analysis" begin
        @info "Testing variance reduction from data"
        
        data = ones(50)  # Informative data
        prior_a = 2.0
        prior_b = 2.0
        
        result = run_inference(data, prior_a, prior_b; showprogress=false)
        
        # Variance should reduce
        variance_reduction = result.diagnostics["variance_reduction"]
        @test variance_reduction > 0
        
        # Posterior variance should be less than prior variance
        @test result.diagnostics["posterior_variance"] < result.diagnostics["prior_variance"]
        
        @info "Variance reduction analysis tests passed"
    end
    
end

@info "All CoinTossInference tests completed successfully"

