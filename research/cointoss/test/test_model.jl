#!/usr/bin/env julia
# Comprehensive tests for CoinTossModel module

using Test
using Distributions
using Random
using Logging

# Setup test logging
test_logger = ConsoleLogger(stderr, Logging.Info)
global_logger(test_logger)

@info "Starting CoinTossModel tests"

@testset "CoinTossModel Module Tests" begin
    
    @testset "CoinData Structure" begin
        @info "Testing CoinData structure"
        
        data = generate_coin_data(n=100, theta_real=0.6, seed=42)
        
        @test data isa CoinData
        @test data.observations isa Vector{Float64}
        @test data.theta_real isa Float64
        @test data.n_samples isa Int
        @test data.seed isa Int
        @test data.timestamp isa String
        
        @test length(data.observations) == data.n_samples
        @test data.n_samples == 100
        @test data.theta_real == 0.6
        @test data.seed == 42
        
        @info "CoinData structure tests passed"
    end
    
    @testset "Data Generation - Basic" begin
        @info "Testing basic data generation"
        
        # Standard case
        data = generate_coin_data(n=100, theta_real=0.75, seed=42)
        @test length(data.observations) == 100
        @test all(x -> x in [0.0, 1.0], data.observations)
        
        # Different parameters
        data2 = generate_coin_data(n=500, theta_real=0.3, seed=123)
        @test length(data2.observations) == 500
        @test data2.theta_real == 0.3
        
        @info "Basic data generation tests passed"
    end
    
    @testset "Data Generation - Edge Cases" begin
        @info "Testing edge case data generation"
        
        # All heads (θ = 1.0)
        data_heads = generate_coin_data(n=100, theta_real=1.0, seed=42)
        @test all(data_heads.observations .== 1.0)
        @test sum(data_heads.observations) == 100
        
        # All tails (θ = 0.0)
        data_tails = generate_coin_data(n=100, theta_real=0.0, seed=42)
        @test all(data_tails.observations .== 0.0)
        @test sum(data_tails.observations) == 0
        
        # Fair coin
        data_fair = generate_coin_data(n=10000, theta_real=0.5, seed=999)
        empirical_mean = mean(data_fair.observations)
        @test abs(empirical_mean - 0.5) < 0.02  # Should be very close to 0.5
        
        # Minimal sample size
        data_min = generate_coin_data(n=1, theta_real=0.5, seed=1)
        @test length(data_min.observations) == 1
        
        # Large sample size
        data_large = generate_coin_data(n=100000, theta_real=0.5, seed=2)
        @test length(data_large.observations) == 100000
        
        @info "Edge case data generation tests passed"
    end
    
    @testset "Data Generation - Reproducibility" begin
        @info "Testing data generation reproducibility"
        
        # Same seed should produce identical results
        data1 = generate_coin_data(n=100, theta_real=0.6, seed=789)
        data2 = generate_coin_data(n=100, theta_real=0.6, seed=789)
        @test data1.observations == data2.observations
        
        # Different seed should produce different results (with high probability)
        data3 = generate_coin_data(n=100, theta_real=0.6, seed=790)
        @test data1.observations != data3.observations
        
        @info "Reproducibility tests passed"
    end
    
    @testset "Data Generation - Invalid Inputs" begin
        @info "Testing invalid input handling"
        
        # Invalid theta values
        @test_throws AssertionError generate_coin_data(n=100, theta_real=-0.1)
        @test_throws AssertionError generate_coin_data(n=100, theta_real=1.5)
        @test_throws AssertionError generate_coin_data(n=100, theta_real=2.0)
        
        # Invalid n values
        @test_throws AssertionError generate_coin_data(n=0, theta_real=0.5)
        @test_throws AssertionError generate_coin_data(n=-10, theta_real=0.5)
        
        # Boundary values (should work)
        @test_nowarn generate_coin_data(n=1, theta_real=0.0)
        @test_nowarn generate_coin_data(n=1, theta_real=1.0)
        
        @info "Invalid input handling tests passed"
    end
    
    @testset "RxInfer Model Definition" begin
        @info "Testing RxInfer model definition"
        
        # Test that model can be created
        data = ones(10)
        @test_nowarn coin_model(a=2.0, b=2.0)
        
        # Model should accept different prior parameters
        @test_nowarn coin_model(a=1.0, b=1.0)  # Uniform prior
        @test_nowarn coin_model(a=0.5, b=0.5)  # Jeffreys prior
        @test_nowarn coin_model(a=10.0, b=5.0)  # Informative prior
        
        @info "RxInfer model definition tests passed"
    end
    
    @testset "Analytical Posterior - Basic" begin
        @info "Testing analytical posterior computation"
        
        # Simple case: uniform prior, balanced data
        data = [1.0, 1.0, 0.0, 1.0]  # 3 heads, 1 tail
        prior_a = 1.0
        prior_b = 1.0
        
        posterior = analytical_posterior(data, prior_a, prior_b)
        
        @test posterior isa Beta
        α, β = params(posterior)
        @test α == prior_a + 3.0  # 1 + 3 heads
        @test β == prior_b + 1.0  # 1 + 1 tail
        
        @info "Basic analytical posterior tests passed"
    end
    
    @testset "Analytical Posterior - Various Priors" begin
        @info "Testing analytical posterior with various priors"
        
        data = ones(10)  # All heads
        
        # Uniform prior
        post1 = analytical_posterior(data, 1.0, 1.0)
        @test params(post1) == (11.0, 1.0)
        
        # Informative prior
        post2 = analytical_posterior(data, 5.0, 5.0)
        @test params(post2) == (15.0, 5.0)
        
        # Jeffreys prior
        post3 = analytical_posterior(data, 0.5, 0.5)
        @test params(post3) == (10.5, 0.5)
        
        @info "Various priors tests passed"
    end
    
    @testset "Posterior Statistics - Complete" begin
        @info "Testing comprehensive posterior statistics"
        
        posterior = Beta(10, 5)
        stats = posterior_statistics(posterior, credible_level=0.95)
        
        # Check all required keys
        @test haskey(stats, "mean")
        @test haskey(stats, "mode")
        @test haskey(stats, "variance")
        @test haskey(stats, "std")
        @test haskey(stats, "credible_interval")
        @test haskey(stats, "credible_level")
        @test haskey(stats, "alpha")
        @test haskey(stats, "beta")
        
        # Verify calculations
        @test isapprox(stats["mean"], 10/15, atol=1e-10)
        @test isapprox(stats["mode"], 9/13, atol=1e-10)
        @test stats["alpha"] == 10.0
        @test stats["beta"] == 5.0
        @test stats["credible_level"] == 0.95
        
        # Credible interval bounds
        ci = stats["credible_interval"]
        @test 0 <= ci[1] < ci[2] <= 1
        @test ci[1] < stats["mean"] < ci[2]
        
        @info "Comprehensive posterior statistics tests passed"
    end
    
    @testset "Posterior Statistics - Edge Cases" begin
        @info "Testing posterior statistics edge cases"
        
        # Degenerate case: α = 1, β = 1 (uniform)
        stats1 = posterior_statistics(Beta(1, 1))
        @test stats1["mean"] == 0.5
        @test isnan(stats1["mode"])  # Mode undefined for α, β ≤ 1
        
        # Highly peaked distribution
        stats2 = posterior_statistics(Beta(100, 100))
        @test isapprox(stats2["mean"], 0.5, atol=1e-6)
        @test isapprox(stats2["mode"], 0.5, atol=1e-6)
        @test stats2["variance"] < 0.01  # Low variance
        
        # Skewed distribution
        stats3 = posterior_statistics(Beta(2, 10))
        @test stats3["mean"] < 0.5  # Skewed left
        
        # Different credible levels
        stats_90 = posterior_statistics(Beta(10, 5), credible_level=0.90)
        stats_99 = posterior_statistics(Beta(10, 5), credible_level=0.99)
        
        ci_90 = stats_90["credible_interval"]
        ci_99 = stats_99["credible_interval"]
        
        # 99% CI should be wider than 90% CI
        @test (ci_99[2] - ci_99[1]) > (ci_90[2] - ci_90[1])
        
        @info "Edge case posterior statistics tests passed"
    end
    
    @testset "Log Marginal Likelihood - Basic" begin
        @info "Testing log marginal likelihood computation"
        
        data = ones(10)  # All heads
        prior_a = 1.0
        prior_b = 1.0
        
        log_ml = log_marginal_likelihood(data, prior_a, prior_b)
        
        @test isfinite(log_ml)
        @test log_ml < 0  # Log probability should be negative
        
        @info "Basic log marginal likelihood tests passed"
    end
    
    @testset "Log Marginal Likelihood - Properties" begin
        @info "Testing log marginal likelihood properties"
        
        data = [1.0, 1.0, 0.0, 1.0]  # 3 heads, 1 tail
        
        # Different priors should give different marginal likelihoods
        log_ml1 = log_marginal_likelihood(data, 1.0, 1.0)
        log_ml2 = log_marginal_likelihood(data, 5.0, 5.0)
        log_ml3 = log_marginal_likelihood(data, 10.0, 10.0)
        
        @test log_ml1 != log_ml2
        @test log_ml2 != log_ml3
        
        # More data should change marginal likelihood
        data_short = data[1:2]
        data_long = vcat(data, data)
        
        log_ml_short = log_marginal_likelihood(data_short, 1.0, 1.0)
        log_ml_long = log_marginal_likelihood(data_long, 1.0, 1.0)
        
        @test log_ml_short != log_ml_long
        
        @info "Log marginal likelihood properties tests passed"
    end
    
    @testset "Log Marginal Likelihood - Edge Cases" begin
        @info "Testing log marginal likelihood edge cases"
        
        # Empty data (edge case)
        data_empty = Float64[]
        log_ml_empty = log_marginal_likelihood(data_empty, 1.0, 1.0)
        @test isfinite(log_ml_empty)
        @test log_ml_empty == 0.0  # Should be log(1) = 0
        
        # All heads
        data_heads = ones(100)
        log_ml_heads = log_marginal_likelihood(data_heads, 1.0, 1.0)
        @test isfinite(log_ml_heads)
        
        # All tails
        data_tails = zeros(100)
        log_ml_tails = log_marginal_likelihood(data_tails, 1.0, 1.0)
        @test isfinite(log_ml_tails)
        
        @info "Log marginal likelihood edge cases tests passed"
    end
    
    @testset "Conjugate Property Verification" begin
        @info "Testing Beta-Bernoulli conjugacy"
        
        # Prior is Beta, posterior should also be Beta
        data = generate_coin_data(n=50, theta_real=0.7, seed=42).observations
        prior_a = 2.0
        prior_b = 3.0
        
        posterior = analytical_posterior(data, prior_a, prior_b)
        @test posterior isa Beta
        
        # Verify conjugate update formula
        n_heads = sum(data)
        n_tails = length(data) - n_heads
        
        α_post, β_post = params(posterior)
        @test α_post == prior_a + n_heads
        @test β_post == prior_b + n_tails
        
        # Posterior should be more concentrated than prior
        @test var(posterior) <= var(Beta(prior_a, prior_b))
        
        @info "Conjugacy verification tests passed"
    end
    
    @testset "Statistical Consistency" begin
        @info "Testing statistical consistency"
        
        # Generate data with known theta
        theta_true = 0.65
        data = generate_coin_data(n=1000, theta_real=theta_true, seed=123).observations
        
        # Compute posterior with weak prior
        posterior = analytical_posterior(data, 1.0, 1.0)
        
        # Posterior mean should be close to true value with large sample
        @test abs(mean(posterior) - theta_true) < 0.05
        
        # Empirical rate should be close to true value
        empirical_rate = sum(data) / length(data)
        @test abs(empirical_rate - theta_true) < 0.05
        
        @info "Statistical consistency tests passed"
    end
    
end

@info "All CoinTossModel tests completed successfully"

