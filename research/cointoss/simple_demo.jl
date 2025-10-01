#!/usr/bin/env julia

# Simple demonstration script for Coin Toss Model
# Quick test without full experiment workflow

using Pkg
Pkg.activate(".")

println("="^80)
println("Coin Toss Model - Simple Demo")
println("="^80)

# Load modules
include("src/model.jl")
include("src/inference.jl")

using .CoinTossModel
using .CoinTossInference
using Distributions
using Statistics

# Generate synthetic data
println("\n1. Generating synthetic data...")
data = generate_coin_data(n=100, theta_real=0.75, seed=42)
println("   Generated $(length(data.observations)) coin tosses")
println("   True θ: $(data.theta_real)")
println("   Observed heads: $(sum(data.observations)) ($(round(mean(data.observations)*100, digits=1))%)")

# Run simple inference
println("\n2. Running Bayesian inference...")
result = run_inference(
    data.observations,
    4.0,  # prior_a
    8.0;  # prior_b
    iterations=5,
    track_fe=false,
    showprogress=false
)

println("   Inference completed in $(round(result.execution_time, digits=4))s")

# Display results
println("\n3. Posterior Statistics:")
stats = posterior_statistics(result.posterior, credible_level=0.95)
println("   Mean: $(round(stats["mean"], digits=4))")
println("   Mode: $(round(stats["mode"], digits=4))")
println("   Std: $(round(stats["std"], digits=4))")
println("   95% CI: [$(round(stats["credible_interval"][1], digits=4)), $(round(stats["credible_interval"][2], digits=4))]")

# Check if true value is in credible interval
in_ci = stats["credible_interval"][1] <= data.theta_real <= stats["credible_interval"][2]
println("   True θ in CI: $in_ci ✓" * (in_ci ? "" : " (outside CI)"))

# Compare with analytical solution
println("\n4. Analytical Validation:")
analytical_post = analytical_posterior(data.observations, 4.0, 8.0)
a_analytical, b_analytical = params(analytical_post)
a_rxinfer, b_rxinfer = params(result.posterior)
println("   Analytical: α=$a_analytical, β=$b_analytical")
println("   RxInfer: α=$a_rxinfer, β=$b_rxinfer")
match = isapprox(a_analytical, a_rxinfer) && isapprox(b_analytical, b_rxinfer)
println("   Match: $match")

# Diagnostics
println("\n5. Key Diagnostics:")
println("   Empirical rate: $(round(result.diagnostics["empirical_rate"], digits=4))")
println("   KL divergence: $(round(result.diagnostics["kl_divergence"], digits=4))")
println("   Information gain: $(round(result.diagnostics["information_gain"], digits=4))")

println("\n" * "="^80)
println("Demo completed successfully!")
println("="^80)
println("\nFor full experiment with visualizations and exports, run:")
println("  julia run.jl")

