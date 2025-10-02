# Model module for Coin Toss Beta-Bernoulli inference
# Defines the probabilistic model and data generation

module CoinTossModel

using RxInfer
using Random
using Distributions
using SpecialFunctions: logbeta
using Dates: now

export coin_model, generate_coin_data, CoinData, posterior_statistics, analytical_posterior, log_marginal_likelihood

"""
    CoinData

Container for coin toss data and metadata.
"""
struct CoinData
    observations::Vector{Float64}
    theta_real::Float64
    n_samples::Int
    seed::Int
    timestamp::String
end

"""
    generate_coin_data(; n::Int=500, theta_real::Float64=0.75, seed::Int=42)

Generate synthetic coin toss data from a Bernoulli distribution.

# Arguments
- `n::Int`: Number of coin tosses
- `theta_real::Float64`: True probability of heads (must be in [0, 1])
- `seed::Int`: Random seed for reproducibility

# Returns
- `CoinData`: Struct containing observations and metadata
"""
function generate_coin_data(; n::Int=500, theta_real::Float64=0.75, seed::Int=42)
    @assert 0 <= theta_real <= 1 "theta_real must be in [0, 1]"
    @assert n > 0 "n must be positive"
    
    rng = MersenneTwister(seed)
    observations = float.(rand(rng, Bernoulli(theta_real), n))
    
    return CoinData(
        observations,
        theta_real,
        n,
        seed,
        string(now())
    )
end

# Define the Beta-Bernoulli coin toss model using RxInfer.
# Model Structure:
#   θ ~ Beta(a, b)           # Prior over coin bias
#   y[i] ~ Bernoulli(θ)      # Likelihood for each observation
# Arguments:
#   - y: Vector of observations (0s and 1s)
#   - a::Float64: Beta prior parameter (pseudo-count of heads)
#   - b::Float64: Beta prior parameter (pseudo-count of tails)
# Returns: Factor graph model suitable for RxInfer inference
@model function coin_model(y, a, b)
    # Prior over the coin bias parameter θ
    # Beta(a, b) encodes our prior belief about the fairness of the coin
    # Mean = a/(a+b), Mode = (a-1)/(a+b-2) for a,b > 1
    θ ~ Beta(a, b)
    
    # Likelihood: each coin toss is conditionally independent given θ
    # Using IID Bernoulli observations
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)
    end
end

"""
    posterior_statistics(posterior::Beta)

Compute comprehensive statistics for Beta posterior distribution.

# Returns
- Dict with mean, mode, variance, std, credible intervals
"""
function posterior_statistics(posterior::Beta; credible_level::Float64=0.95)
    α, β = params(posterior)
    
    # Analytical statistics for Beta distribution
    mean_val = mean(posterior)
    mode_val = α > 1 && β > 1 ? (α - 1) / (α + β - 2) : NaN
    var_val = var(posterior)
    std_val = sqrt(var_val)
    
    # Credible interval using quantiles
    lower_q = (1 - credible_level) / 2
    upper_q = 1 - lower_q
    ci_lower = quantile(posterior, lower_q)
    ci_upper = quantile(posterior, upper_q)
    
    return Dict(
        "mean" => mean_val,
        "mode" => mode_val,
        "variance" => var_val,
        "std" => std_val,
        "credible_interval" => (ci_lower, ci_upper),
        "credible_level" => credible_level,
        "alpha" => α,
        "beta" => β
    )
end

"""
    analytical_posterior(data::Vector{Float64}, prior_a::Float64, prior_b::Float64)

Compute analytical posterior for Beta-Bernoulli conjugate model.

# Arguments
- `data`: Vector of binary observations (0s and 1s)
- `prior_a`: Beta prior parameter a
- `prior_b`: Beta prior parameter b

# Returns
- `Beta`: Posterior distribution
"""
function analytical_posterior(data::Vector{Float64}, prior_a::Float64, prior_b::Float64)
    n_heads = sum(data)
    n_tails = length(data) - n_heads
    
    # Conjugate update: Beta(a + n_heads, b + n_tails)
    posterior_a = prior_a + n_heads
    posterior_b = prior_b + n_tails
    
    return Beta(posterior_a, posterior_b)
end

"""
    log_marginal_likelihood(data::Vector{Float64}, prior_a::Float64, prior_b::Float64)

Compute log marginal likelihood (model evidence) for Beta-Bernoulli model.

Uses the analytical formula:
log p(data) = log B(a + n_heads, b + n_tails) - log B(a, b)

where B is the Beta function.
"""
function log_marginal_likelihood(data::Vector{Float64}, prior_a::Float64, prior_b::Float64)
    n_heads = sum(data)
    n_tails = length(data) - n_heads
    
    posterior_a = prior_a + n_heads
    posterior_b = prior_b + n_tails
    
    # Log marginal likelihood using Beta function
    # log B(a, b) = log Γ(a) + log Γ(b) - log Γ(a + b)
    log_ml = logbeta(posterior_a, posterior_b) - logbeta(prior_a, prior_b)
    
    return log_ml
end

end # module CoinTossModel

