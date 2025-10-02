# Inference module for Coin Toss model
# Handles RxInfer execution, diagnostics, and convergence tracking

module CoinTossInference

using RxInfer
using Distributions
using Statistics
using SpecialFunctions: digamma, logbeta
using Random

export run_inference, InferenceResult, track_free_energy, 
       compute_convergence_diagnostics, posterior_predictive_check,
       kl_divergence, expected_log_likelihood, compute_inference_diagnostics

"""
    InferenceResult

Container for comprehensive inference results.
"""
struct InferenceResult
    posterior::Beta                          # Posterior distribution
    prior::Beta                              # Prior distribution
    observations::Vector{Float64}            # Observed data
    free_energy::Union{Vector{Float64}, Nothing}  # Free energy trace (if tracked)
    execution_time::Float64                  # Inference execution time
    iterations::Int                          # Number of iterations
    converged::Bool                          # Convergence status
    convergence_iteration::Union{Int, Nothing}  # Iteration where convergence occurred
    diagnostics::Dict{String, Any}          # Additional diagnostics
end

"""
    run_inference(data::Vector{Float64}, prior_a::Float64, prior_b::Float64;
                  model_func=nothing, iterations::Int=10, track_fe::Bool=true, 
                  convergence_check::Bool=true, convergence_tol::Float64=1e-6,
                  showprogress::Bool=true)

Run Bayesian inference using RxInfer with comprehensive diagnostics.

# Arguments
- `data`: Vector of binary observations (0s and 1s)
- `prior_a`: Beta prior parameter a
- `prior_b`: Beta prior parameter b
- `model_func`: The RxInfer model function (default: use coin_model from global scope)
- `iterations`: Number of inference iterations
- `track_fe`: Track free energy during inference
- `convergence_check`: Check for convergence
- `convergence_tol`: Convergence tolerance for free energy
- `showprogress`: Show progress during inference

# Returns
- `InferenceResult`: Comprehensive inference results
"""
function run_inference(data::Vector{Float64}, prior_a::Float64, prior_b::Float64;
                       model_func=nothing, iterations::Int=10, track_fe::Bool=true, 
                       convergence_check::Bool=true, convergence_tol::Float64=1e-6,
                       showprogress::Bool=true)
    
    @info "Starting inference" n_observations=length(data) prior_a=prior_a prior_b=prior_b
    
    # Get model function from global scope if not provided
    if model_func === nothing
        model_func = Main.CoinTossModel.coin_model
    end
    
    # Time the inference
    start_time = time()
    
    # Run inference with RxInfer
    # Note: For conjugate Beta-Bernoulli, inference is analytical, but we can still
    # use RxInfer's framework for consistency and to enable future extensions
    result = infer(
        model = model_func(a = prior_a, b = prior_b),
        data = (y = data,),
        iterations = iterations,
        free_energy = track_fe,
        showprogress = showprogress
    )
    
    execution_time = time() - start_time
    
    # Extract posterior (get the last iteration if it's a vector)
    posterior_raw = result.posteriors[:θ]
    posterior = posterior_raw isa Vector ? posterior_raw[end] : posterior_raw
    prior = Beta(prior_a, prior_b)
    
    # Extract free energy if tracked and convert to Float64
    free_energy = track_fe ? (result.free_energy isa Vector ? Float64.(result.free_energy) : nothing) : nothing
    
    # Check convergence
    converged = false
    convergence_iteration = nothing
    
    if convergence_check && free_energy !== nothing && length(free_energy) > 1
        # Check for convergence based on free energy change
        for i in 2:length(free_energy)
            fe_change = abs(free_energy[i] - free_energy[i-1])
            if fe_change < convergence_tol
                converged = true
                convergence_iteration = i
                @info "Convergence detected" iteration=i fe_change=fe_change
                break
            end
        end
        
        if !converged
            @warn "Did not converge within tolerance" final_fe_change=abs(free_energy[end] - free_energy[end-1])
        end
    end
    
    # Compute diagnostics
    diagnostics = compute_inference_diagnostics(posterior, prior, data, free_energy)
    
    @info "Inference completed" execution_time=round(execution_time, digits=4) converged=converged
    
    return InferenceResult(
        posterior,
        prior,
        data,
        free_energy,
        execution_time,
        iterations,
        converged,
        convergence_iteration,
        diagnostics
    )
end

"""
    compute_inference_diagnostics(posterior::Beta, prior::Beta, 
                                   data::Vector{Float64}, 
                                   free_energy::Union{Vector{Float64}, Nothing})

Compute comprehensive inference diagnostics.
"""
function compute_inference_diagnostics(posterior::Beta, prior::Beta, 
                                        data::Vector{Float64}, 
                                        free_energy::Union{Vector{Float64}, Nothing})
    
    # Basic statistics
    n_heads = sum(data)
    n_tails = length(data) - n_heads
    empirical_rate = n_heads / length(data)
    
    # KL divergence from prior to posterior
    kl_div = kl_divergence(posterior, prior)
    
    # Expected log likelihood
    expected_ll = expected_log_likelihood(posterior, data)
    
    # Posterior statistics
    post_mean = mean(posterior)
    post_var = var(posterior)
    post_mode = params(posterior)[1] > 1 && params(posterior)[2] > 1 ? 
                mode(posterior) : NaN
    
    # Prior statistics  
    prior_mean = mean(prior)
    prior_var = var(prior)
    
    # Information gain
    info_gain = kl_div
    
    diagnostics = Dict(
        "n_observations" => length(data),
        "n_heads" => n_heads,
        "n_tails" => n_tails,
        "empirical_rate" => empirical_rate,
        "kl_divergence" => kl_div,
        "information_gain" => info_gain,
        "expected_log_likelihood" => expected_ll,
        "posterior_mean" => post_mean,
        "posterior_variance" => post_var,
        "posterior_mode" => post_mode,
        "prior_mean" => prior_mean,
        "prior_variance" => prior_var,
        "mean_shift" => post_mean - prior_mean,
        "variance_reduction" => prior_var - post_var
    )
    
    # Add free energy diagnostics if available
    if free_energy !== nothing && !isempty(free_energy)
        diagnostics["final_free_energy"] = free_energy[end]
        diagnostics["initial_free_energy"] = free_energy[1]
        diagnostics["free_energy_reduction"] = free_energy[1] - free_energy[end]
        
        if length(free_energy) > 1
            fe_changes = diff(free_energy)
            diagnostics["mean_fe_change"] = mean(abs.(fe_changes))
            diagnostics["max_fe_change"] = maximum(abs.(fe_changes))
            diagnostics["final_fe_change"] = abs(fe_changes[end])
        end
    end
    
    return diagnostics
end

"""
    kl_divergence(q::Beta, p::Beta)

Compute KL divergence KL(q || p) between two Beta distributions.

Uses analytical formula:
KL(Beta(α₁,β₁) || Beta(α₂,β₂)) = log[B(α₂,β₂)/B(α₁,β₁)] + 
    (α₁-α₂)[ψ(α₁)-ψ(α₁+β₁)] + (β₁-β₂)[ψ(β₁)-ψ(α₁+β₁)]

where ψ is the digamma function and B is the beta function.
"""
function kl_divergence(q::Beta, p::Beta)
    α₁, β₁ = params(q)
    α₂, β₂ = params(p)
    
    kl = logbeta(α₂, β₂) - logbeta(α₁, β₁) +
         (α₁ - α₂) * (digamma(α₁) - digamma(α₁ + β₁)) +
         (β₁ - β₂) * (digamma(β₁) - digamma(α₁ + β₁))
    
    return kl
end

"""
    expected_log_likelihood(posterior::Beta, data::Vector{Float64})

Compute expected log likelihood under the posterior.

E_q[log p(data | θ)] where q is the posterior
"""
function expected_log_likelihood(posterior::Beta, data::Vector{Float64})
    α, β = params(posterior)
    n_heads = sum(data)
    n_tails = length(data) - n_heads
    
    # E[log θ] = ψ(α) - ψ(α + β)
    # E[log(1-θ)] = ψ(β) - ψ(α + β)
    expected_log_theta = digamma(α) - digamma(α + β)
    expected_log_one_minus_theta = digamma(β) - digamma(α + β)
    
    ell = n_heads * expected_log_theta + n_tails * expected_log_one_minus_theta
    
    return ell
end

"""
    posterior_predictive_check(posterior::Beta, n_samples::Int=10000; seed::Int=123)

Generate posterior predictive samples for model checking.

# Returns
- Dict with predictive statistics and samples
"""
function posterior_predictive_check(posterior::Beta, n_samples::Int=10000; seed::Int=123)
    rng = MersenneTwister(seed)
    
    # Sample θ from posterior
    theta_samples = rand(rng, posterior, n_samples)
    
    # For each θ, sample a binary outcome
    predictive_samples = [rand(rng, Bernoulli(θ)) for θ in theta_samples]
    
    # Compute predictive statistics
    predictive_mean = mean(predictive_samples)
    predictive_var = var(predictive_samples)
    
    # Posterior predictive probability of heads
    # This equals E[θ] = mean of posterior
    pp_heads = mean(theta_samples)
    
    return Dict(
        "theta_samples" => theta_samples,
        "predictive_samples" => predictive_samples,
        "predictive_mean" => predictive_mean,
        "predictive_variance" => predictive_var,
        "pp_prob_heads" => pp_heads,
        "n_samples" => n_samples
    )
end

"""
    track_free_energy(model_fn, data; iterations::Int=50)

Track free energy convergence over iterations.

# Returns
- Vector of free energy values
"""
function track_free_energy(model_fn, data; iterations::Int=50)
    result = infer(
        model = model_fn(),
        data = data,
        iterations = iterations,
        free_energy = true,
        showprogress = false
    )
    
    return result.free_energy
end

"""
    compute_convergence_diagnostics(free_energy::Vector{Float64})

Analyze free energy convergence behavior.
"""
function compute_convergence_diagnostics(free_energy::Vector{Float64})
    if isempty(free_energy) || length(free_energy) < 2
        return Dict("error" => "Insufficient free energy data")
    end
    
    fe_changes = diff(free_energy)
    
    return Dict(
        "n_iterations" => length(free_energy),
        "initial_fe" => free_energy[1],
        "final_fe" => free_energy[end],
        "total_reduction" => free_energy[1] - free_energy[end],
        "mean_change" => mean(abs.(fe_changes)),
        "max_change" => maximum(abs.(fe_changes)),
        "final_change" => abs(fe_changes[end]),
        "monotonic_decrease" => all(fe_changes .<= 0),
        "fe_trace" => free_energy
    )
end

end # module CoinTossInference

