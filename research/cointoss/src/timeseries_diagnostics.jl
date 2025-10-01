"""
Comprehensive Timeseries Diagnostic Module

Creates complete temporal evolution visualizations showing:
- Posterior evolution (mean, mode, std, credible intervals)
- Free energy progression
- KL divergence accumulation
- Information gain
- Empirical statistics
- All RxInfer diagnostic values over time
"""
module TimeseriesDiagnostics

using Plots
using Distributions
using Statistics
using StatsPlots
using DataFrames
using SpecialFunctions: digamma, logbeta

export create_comprehensive_timeseries_dashboard
export compute_temporal_evolution
export plot_all_metrics_through_time

"""
Compute all metrics at each time step
"""
function compute_temporal_evolution(data::Vector{Float64}, prior_a::Float64, prior_b::Float64;
                                    sample_points::Union{Vector{Int}, Nothing}=nothing)
    n = length(data)
    
    # Default sample points
    if sample_points === nothing
        sample_points = sort(unique(vcat(
            collect(1:min(10, n)),                           # First 10 samples
            collect(20:10:min(100, n)),            # Every 10 up to 100
            collect(100:50:min(500, n)),           # Every 50 up to 500
            [n]                                     # Final point
        )))
    end
    
    evolution = Dict{String, Vector{Float64}}()
    
    # Initialize arrays
    evolution["n_samples"] = Float64[]
    evolution["posterior_mean"] = Float64[]
    evolution["posterior_mode"] = Float64[]
    evolution["posterior_std"] = Float64[]
    evolution["posterior_var"] = Float64[]
    evolution["ci_lower"] = Float64[]
    evolution["ci_upper"] = Float64[]
    evolution["ci_width"] = Float64[]
    
    evolution["empirical_mean"] = Float64[]
    evolution["n_heads"] = Float64[]
    evolution["n_tails"] = Float64[]
    evolution["head_rate"] = Float64[]
    
    evolution["posterior_alpha"] = Float64[]
    evolution["posterior_beta"] = Float64[]
    evolution["alpha_growth"] = Float64[]
    evolution["beta_growth"] = Float64[]
    
    evolution["kl_divergence"] = Float64[]
    evolution["information_gain"] = Float64[]
    evolution["free_energy"] = Float64[]
    evolution["log_marginal_likelihood"] = Float64[]
    evolution["expected_log_likelihood"] = Float64[]
    
    evolution["prior_mean"] = Float64[]
    evolution["posterior_prior_diff"] = Float64[]
    evolution["uncertainty_reduction"] = Float64[]
    
    # Prior statistics
    prior = Beta(prior_a, prior_b)
    prior_mean_val = mean(prior)
    prior_var = var(prior)
    
    # Compute at each sample point
    for n_obs in sample_points
        obs = data[1:n_obs]
        n_heads = sum(obs)
        n_tails = n_obs - n_heads
        
        # Posterior parameters
        α_post = prior_a + n_heads
        β_post = prior_b + n_tails
        posterior = Beta(α_post, β_post)
        
        # Posterior statistics
        post_mean = mean(posterior)
        post_var = var(posterior)
        post_std = sqrt(post_var)
        post_mode = (α_post > 1 && β_post > 1) ? (α_post - 1) / (α_post + β_post - 2) : NaN
        
        # Credible interval
        ci_lower = quantile(posterior, 0.025)
        ci_upper = quantile(posterior, 0.975)
        
        # KL divergence
        kl_div = (α_post - prior_a) * (digamma(α_post) - digamma(α_post + β_post)) +
                 (β_post - prior_b) * (digamma(β_post) - digamma(α_post + β_post)) +
                 logbeta(prior_a, prior_b) - logbeta(α_post, β_post)
        
        # Free energy (negative ELBO)
        # FE = KL(q||p) - E_q[log p(y|θ)]
        expected_ll = n_heads * (digamma(α_post) - digamma(α_post + β_post)) +
                     n_tails * (digamma(β_post) - digamma(α_post + β_post))
        fe = kl_div - expected_ll
        
        # Log marginal likelihood
        lml = logbeta(α_post, β_post) - logbeta(prior_a, prior_b) +
              sum(obs .* log.(obs .+ 1e-10) .+ (1 .- obs) .* log.(1 .- obs .+ 1e-10))
        
        # Store values
        push!(evolution["n_samples"], Float64(n_obs))
        push!(evolution["posterior_mean"], post_mean)
        push!(evolution["posterior_mode"], post_mode)
        push!(evolution["posterior_std"], post_std)
        push!(evolution["posterior_var"], post_var)
        push!(evolution["ci_lower"], ci_lower)
        push!(evolution["ci_upper"], ci_upper)
        push!(evolution["ci_width"], ci_upper - ci_lower)
        
        push!(evolution["empirical_mean"], n_heads / n_obs)
        push!(evolution["n_heads"], Float64(n_heads))
        push!(evolution["n_tails"], Float64(n_tails))
        push!(evolution["head_rate"], n_heads / n_obs)
        
        push!(evolution["posterior_alpha"], α_post)
        push!(evolution["posterior_beta"], β_post)
        push!(evolution["alpha_growth"], α_post - prior_a)
        push!(evolution["beta_growth"], β_post - prior_b)
        
        push!(evolution["kl_divergence"], kl_div)
        push!(evolution["information_gain"], kl_div)
        push!(evolution["free_energy"], fe)
        push!(evolution["log_marginal_likelihood"], lml)
        push!(evolution["expected_log_likelihood"], expected_ll)
        
        push!(evolution["prior_mean"], prior_mean_val)
        push!(evolution["posterior_prior_diff"], post_mean - prior_mean_val)
        push!(evolution["uncertainty_reduction"], prior_var - post_var)
    end
    
    return evolution
end

"""
Create comprehensive timeseries dashboard with all metrics
"""
function create_comprehensive_timeseries_dashboard(
    data::Vector{Float64},
    prior_a::Float64,
    prior_b::Float64;
    true_theta::Union{Float64, Nothing}=nothing,
    theme::String="default"
)
    # Compute temporal evolution
    evolution = compute_temporal_evolution(data, prior_a, prior_b)
    n_samples = evolution["n_samples"]
    
    # Color palette
    colors = palette(:tab10)
    
    # Create 4x3 grid of plots
    plots_array = []
    
    # Plot 1: Posterior Mean Evolution
    p1 = plot(n_samples, evolution["posterior_mean"],
              label="Posterior Mean", lw=2, color=colors[1],
              xlabel="Number of Observations", ylabel="θ",
              title="Posterior Mean Evolution", legend=:bottomright)
    plot!(p1, n_samples, evolution["empirical_mean"],
          label="Empirical Mean", lw=2, ls=:dash, color=colors[2])
    if true_theta !== nothing
        hline!(p1, [true_theta], label="True θ", lw=2, ls=:dot, color=:black)
    end
    push!(plots_array, p1)
    
    # Plot 2: Credible Interval Evolution
    p2 = plot(n_samples, evolution["ci_lower"],
              fillrange=evolution["ci_upper"], fillalpha=0.3, fillcolor=colors[1],
              label="95% CI", lw=1, color=colors[1],
              xlabel="Number of Observations", ylabel="θ",
              title="Credible Interval Evolution")
    plot!(p2, n_samples, evolution["posterior_mean"],
          label="Posterior Mean", lw=2, color=colors[3])
    if true_theta !== nothing
        hline!(p2, [true_theta], label="True θ", lw=2, ls=:dot, color=:black)
    end
    push!(plots_array, p2)
    
    # Plot 3: Uncertainty Reduction (Std Dev)
    p3 = plot(n_samples, evolution["posterior_std"],
              label="Posterior Std", lw=2, color=colors[4],
              xlabel="Number of Observations", ylabel="Standard Deviation",
              title="Uncertainty Reduction", legend=:topright)
    push!(plots_array, p3)
    
    # Plot 4: CI Width Evolution
    p4 = plot(n_samples, evolution["ci_width"],
              label="95% CI Width", lw=2, color=colors[5],
              xlabel="Number of Observations", ylabel="CI Width",
              title="Credible Interval Width", legend=:topright)
    push!(plots_array, p4)
    
    # Plot 5: Posterior Parameters (Alpha & Beta)
    p5 = plot(n_samples, evolution["posterior_alpha"],
              label="α (posterior)", lw=2, color=colors[6],
              xlabel="Number of Observations", ylabel="Parameter Value",
              title="Posterior Parameters Evolution", legend=:topleft)
    plot!(p5, n_samples, evolution["posterior_beta"],
          label="β (posterior)", lw=2, color=colors[7])
    push!(plots_array, p5)
    
    # Plot 6: Parameter Growth
    p6 = plot(n_samples, evolution["alpha_growth"],
              label="Δα (heads count)", lw=2, color=colors[6],
              xlabel="Number of Observations", ylabel="Parameter Growth",
              title="Parameter Growth from Prior", legend=:topleft)
    plot!(p6, n_samples, evolution["beta_growth"],
          label="Δβ (tails count)", lw=2, color=colors[7])
    push!(plots_array, p6)
    
    # Plot 7: KL Divergence (Information Gain)
    p7 = plot(n_samples, evolution["kl_divergence"],
              label="KL Divergence", lw=2, color=colors[8],
              xlabel="Number of Observations", ylabel="KL(posterior || prior)",
              title="Information Gain (KL Divergence)", legend=:bottomright)
    push!(plots_array, p7)
    
    # Plot 8: Free Energy
    p8 = plot(n_samples, evolution["free_energy"],
              label="Free Energy", lw=2, color=colors[9],
              xlabel="Number of Observations", ylabel="Free Energy",
              title="Bethe Free Energy", legend=:topright)
    push!(plots_array, p8)
    
    # Plot 9: Log Marginal Likelihood
    p9 = plot(n_samples, evolution["log_marginal_likelihood"],
              label="Log P(y)", lw=2, color=colors[10],
              xlabel="Number of Observations", ylabel="Log Marginal Likelihood",
              title="Model Evidence Evolution", legend=:bottomright)
    push!(plots_array, p9)
    
    # Plot 10: Expected Log Likelihood
    p10 = plot(n_samples, evolution["expected_log_likelihood"],
               label="E[log p(y|θ)]", lw=2, color=colors[1],
               xlabel="Number of Observations", ylabel="Expected LL",
               title="Expected Log Likelihood", legend=:bottomright)
    push!(plots_array, p10)
    
    # Plot 11: Data Accumulation (Heads vs Tails)
    p11 = plot(n_samples, evolution["n_heads"],
               label="Heads Count", lw=2, color=:green,
               xlabel="Number of Observations", ylabel="Count",
               title="Data Accumulation", legend=:topleft)
    plot!(p11, n_samples, evolution["n_tails"],
          label="Tails Count", lw=2, color=:red)
    push!(plots_array, p11)
    
    # Plot 12: Posterior-Prior Divergence
    p12 = plot(n_samples, evolution["posterior_prior_diff"],
               label="Posterior - Prior Mean", lw=2, color=colors[2],
               xlabel="Number of Observations", ylabel="Difference",
               title="Learning Progress", legend=:bottomright)
    hline!(p12, [0], label="No Learning", lw=1, ls=:dash, color=:gray)
    push!(plots_array, p12)
    
    # Combine all plots
    dashboard = plot(plots_array...,
                    layout=(4, 3),
                    size=(1800, 1600),
                    plot_title="Comprehensive Temporal Evolution Dashboard",
                    plot_titlevspan=0.05,
                    margin=5Plots.mm)
    
    return dashboard, evolution
end

"""
Create individual timeseries plots for each metric
"""
function plot_all_metrics_through_time(
    data::Vector{Float64},
    prior_a::Float64,
    prior_b::Float64;
    true_theta::Union{Float64, Nothing}=nothing,
    output_dir::String="outputs/timeseries"
)
    mkpath(output_dir)
    
    evolution = compute_temporal_evolution(data, prior_a, prior_b)
    n_samples = evolution["n_samples"]
    
    plots = Dict{String, Plots.Plot}()
    
    # Create individual plots for each metric
    metrics = [
        ("posterior_mean", "Posterior Mean θ", "θ"),
        ("posterior_mode", "Posterior Mode θ", "θ"),
        ("posterior_std", "Posterior Std Dev", "σ"),
        ("posterior_var", "Posterior Variance", "σ²"),
        ("ci_width", "95% CI Width", "Width"),
        ("posterior_alpha", "Posterior α Parameter", "α"),
        ("posterior_beta", "Posterior β Parameter", "β"),
        ("kl_divergence", "KL Divergence (Information Gain)", "KL(q||p)"),
        ("free_energy", "Bethe Free Energy", "FE"),
        ("log_marginal_likelihood", "Log Marginal Likelihood", "log P(y)"),
        ("expected_log_likelihood", "Expected Log Likelihood", "E[log p(y|θ)]"),
        ("empirical_mean", "Empirical Mean (Data)", "Empirical θ"),
        ("head_rate", "Head Rate", "P(heads)"),
        ("uncertainty_reduction", "Variance Reduction", "Δσ²"),
        ("posterior_prior_diff", "Posterior - Prior Mean", "Δμ")
    ]
    
    for (key, title, ylabel) in metrics
        p = plot(n_samples, evolution[key],
                label=title, lw=2,
                xlabel="Number of Observations",
                ylabel=ylabel,
                title=title,
                legend=:best)
        
        if key in ["posterior_mean", "empirical_mean", "head_rate"] && true_theta !== nothing
            hline!(p, [true_theta], label="True θ", lw=2, ls=:dot, color=:black)
        end
        
        plots[key] = p
        
        # Save individual plot
        savefig(p, joinpath(output_dir, "$(key)_timeseries.png"))
    end
    
    return plots, evolution
end

end # module

