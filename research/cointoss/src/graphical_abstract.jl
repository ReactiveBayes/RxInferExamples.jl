"""
Comprehensive Graphical Abstract Module

Creates a unified mega-visualization combining all statistical, computational,
and diagnostic information into a single comprehensive graphical abstract.
"""
module GraphicalAbstract

using Plots
using Distributions
using Statistics
using DataFrames
using CSV
using JSON
using Printf

export create_graphical_abstract, create_mega_dashboard

"""
Create comprehensive graphical abstract with all metrics, diagnostics, and computational logs
"""
function create_graphical_abstract(
    data::Vector{Float64},
    prior_a::Float64,
    prior_b::Float64,
    result,
    diagnostics,
    temporal_evolution::Dict{String, Vector{Float64}};
    true_theta::Union{Float64, Nothing}=nothing,
    benchmark_stats=nothing,
    theme::String="default"
)
    
    # Create 6x4 mega grid (24 panels)
    plots_array = []
    
    # Extract key data
    n_samples = temporal_evolution["n_samples"]
    posterior_raw = result.posteriors[:θ]
    posterior_marginal = posterior_raw isa Vector ? posterior_raw[end] : posterior_raw
    α, β = params(posterior_marginal)
    posterior = Beta(α, β)
    prior = Beta(prior_a, prior_b)
    
    colors = palette(:tab10)
    
    # ============================================================================
    # ROW 1: POSTERIOR EVOLUTION & KEY METRICS
    # ============================================================================
    
    # Plot 1: Posterior Mean Evolution with Empirical
    p1 = plot(n_samples, temporal_evolution["posterior_mean"],
              label="Posterior Mean", lw=3, color=colors[1],
              xlabel="n", ylabel="θ", title="Posterior Mean Evolution",
              legend=:bottomright, grid=true)
    plot!(p1, n_samples, temporal_evolution["empirical_mean"],
          label="Empirical", lw=2, ls=:dash, color=colors[2])
    if true_theta !== nothing
        hline!(p1, [true_theta], label="True θ", lw=2, ls=:dot, color=:black, alpha=0.5)
    end
    push!(plots_array, p1)
    
    # Plot 2: Credible Interval Evolution
    p2 = plot(n_samples, temporal_evolution["ci_lower"],
              fillrange=temporal_evolution["ci_upper"], fillalpha=0.3, fillcolor=colors[1],
              label="95% CI", lw=1, color=colors[1],
              xlabel="n", ylabel="θ", title="Credible Interval Evolution",
              legend=:topright, grid=true)
    plot!(p2, n_samples, temporal_evolution["posterior_mean"],
          label="Mean", lw=2, color=colors[3])
    if true_theta !== nothing
        hline!(p2, [true_theta], label="True θ", lw=2, ls=:dot, color=:black, alpha=0.5)
    end
    push!(plots_array, p2)
    
    # Plot 3: Uncertainty Reduction (Std + Variance)
    p3 = plot(n_samples, temporal_evolution["posterior_std"],
              label="Posterior Std", lw=2, color=colors[4],
              xlabel="n", ylabel="σ", title="Uncertainty Reduction",
              legend=:topright, grid=true)
    push!(plots_array, p3)
    
    # Plot 4: CI Width Reduction
    p4 = plot(n_samples, temporal_evolution["ci_width"],
              label="95% CI Width", lw=2, color=colors[5], fill=0, fillalpha=0.2,
              xlabel="n", ylabel="Width", title="CI Width Reduction",
              legend=:topright, grid=true)
    push!(plots_array, p4)
    
    # ============================================================================
    # ROW 2: BAYESIAN LEARNING DYNAMICS
    # ============================================================================
    
    # Plot 5: KL Divergence (Information Gain)
    p5 = plot(n_samples, temporal_evolution["kl_divergence"],
              label="KL(q||p)", lw=2, color=colors[8], fill=0, fillalpha=0.2,
              xlabel="n", ylabel="nats", title="Information Gain (KL Divergence)",
              legend=:bottomright, grid=true)
    push!(plots_array, p5)
    
    # Plot 6: Free Energy Evolution
    p6 = plot(n_samples, temporal_evolution["free_energy"],
              label="Free Energy", lw=2, color=colors[9],
              xlabel="n", ylabel="FE", title="Bethe Free Energy",
              legend=:topright, grid=true)
    push!(plots_array, p6)
    
    # Plot 7: Log Marginal Likelihood
    p7 = plot(n_samples, temporal_evolution["log_marginal_likelihood"],
              label="log P(y)", lw=2, color=colors[10],
              xlabel="n", ylabel="log P(y)", title="Model Evidence Evolution",
              legend=:bottomright, grid=true)
    push!(plots_array, p7)
    
    # Plot 8: Expected Log Likelihood
    p8 = plot(n_samples, temporal_evolution["expected_log_likelihood"],
              label="E[log p(y|θ)]", lw=2, color=colors[1],
              xlabel="n", ylabel="E[LL]", title="Expected Log Likelihood",
              legend=:bottomright, grid=true)
    push!(plots_array, p8)
    
    # ============================================================================
    # ROW 3: POSTERIOR PARAMETERS & DATA
    # ============================================================================
    
    # Plot 9: Posterior Alpha Parameter
    p9 = plot(n_samples, temporal_evolution["posterior_alpha"],
              label="α (posterior)", lw=2, color=colors[6],
              xlabel="n", ylabel="α", title="Posterior α Parameter",
              legend=:bottomright, grid=true)
    hline!(p9, [prior_a], label="α (prior)", lw=1, ls=:dash, color=:gray)
    push!(plots_array, p9)
    
    # Plot 10: Posterior Beta Parameter
    p10 = plot(n_samples, temporal_evolution["posterior_beta"],
               label="β (posterior)", lw=2, color=colors[7],
               xlabel="n", ylabel="β", title="Posterior β Parameter",
               legend=:bottomright, grid=true)
    hline!(p10, [prior_b], label="β (prior)", lw=1, ls=:dash, color=:gray)
    push!(plots_array, p10)
    
    # Plot 11: Data Accumulation (Heads vs Tails)
    p11 = plot(n_samples, temporal_evolution["n_heads"],
               label="Heads", lw=2, color=:green, fill=0, fillalpha=0.2,
               xlabel="n", ylabel="Count", title="Data Accumulation",
               legend=:topleft, grid=true)
    plot!(p11, n_samples, temporal_evolution["n_tails"],
          label="Tails", lw=2, color=:red, fill=0, fillalpha=0.2)
    push!(plots_array, p11)
    
    # Plot 12: Head Rate Evolution
    p12 = plot(n_samples, temporal_evolution["head_rate"],
               label="Empirical Rate", lw=2, color=colors[2],
               xlabel="n", ylabel="P(heads)", title="Empirical Head Rate",
               legend=:bottomright, grid=true)
    if true_theta !== nothing
        hline!(p12, [true_theta], label="True θ", lw=2, ls=:dot, color=:black, alpha=0.5)
    end
    push!(plots_array, p12)
    
    # ============================================================================
    # ROW 4: CONVERGENCE & DIAGNOSTICS
    # ============================================================================
    
    # Plot 13: Posterior-Prior Shift
    p13 = plot(n_samples, temporal_evolution["posterior_prior_diff"],
               label="μ_post - μ_prior", lw=2, color=colors[2],
               xlabel="n", ylabel="Δμ", title="Learning Progress (Mean Shift)",
               legend=:bottomright, grid=true)
    hline!(p13, [0], label="No Learning", lw=1, ls=:dash, color=:gray, alpha=0.5)
    push!(plots_array, p13)
    
    # Plot 14: Variance Reduction
    p14 = plot(n_samples, temporal_evolution["uncertainty_reduction"],
               label="σ²_prior - σ²_post", lw=2, color=colors[4], fill=0, fillalpha=0.2,
               xlabel="n", ylabel="Δσ²", title="Variance Reduction",
               legend=:bottomright, grid=true)
    push!(plots_array, p14)
    
    # Plot 15: Parameter Growth (Alpha)
    p15 = plot(n_samples, temporal_evolution["alpha_growth"],
               label="Δα (heads)", lw=2, color=colors[6], fill=0, fillalpha=0.2,
               xlabel="n", ylabel="Δα", title="Alpha Growth (Heads Count)",
               legend=:bottomright, grid=true)
    push!(plots_array, p15)
    
    # Plot 16: Parameter Growth (Beta)
    p16 = plot(n_samples, temporal_evolution["beta_growth"],
               label="Δβ (tails)", lw=2, color=colors[7], fill=0, fillalpha=0.2,
               xlabel="n", ylabel="Δβ", title="Beta Growth (Tails Count)",
               legend=:bottomright, grid=true)
    push!(plots_array, p16)
    
    # ============================================================================
    # ROW 5: FINAL DISTRIBUTIONS & COMPARISONS
    # ============================================================================
    
    # Plot 17: Prior vs Posterior PDFs
    θ_range = range(0, 1, length=500)
    p17 = plot(θ_range, pdf.(prior, θ_range),
               label="Prior", lw=2, color=colors[1], fillalpha=0.2, fill=0,
               xlabel="θ", ylabel="Density", title="Prior vs Posterior",
               legend=:topright, grid=true)
    plot!(p17, θ_range, pdf.(posterior, θ_range),
          label="Posterior", lw=3, color=colors[3], fillalpha=0.3, fill=0)
    if true_theta !== nothing
        vline!(p17, [true_theta], label="True θ", lw=2, ls=:dot, color=:black)
    end
    push!(plots_array, p17)
    
    # Plot 18: Final Posterior with Statistics
    ci_lower = quantile(posterior, 0.025)
    ci_upper = quantile(posterior, 0.975)
    p18 = plot(θ_range, pdf.(posterior, θ_range),
               label="Posterior", lw=3, color=colors[3], fillalpha=0.3, fill=0,
               xlabel="θ", ylabel="Density", title="Final Posterior Distribution",
               legend=:topright, grid=true)
    vline!(p18, [mean(posterior)], label="Mean", lw=2, ls=:dash, color=colors[1])
    vline!(p18, [ci_lower, ci_upper], label="95% CI", lw=1, ls=:dot, color=colors[5])
    if true_theta !== nothing
        vline!(p18, [true_theta], label="True θ", lw=2, ls=:solid, color=:black)
    end
    push!(plots_array, p18)
    
    # Plot 19: Data Histogram
    n_heads = sum(data)
    n_tails = length(data) - n_heads
    p19 = bar(["Heads", "Tails"], [n_heads, n_tails],
              color=[:green :red], alpha=0.6,
              xlabel="Outcome", ylabel="Count", title="Observed Data",
              legend=false, grid=true)
    annotate!(p19, [(0.7, n_heads * 0.9, text(@sprintf("%.1f%%", 100 * n_heads / length(data)), 10)),
                    (1.7, n_tails * 0.9, text(@sprintf("%.1f%%", 100 * n_tails / length(data)), 10))])
    push!(plots_array, p19)
    
    # Plot 20: Posterior Mode Evolution
    p20 = plot(n_samples, temporal_evolution["posterior_mode"],
               label="Mode", lw=2, color=colors[3],
               xlabel="n", ylabel="θ", title="Posterior Mode Evolution",
               legend=:bottomright, grid=true)
    if true_theta !== nothing
        hline!(p20, [true_theta], label="True θ", lw=2, ls=:dot, color=:black, alpha=0.5)
    end
    push!(plots_array, p20)
    
    # ============================================================================
    # ROW 6: COMPUTATIONAL DIAGNOSTICS & SUMMARY
    # ============================================================================
    
    # Plot 21: Summary Statistics Table (as annotation plot)
    p21 = plot(xlims=(0, 1), ylims=(0, 1), framestyle=:none, legend=false,
               title="Summary Statistics", titlefontsize=10)
    stats_text = """
    Final Posterior: Beta($(round(α, digits=1)), $(round(β, digits=1)))
    Mean: $(round(mean(posterior), digits=4))
    Mode: $(round((α-1)/(α+β-2), digits=4))
    Std:  $(round(std(posterior), digits=4))
    95% CI: [$(round(ci_lower, digits=3)), $(round(ci_upper, digits=3))]
    
    Data: n=$(length(data)), heads=$(Int(n_heads))
    Empirical: $(round(n_heads/length(data), digits=4))
    """
    annotate!(p21, [(0.5, 0.5, text(stats_text, :center, 8))])
    push!(plots_array, p21)
    
    # Plot 22: Benchmark Summary (if available)
    p22 = plot(xlims=(0, 1), ylims=(0, 1), framestyle=:none, legend=false,
               title="Performance Metrics", titlefontsize=10)
    if benchmark_stats !== nothing
        bench_text = """
        Model Creation: $(round(benchmark_stats[1, :Mean_μs], digits=2)) μs
        Inference:      $(round(benchmark_stats[2, :Mean_μs], digits=2)) μs
        Per Iteration:  $(round(benchmark_stats[3, :Mean_μs], digits=2)) μs
        
        Total Runs: 3
        """
        annotate!(p22, [(0.5, 0.5, text(bench_text, :center, 8))])
    else
        annotate!(p22, [(0.5, 0.5, text("No benchmark data", :center, 10))])
    end
    push!(plots_array, p22)
    
    # Plot 23: Diagnostic Summary (if available)
    p23 = plot(xlims=(0, 1), ylims=(0, 1), framestyle=:none, legend=false,
               title="Diagnostic Summary", titlefontsize=10)
    if diagnostics !== nothing
        diag_text = """
        Memory Addon: Enabled
        Callbacks: $(diagnostics.callback_trace !== nothing ? length(diagnostics.callback_trace) : 0) events
        
        KL Divergence: $(round(temporal_evolution["kl_divergence"][end], digits=4))
        Info Gain: $(round(temporal_evolution["information_gain"][end], digits=4)) nats
        """
        annotate!(p23, [(0.5, 0.5, text(diag_text, :center, 8))])
    else
        annotate!(p23, [(0.5, 0.5, text("No diagnostic data", :center, 10))])
    end
    push!(plots_array, p23)
    
    # Plot 24: Configuration Summary
    p24 = plot(xlims=(0, 1), ylims=(0, 1), framestyle=:none, legend=false,
               title="Configuration", titlefontsize=10)
    config_text = """
    Prior: Beta($(prior_a), $(prior_b))
    True θ: $(true_theta !== nothing ? round(true_theta, digits=3) : "N/A")
    Samples: $(length(data))
    
    Final FE: $(round(temporal_evolution["free_energy"][end], digits=2))
    """
    annotate!(p24, [(0.5, 0.5, text(config_text, :center, 8))])
    push!(plots_array, p24)
    
    # Combine into mega dashboard
    mega_dashboard = plot(plots_array...,
                         layout=(6, 4),
                         size=(2400, 3600),
                         plot_title="COMPREHENSIVE GRAPHICAL ABSTRACT: Bayesian Coin Toss Inference",
                         plot_titlefontsize=20,
                         plot_titlevspan=0.02,
                         margin=3Plots.mm,
                         thickness_scaling=1.2)
    
    return mega_dashboard
end

"""
Create mega dashboard combining all visualizations
"""
function create_mega_dashboard(
    data::Vector{Float64},
    prior_a::Float64,
    prior_b::Float64,
    result,
    diagnostics,
    temporal_evolution::Dict{String, Vector{Float64}};
    true_theta::Union{Float64, Nothing}=nothing,
    benchmark_stats=nothing,
    theme::String="default"
)
    return create_graphical_abstract(
        data, prior_a, prior_b, result, diagnostics, temporal_evolution;
        true_theta=true_theta,
        benchmark_stats=benchmark_stats,
        theme=theme
    )
end

end # module

