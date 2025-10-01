# Visualization module for Coin Toss model
# Comprehensive plotting and animation capabilities

module CoinTossVisualization

using Plots
using Distributions
using Statistics
using StatsPlots
using Random

export plot_prior_posterior, plot_convergence, plot_data_histogram,
       plot_credible_interval, plot_predictive, create_inference_animation,
       plot_comprehensive_dashboard, save_plot, get_theme_colors,
       plot_posterior_evolution

"""
    get_theme_colors(theme::String="default")

Get color scheme for specified theme.
"""
function get_theme_colors(theme::String="default")
    if theme == "dark"
        return (
            background = "#2b2b2b",
            prior = "#4dabf7",
            posterior = "#ff6b6b",
            data = "#51cf66",
            true_value = "#ffd43b",
            grid = "#404040"
        )
    elseif theme == "colorblind"
        return (
            background = "#ffffff",
            prior = "#0173b2",
            posterior = "#de8f05",
            data = "#029e73",
            true_value = "#cc78bc",
            grid = "#e0e0e0"
        )
    else  # default
        return (
            background = "#ffffff",
            prior = "#3498db",
            posterior = "#e74c3c",
            data = "#2ecc71",
            true_value = "#f39c12",
            grid = "#ecf0f1"
        )
    end
end

"""
    plot_prior_posterior(prior::Beta, posterior::Beta; 
                         theta_real::Union{Float64, Nothing}=nothing,
                         resolution::Int=1000, theme::String="default")

Plot prior and posterior distributions.
"""
function plot_prior_posterior(prior::Beta, posterior::Beta; 
                               theta_real::Union{Float64, Nothing}=nothing,
                               resolution::Int=1000, theme::String="default")
    
    colors = get_theme_colors(theme)
    
    # Create range for theta
    θ_range = range(0, 1, length=resolution)
    
    # Compute PDFs
    prior_pdf = pdf.(prior, θ_range)
    posterior_pdf = pdf.(posterior, θ_range)
    
    # Create plot
    p = plot(
        title = "Prior and Posterior Distributions",
        xlabel = "θ (Coin Bias)",
        ylabel = "Probability Density",
        legend = :topright,
        grid = true,
        gridcolor = colors.grid,
        background_color = colors.background,
        size = (800, 600)
    )
    
    # Plot prior
    plot!(p, θ_range, prior_pdf,
          fillrange = 0,
          fillalpha = 0.3,
          label = "Prior P(θ)",
          color = colors.prior,
          linewidth = 2)
    
    # Plot posterior
    plot!(p, θ_range, posterior_pdf,
          fillrange = 0,
          fillalpha = 0.3,
          label = "Posterior P(θ|data)",
          color = colors.posterior,
          linewidth = 2)
    
    # Plot true value if provided
    if theta_real !== nothing
        vline!(p, [theta_real],
               label = "True θ = $(round(theta_real, digits=3))",
               color = colors.true_value,
               linewidth = 3,
               linestyle = :dash)
    end
    
    # Add posterior mean and mode
    post_mean = mean(posterior)
    α, β = params(posterior)
    if α > 1 && β > 1
        post_mode = mode(posterior)
        vline!(p, [post_mode],
               label = "Posterior Mode = $(round(post_mode, digits=3))",
               color = colors.posterior,
               linewidth = 2,
               linestyle = :dot,
               alpha = 0.7)
    end
    
    vline!(p, [post_mean],
           label = "Posterior Mean = $(round(post_mean, digits=3))",
           color = colors.posterior,
           linewidth = 2,
           linestyle = :dashdot,
           alpha = 0.7)
    
    return p
end

"""
    plot_convergence(free_energy::Vector{Float64}; theme::String="default")

Plot free energy convergence.
"""
function plot_convergence(free_energy::Vector{Float64}; theme::String="default")
    colors = get_theme_colors(theme)
    
    # Calculate convergence metrics
    fe_changes = length(free_energy) > 1 ? [abs(free_energy[i] - free_energy[i-1]) for i in 2:length(free_energy)] : [0.0]
    max_change = maximum(fe_changes)
    final_change = length(fe_changes) > 0 ? fe_changes[end] : 0.0
    
    p = plot(
        title = "Free Energy Convergence\n(Lower is Better)",
        xlabel = "Iteration",
        ylabel = "Free Energy (Variational Bound)",
        legend = :best,
        grid = true,
        gridcolor = colors.grid,
        background_color = colors.background,
        size = (900, 600),
        margin = 5Plots.mm
    )
    
    # Plot free energy trajectory
    plot!(p, 1:length(free_energy), free_energy,
          label = "Free Energy",
          color = colors.posterior,
          linewidth = 3,
          marker = :circle,
          markersize = 5)
    
    # Add annotations with comprehensive info
    initial_fe = free_energy[1]
    final_fe = free_energy[end]
    fe_reduction = initial_fe - final_fe
    
    # Annotation text
    ann_text = """
    Initial FE: $(round(initial_fe, digits=4))
    Final FE: $(round(final_fe, digits=4))
    Reduction: $(round(fe_reduction, digits=4))
    Max Δ: $(round(max_change, digits=6))
    Final Δ: $(round(final_change, digits=6))
    """
    
    annotate!(p, length(free_energy) * 0.95, maximum(free_energy) * 0.995,
              text(ann_text, 9, :right, font("Courier")))
    
    # Add horizontal line at final value for reference
    hline!(p, [final_fe],
           color=:gray,
           linestyle=:dash,
           linewidth=1,
           alpha=0.5,
           label="Final Value")
    
    return p
end

"""
    plot_data_histogram(data::Vector{Float64}; theta_real::Union{Float64, Nothing}=nothing,
                        theme::String="default")

Plot histogram of observed data.
"""
function plot_data_histogram(data::Vector{Float64}; 
                              theta_real::Union{Float64, Nothing}=nothing,
                              theme::String="default")
    colors = get_theme_colors(theme)
    
    n_heads = sum(data)
    n_tails = length(data) - n_heads
    empirical_rate = n_heads / length(data)
    
    # Create bar plot
    p = bar(
        ["Tails (0)", "Heads (1)"],
        [n_tails, n_heads],
        title = "Observed Data Distribution",
        xlabel = "Outcome",
        ylabel = "Count",
        legend = false,
        grid = true,
        gridcolor = colors.grid,
        background_color = colors.background,
        color = [colors.data, colors.data],
        alpha = 0.7,
        size = (800, 600)
    )
    
    # Add annotations
    annotate!(p, 1, n_tails + max(n_tails, n_heads) * 0.05,
              text("$n_tails ($(round((1-empirical_rate)*100, digits=1))%)", 10))
    annotate!(p, 2, n_heads + max(n_tails, n_heads) * 0.05,
              text("$n_heads ($(round(empirical_rate*100, digits=1))%)", 10))
    
    if theta_real !== nothing
        annotate!(p, 1.5, max(n_tails, n_heads) * 0.9,
                  text("True θ = $theta_real\nEmpirical = $(round(empirical_rate, digits=3))", 12, :center))
    end
    
    return p
end

"""
    plot_credible_interval(posterior::Beta; level::Float64=0.95, 
                           theta_real::Union{Float64, Nothing}=nothing,
                           resolution::Int=1000, theme::String="default")

Plot posterior with credible interval highlighted.
"""
function plot_credible_interval(posterior::Beta; level::Float64=0.95, 
                                 theta_real::Union{Float64, Nothing}=nothing,
                                 resolution::Int=1000, theme::String="default")
    colors = get_theme_colors(theme)
    
    # Compute credible interval
    lower_q = (1 - level) / 2
    upper_q = 1 - lower_q
    ci_lower = quantile(posterior, lower_q)
    ci_upper = quantile(posterior, upper_q)
    
    # Create range
    θ_range = range(0, 1, length=resolution)
    posterior_pdf = pdf.(posterior, θ_range)
    
    # Create plot
    p = plot(
        title = "Posterior with $(Int(level*100))% Credible Interval",
        xlabel = "θ (Coin Bias)",
        ylabel = "Probability Density",
        legend = :topright,
        grid = true,
        gridcolor = colors.grid,
        background_color = colors.background,
        size = (800, 600)
    )
    
    # Plot full posterior
    plot!(p, θ_range, posterior_pdf,
          label = "Posterior",
          color = colors.posterior,
          linewidth = 2,
          alpha = 0.5)
    
    # Highlight credible interval
    ci_mask = (θ_range .>= ci_lower) .& (θ_range .<= ci_upper)
    ci_theta = θ_range[ci_mask]
    ci_pdf = posterior_pdf[ci_mask]
    
    plot!(p, ci_theta, ci_pdf,
          fillrange = 0,
          fillalpha = 0.4,
          label = "$(Int(level*100))% CI: [$(round(ci_lower, digits=3)), $(round(ci_upper, digits=3))]",
          color = colors.posterior,
          linewidth = 2)
    
    # Plot true value if provided
    if theta_real !== nothing
        vline!(p, [theta_real],
               label = "True θ = $(round(theta_real, digits=3))",
               color = colors.true_value,
               linewidth = 3,
               linestyle = :dash)
        
        # Check if true value is in CI
        in_ci = ci_lower <= theta_real <= ci_upper
        annotate!(p, theta_real, maximum(posterior_pdf) * 0.5,
                  text(in_ci ? "✓ In CI" : "✗ Outside CI", 10, :left))
    end
    
    return p
end

"""
    plot_predictive(posterior::Beta, observed_data::Vector{Float64}; 
                    n_samples::Int=10000, theme::String="default")

Plot posterior predictive distribution vs observed data.
"""
function plot_predictive(posterior::Beta, observed_data::Vector{Float64}; 
                         n_samples::Int=10000, theme::String="default")
    colors = get_theme_colors(theme)
    
    # Generate posterior predictive samples
    rng = MersenneTwister(123)
    theta_samples = rand(rng, posterior, n_samples)
    predictive_samples = [rand(rng, Bernoulli(θ)) for θ in theta_samples]
    
    # Compute statistics
    pred_mean = mean(predictive_samples)
    obs_mean = mean(observed_data)
    
    # Create comparison plot
    p = bar(
        ["Observed", "Predicted"],
        [obs_mean, pred_mean],
        title = "Posterior Predictive Check",
        xlabel = "Dataset",
        ylabel = "Proportion of Heads",
        legend = false,
        grid = true,
        gridcolor = colors.grid,
        background_color = colors.background,
        color = [colors.data, colors.posterior],
        alpha = 0.7,
        ylim = (0, 1),
        size = (800, 600)
    )
    
    # Add error bars (standard error)
    obs_se = sqrt(obs_mean * (1 - obs_mean) / length(observed_data))
    pred_se = sqrt(pred_mean * (1 - pred_mean) / n_samples)
    
    plot!(p, [1, 2], [obs_mean, pred_mean],
          yerror = [obs_se, pred_se],
          seriestype = :scatter,
          color = :black,
          markersize = 0)
    
    # Annotations
    annotate!(p, 1, obs_mean + 0.1,
              text("$(round(obs_mean, digits=3)) ± $(round(obs_se, digits=3))", 10))
    annotate!(p, 2, pred_mean + 0.1,
              text("$(round(pred_mean, digits=3)) ± $(round(pred_se, digits=3))", 10))
    
    return p
end

"""
    plot_comprehensive_dashboard(prior::Beta, posterior::Beta, 
                                  data::Vector{Float64}, 
                                  free_energy::Union{Vector{Float64}, Nothing};
                                  theta_real::Union{Float64, Nothing}=nothing,
                                  theme::String="default")

Create comprehensive dashboard with multiple diagnostic plots.
"""
function plot_comprehensive_dashboard(prior::Beta, posterior::Beta, 
                                       data::Vector{Float64}, 
                                       free_energy::Union{Vector{Float64}, Nothing};
                                       theta_real::Union{Float64, Nothing}=nothing,
                                       theme::String="default")
    
    # Create individual plots
    p1 = plot_prior_posterior(prior, posterior; theta_real=theta_real, theme=theme)
    p2 = plot_data_histogram(data; theta_real=theta_real, theme=theme)
    p3 = plot_credible_interval(posterior; theta_real=theta_real, theme=theme)
    p4 = plot_predictive(posterior, data; theme=theme)
    
    # Create layout
    if free_energy !== nothing
        p5 = plot_convergence(free_energy; theme=theme)
        layout = @layout [a b; c d; e]
        dashboard = plot(p1, p2, p3, p4, p5,
                         layout = layout,
                         size = (1600, 1800))
    else
        layout = @layout [a b; c d]
        dashboard = plot(p1, p2, p3, p4,
                         layout = layout,
                         size = (1600, 1200))
    end
    
    return dashboard
end

"""
    create_inference_animation(data::Vector{Float64}, prior_a::Float64, prior_b::Float64,
                                sample_sizes::Vector{Int}; 
                                theta_real::Union{Float64, Nothing}=nothing,
                                fps::Int=10, theme::String="default")

Create animation showing how posterior updates with increasing data.
"""
function create_inference_animation(data::Vector{Float64}, prior_a::Float64, prior_b::Float64,
                                     sample_sizes::Vector{Int}; 
                                     theta_real::Union{Float64, Nothing}=nothing,
                                     fps::Int=10, theme::String="default")
    
    colors = get_theme_colors(theme)
    prior = Beta(prior_a, prior_b)
    θ_range = range(0, 1, length=1000)
    
    anim = @animate for n in sample_sizes
        # Get data subset
        data_subset = data[1:min(n, length(data))]
        
        # Compute posterior
        n_heads = sum(data_subset)
        n_tails = length(data_subset) - n_heads
        posterior = Beta(prior_a + n_heads, prior_b + n_tails)
        
        # Create plot
        p = plot(
            title = "Bayesian Update with n = $n observations",
            xlabel = "θ (Coin Bias)",
            ylabel = "Probability Density",
            legend = :topright,
            grid = true,
            gridcolor = colors.grid,
            background_color = colors.background,
            size = (800, 600),
            ylim = (0, maximum(pdf.(posterior, θ_range)) * 1.1)
        )
        
        # Plot prior
        plot!(p, θ_range, pdf.(prior, θ_range),
              fillrange = 0,
              fillalpha = 0.2,
              label = "Prior",
              color = colors.prior,
              linewidth = 2,
              linestyle = :dash)
        
        # Plot current posterior
        plot!(p, θ_range, pdf.(posterior, θ_range),
              fillrange = 0,
              fillalpha = 0.4,
              label = "Posterior (n=$n)",
              color = colors.posterior,
              linewidth = 3)
        
        # Plot true value
        if theta_real !== nothing
            vline!(p, [theta_real],
                   label = "True θ",
                   color = colors.true_value,
                   linewidth = 3,
                   linestyle = :dash)
        end
        
        # Add statistics
        post_mean = mean(posterior)
        post_std = std(posterior)
        empirical_rate = n_heads / length(data_subset)
        
        annotate!(p, 0.98, maximum(pdf.(posterior, θ_range)) * 1.05,
                  text("Observations: $n\nHeads: $n_heads ($(round(empirical_rate*100, digits=1))%)\n" *
                       "Posterior: μ=$(round(post_mean, digits=3)), σ=$(round(post_std, digits=3))",
                       10, :right))
    end
    
    return anim
end

"""
    save_plot(p, filepath::String)

Save plot to file, ensuring directory exists.
"""
function save_plot(p, filepath::String)
    dir = dirname(filepath)
    if !isdir(dir) && dir != ""
        mkpath(dir)
    end
    
    savefig(p, filepath)
    @info "Saved plot" filepath=filepath
end

"""
    plot_posterior_evolution(data::Vector{Float64}, prior_a::Float64, prior_b::Float64;
                             sample_increments::Vector{Int}=[10, 25, 50, 100, 200, 500],
                             credible_level::Float64=0.95,
                             theme::String="default")

Plot posterior mean and credible interval evolution as data accumulates.

Shows how the posterior distribution changes with increasing sample size.
"""
function plot_posterior_evolution(data::Vector{Float64}, prior_a::Float64, prior_b::Float64;
                                   sample_increments::Vector{Int}=[10, 25, 50, 100, 200, 500],
                                   credible_level::Float64=0.95,
                                   theme::String="default")
    colors = get_theme_colors(theme)
    
    # Filter increments to valid sample sizes
    valid_increments = filter(n -> n <= length(data), sample_increments)
    
    # Calculate posterior statistics for each increment
    means = Float64[]
    ci_lower = Float64[]
    ci_upper = Float64[]
    
    for n in valid_increments
        # Compute posterior with first n observations
        n_heads = sum(data[1:n])
        n_tails = n - n_heads
        posterior = Beta(prior_a + n_heads, prior_b + n_tails)
        
        # Get statistics
        push!(means, mean(posterior))
        
        # Calculate credible interval
        lower_q = (1 - credible_level) / 2
        upper_q = 1 - lower_q
        push!(ci_lower, quantile(posterior, lower_q))
        push!(ci_upper, quantile(posterior, upper_q))
    end
    
    # Create plot
    p = plot(valid_increments, means,
        label="Posterior Mean",
        color=colors[:posterior],
        linewidth=3,
        xlabel="Number of Observations",
        ylabel="θ (Coin Bias)",
        title="Posterior Evolution Over Time\n$(Int(credible_level*100))% Credible Interval",
        legend=:topright,
        grid=true,
        background_color=colors[:background],
        gridcolor=colors[:grid],
        gridlinewidth=1,
        gridalpha=0.3,
        size=(800, 500),
        margin=5Plots.mm
    )
    
    # Add credible interval ribbon
    plot!(p, valid_increments, means,
        ribbon=(means .- ci_lower, ci_upper .- means),
        fillalpha=0.3,
        fillcolor=colors[:posterior],
        linewidth=0,
        label="$(Int(credible_level*100))% CI"
    )
    
    # Add individual points
    scatter!(p, valid_increments, means,
        color=colors[:posterior],
        markersize=6,
        label=nothing
    )
    
    # Add prior mean as reference line
    prior_mean = prior_a / (prior_a + prior_b)
    hline!(p, [prior_mean],
        color=colors[:prior],
        linestyle=:dash,
        linewidth=2,
        label="Prior Mean"
    )
    
    return p
end

end # module CoinTossVisualization

