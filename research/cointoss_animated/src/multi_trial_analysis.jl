module MultiTrialAnalysis

using DataFrames
using Statistics
using Plots
using JSON
using CSV
using StatsBase
using Distributions

export MultiTrialConfig, load_trial_results, analyze_trials, create_comparison_plots, create_performance_dashboard

"""
Configuration for multi-trial analysis
"""
struct MultiTrialConfig
    results_dir::String
    output_dir::String
    comparison_metrics::Vector{String}
    group_by::Vector{String}
    significance_level::Float64
end

"""
Load trial results from JSON files
"""
function load_trial_results(results_dir::String)
    results = []

    for file in readdir(results_dir)
        if endswith(file, ".json") && startswith(file, "sweep_trial_")
            filepath = joinpath(results_dir, file)
            try
                result = JSON.parsefile(filepath)
                push!(results, result)
            catch e
                @warn "Failed to load $filepath: $e"
            end
        end
    end

    return results
end

"""
Analyze multiple trials and compute summary statistics
"""
function analyze_trials(results::Vector{Any}, config::MultiTrialConfig)
    # Filter successful trials
    successful_results = filter(r -> !haskey(r, "error"), results)

    if length(successful_results) == 0
        @warn "No successful trials found"
        return Dict()
    end

    # Convert to DataFrame for analysis
    df = create_trial_dataframe(successful_results)

    # Group by specified parameters
    grouped_stats = Dict()
    for group in config.group_by
        grouped_stats[group] = groupby(df, group)
    end

    # Compute overall statistics
    overall_stats = Dict(
        "n_trials" => length(successful_results),
        "mean_posterior" => mean(df.posterior_mean),
        "std_posterior" => std(df.posterior_mean),
        "mean_execution_time" => mean(df.execution_time),
        "convergence_rate" => mean(df.converged),
        "parameter_effects" => compute_parameter_effects(df, config.comparison_metrics)
    )

    # Compute correlations
    correlations = compute_correlations(df, config.comparison_metrics)

    # Save analysis results
    analysis_results = Dict(
        "overall_stats" => overall_stats,
        "grouped_stats" => grouped_stats,
        "correlations" => correlations,
        "config" => Dict(
            "comparison_metrics" => config.comparison_metrics,
            "group_by" => config.group_by,
            "significance_level" => config.significance_level
        )
    )

    output_file = joinpath(config.output_dir, "multi_trial_analysis.json")
    open(output_file, "w") do f
        JSON.print(f, analysis_results, 2)
    end

    return analysis_results
end

"""
Create DataFrame from trial results
"""
function create_trial_dataframe(results::Vector{Any})
    df_data = []

    for result in results
        if !haskey(result, "error")
            push!(df_data, Dict(
                "n_samples" => result["data"]["n_samples"],
                "theta_real" => result["data"]["theta_real"],
                "prior_a" => result["params"]["prior_a"],
                "prior_b" => result["params"]["prior_b"],
                "n_heads" => result["data"]["n_heads"],
                "empirical_rate" => result["data"]["empirical_rate"],
                "posterior_mean" => result["inference"]["posterior_mean"],
                "posterior_std" => result["inference"]["posterior_std"],
                "converged" => result["inference"]["converged"] ? 1 : 0,
                "iterations" => result["inference"]["iterations"],
                "execution_time" => result["inference"]["execution_time"]
            ))
        end
    end

    return DataFrame(df_data)
end

"""
Compute parameter effects using ANOVA-like analysis
"""
function compute_parameter_effects(df::DataFrame, metrics::Vector{String})
    effects = Dict()

    for metric in metrics
        if metric in names(df) && metric != "posterior_mean"
            # Simple effect size calculation
            grouped = groupby(df, :theta_real)
            means = combine(grouped, metric => mean)
            effects[metric] = Dict(
                "effect_size" => std(means[!, Symbol(metric * "_mean")]),
                "range" => maximum(means[!, Symbol(metric * "_mean")]) - minimum(means[!, Symbol(metric * "_mean")])
            )
        end
    end

    return effects
end

"""
Compute correlations between metrics
"""
function compute_correlations(df::DataFrame, metrics::Vector{String})
    correlations = Dict()

    available_metrics = filter(m -> m in names(df), metrics)

    for i in 1:length(available_metrics)
        for j in i+1:length(available_metrics)
            metric1 = available_metrics[i]
            metric2 = available_metrics[j]

            corr_val = cor(df[!, Symbol(metric1)], df[!, Symbol(metric2)])
            correlations["$(metric1)_vs_$(metric2)"] = corr_val
        end
    end

    return correlations
end

"""
Create comparison plots for multi-trial analysis
"""
function create_comparison_plots(df::DataFrame, config::MultiTrialConfig)
    plots = Dict()

    # Posterior mean vs theta_real
    p1 = scatter(df.theta_real, df.posterior_mean,
                xlabel="True θ", ylabel="Posterior Mean",
                title="Posterior Mean vs True θ",
                legend=false, alpha=0.6)
    plot!(p1, df.theta_real, df.theta_real, ls=:dash, color=:red, label="Perfect Recovery")
    plots["posterior_vs_theta"] = p1

    # Execution time vs sample size
    p2 = scatter(df.n_samples, df.execution_time,
                xlabel="Sample Size", ylabel="Execution Time (s)",
                title="Performance Scaling",
                legend=false, alpha=0.6)
    plots["time_vs_samples"] = p2

    # Convergence rate vs prior strength
    p3 = scatter(df.prior_a, df.converged,
                xlabel="Prior Strength (α)", ylabel="Convergence Rate",
                title="Convergence vs Prior Strength",
                legend=false, alpha=0.6)
    plots["convergence_vs_prior"] = p3

    # Empirical rate vs posterior mean
    p4 = scatter(df.empirical_rate, df.posterior_mean,
                xlabel="Empirical Rate", ylabel="Posterior Mean",
                title="Empirical vs Posterior",
                legend=false, alpha=0.6)
    plots["empirical_vs_posterior"] = p4

    # Save plots
    for (name, plot) in plots
        filepath = joinpath(config.output_dir, "$(name).png")
        savefig(plot, filepath)
    end

    return plots
end

"""
Create comprehensive performance dashboard
"""
function create_performance_dashboard(df::DataFrame, config::MultiTrialConfig)
    # Create subplots
    p1 = histogram(df.posterior_mean, bins=20, xlabel="Posterior Mean", ylabel="Frequency",
                   title="Distribution of Posterior Means", legend=false)

    p2 = scatter(df.theta_real, df.posterior_mean, xlabel="True θ", ylabel="Posterior Mean",
                 title="Recovery Accuracy", legend=false, alpha=0.6)

    p3 = scatter(df.n_samples, df.execution_time, xlabel="Sample Size", ylabel="Time (s)",
                 title="Performance Scaling", legend=false, alpha=0.6)

    p4 = scatter(df.prior_a, df.posterior_std, xlabel="Prior α", ylabel="Posterior Std",
                 title="Uncertainty vs Prior", legend=false, alpha=0.6)

    # Combine into dashboard
    dashboard = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800))

    # Save dashboard
    filepath = joinpath(config.output_dir, "multi_trial_dashboard.png")
    savefig(dashboard, filepath)

    return dashboard
end

end # module
