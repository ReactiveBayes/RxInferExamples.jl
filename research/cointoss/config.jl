# Configuration module for Coin Toss Model research fork
# Centralizes all configurable parameters with validation

module Config

using TOML
using Dates

export load_config, validate_config, get_config, get_default_config, CONFIG

"""
    load_config(config_file::String = "config.toml")

Load configuration from TOML file and set up global CONFIG constant.
"""
function load_config(config_file::String = "config.toml")
    if !isfile(config_file)
        @warn "Config file not found: $config_file. Using defaults."
        return get_default_config()
    end
    
    config = TOML.parsefile(config_file)
    return config
end

"""
    get_default_config()

Return default configuration as a Dict.
"""
function get_default_config()
    return Dict(
        "data" => Dict(
            "n_samples" => 500,
            "theta_real" => 0.75,
            "seed" => 42,
            "generate_new" => true,
            "data_file" => "outputs/data/coin_toss_data.csv"
        ),
        "model" => Dict(
            "prior_a" => 4.0,
            "prior_b" => 8.0
        ),
        "inference" => Dict(
            "iterations" => 10,
            "free_energy_tracking" => true,
            "convergence_check" => true,
            "convergence_tolerance" => 1e-6,
            "showprogress" => true
        ),
        "analysis" => Dict(
            "credible_interval" => 0.95,
            "n_posterior_samples" => 10000,
            "compute_bayes_factor" => false,
            "diagnostic_plots" => true,
            "posterior_predictive" => true
        ),
        "visualization" => Dict(
            "theme" => "default",
            "plot_resolution" => 1000,
            "figure_size" => [800, 600],
            "dpi" => 100,
            "animation_fps" => 10,
            "show_plots" => false,
            "save_plots" => true
        ),
        "animation" => Dict(
            "enabled" => true,
            "sample_increments" => [10, 25, 50, 100, 200, 500],
            "fps" => 10
        ),
        "output" => Dict(
            "output_dir" => "outputs",
            "data_dir" => "outputs/data",
            "plots_dir" => "outputs/plots",
            "animations_dir" => "outputs/animations",
            "results_dir" => "outputs/results",
            "logs_dir" => "outputs/logs"
        ),
        "logging" => Dict(
            "verbose" => true,
            "structured" => true,
            "performance" => true,
            "log_level" => "info",
            "log_to_file" => true,
            "log_file" => "outputs/logs/cointoss.log",
            "structured_log_file" => "outputs/logs/cointoss_structured.jsonl",
            "performance_log_file" => "outputs/logs/cointoss_performance.csv"
        ),
        "export" => Dict(
            "formats" => ["csv", "json"],
            "export_posterior" => true,
            "export_diagnostics" => true,
            "export_predictions" => true,
            "comprehensive" => true
        ),
        "benchmark" => Dict(
            "enabled" => false,
            "iterations" => 100,
            "track_memory" => true
        )
    )
end

"""
    validate_config(config::Dict)

Validate configuration parameters and return list of issues.
"""
function validate_config(config::Dict)
    issues = String[]
    
    # Validate data parameters
    if haskey(config, "data")
        data = config["data"]
        if haskey(data, "n_samples") && data["n_samples"] <= 0
            push!(issues, "data.n_samples must be positive")
        end
        if haskey(data, "theta_real") && (data["theta_real"] < 0 || data["theta_real"] > 1)
            push!(issues, "data.theta_real must be in [0, 1]")
        end
    end
    
    # Validate model parameters
    if haskey(config, "model")
        model = config["model"]
        if haskey(model, "prior_a") && model["prior_a"] <= 0
            push!(issues, "model.prior_a must be positive")
        end
        if haskey(model, "prior_b") && model["prior_b"] <= 0
            push!(issues, "model.prior_b must be positive")
        end
    end
    
    # Validate inference parameters
    if haskey(config, "inference")
        inf = config["inference"]
        if haskey(inf, "iterations") && inf["iterations"] <= 0
            push!(issues, "inference.iterations must be positive")
        end
        if haskey(inf, "convergence_tolerance") && inf["convergence_tolerance"] <= 0
            push!(issues, "inference.convergence_tolerance must be positive")
        end
    end
    
    # Validate analysis parameters
    if haskey(config, "analysis")
        analysis = config["analysis"]
        if haskey(analysis, "credible_interval") && (analysis["credible_interval"] <= 0 || analysis["credible_interval"] >= 1)
            push!(issues, "analysis.credible_interval must be in (0, 1)")
        end
        if haskey(analysis, "n_posterior_samples") && analysis["n_posterior_samples"] <= 0
            push!(issues, "analysis.n_posterior_samples must be positive")
        end
    end
    
    return issues
end

"""
    get_config()

Return the current global configuration (load if not already loaded).
"""
function get_config()
    if !@isdefined(CONFIG)
        return load_config()
    end
    return CONFIG
end

end # module Config

