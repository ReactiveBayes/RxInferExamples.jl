#!/usr/bin/env julia

"""
Enhanced Coin Toss Experiment with Comprehensive RxInfer Diagnostics

This script runs the Bayesian coin toss inference with all available RxInfer
diagnostic features enabled:

1. Memory Addon - traces message computations
2. Inference Callbacks - tracks iteration progress and marginal updates
3. Logger Pipeline Stage - traces message passing (optional, very verbose)
4. Benchmark Callbacks - performance analysis with statistics

Usage:
    julia --project=. run_with_diagnostics.jl [--skip-animation] [--config=path/to/config.toml]
"""

using Pkg
Pkg.activate(".")

using Logging
using Dates
using Statistics
using Printf
using DataFrames
using CSV
using Plots
using Distributions
using PrettyTables

# Load configuration
include("config.jl")
using .Config

# Load all modules
include("src/model.jl")
include("src/inference.jl")
include("src/visualization.jl")
include("src/utils.jl")
include("src/diagnostics.jl")
include("src/timeseries_diagnostics.jl")
include("src/graphical_abstract.jl")

using .CoinTossModel
using .CoinTossInference
using .CoinTossVisualization
using .CoinTossUtils
using .CoinTossDiagnostics
using .TimeseriesDiagnostics
using .GraphicalAbstract

# Global variable to hold config (set before model definition)
GLOBAL_CONFIG = Dict{String, Any}()

# Parse command line arguments
function parse_args()
    args = Dict{String, Any}(
        "skip_animation" => false,
        "config_file" => "config.toml"
    )
    
    for arg in ARGS
        if arg == "--skip-animation"
            args["skip_animation"] = true
        elseif startswith(arg, "--config=")
            args["config_file"] = split(arg, "=")[2]
        end
    end
    
    return args
end

"""
Main experiment runner with diagnostics
"""
function run_experiment_with_diagnostics(config::Dict{String, Any}; skip_animation::Bool=false)
    # Print header
    @info "=" ^ 80
    @info "Coin Toss Model - Bayesian Inference with Advanced Diagnostics"
    @info "=" ^ 80
    
    # Ensure output directories exist
    CoinTossUtils.ensure_directories(config)
    diagnostics_dir = joinpath("outputs", "diagnostics")
    mkpath(diagnostics_dir)
    
    # Get directory paths from config
    data_dir = config["output"]["data_dir"]
    plots_dir = config["output"]["plots_dir"]
    animations_dir = config["output"]["animations_dir"]
    results_dir = config["output"]["results_dir"]
    logs_dir = config["output"]["logs_dir"]
    
    # Setup logging
    CoinTossUtils.setup_logging(
        verbose = config["logging"]["verbose"],
        structured = config["logging"]["structured"],
        performance = config["logging"]["performance"],
        log_file = config["logging"]["log_to_file"] ? config["logging"]["log_file"] : nothing
    )
    
    # Print configuration summary
    @info "Configuration Summary:"
    @info "  Data: n=$(config["data"]["n_samples"]), θ=$(config["data"]["theta_real"]), seed=$(config["data"]["seed"])"
    @info "  Prior: Beta($(config["model"]["prior_a"]), $(config["model"]["prior_b"]))"
    @info "  Inference: $(config["inference"]["iterations"]) iterations"
    @info "  Diagnostics: Memory=$(config["diagnostics"]["enable_memory_addon"]), " *
          "Callbacks=$(config["diagnostics"]["enable_callbacks"]), " *
          "Benchmark=$(config["diagnostics"]["enable_benchmark"])"
    @info "  Theme: $(config["visualization"]["theme"])"
    
    # Initialize experiment results container
    experiment_results = Dict{String, Any}(
        "metadata" => Dict(
            "timestamp" => string(now()),
            "config_file" => "config.toml",
            "julia_version" => string(VERSION),
            "diagnostics_enabled" => true
        ),
        "config" => config,
        "results" => Dict{String, Any}()
    )
    
    # ============================================================================
    # Step 1: Data Generation
    # ============================================================================
    @info ""
    @info "=" ^ 80
    @info "Step 1: Data Generation"
    @info "=" ^ 80
    
    data_timer = CoinTossUtils.Timer("data_generation")
    
    coin_data = generate_coin_data(
        n = config["data"]["n_samples"],
        theta_real = config["data"]["theta_real"],
        seed = config["data"]["seed"]
    )
    
    CoinTossUtils.elapsed_time(data_timer)
    
    n_heads = sum(coin_data.observations)
    n_tails = length(coin_data.observations) - n_heads
    
    @info "Generated $(length(coin_data.observations)) coin tosses"
    @info "  True θ: $(config["data"]["theta_real"])"
    @info "  Observed heads: $n_heads ($(round(n_heads/length(coin_data.observations)*100, digits=1))%)"
    
    # Save observations
    obs_df = DataFrame(
        observation_index = 1:length(coin_data.observations),
        value = coin_data.observations
    )
    obs_path = joinpath(data_dir, "coin_toss_observations.csv")
    CSV.write(obs_path, obs_df)
    @info "Saved observations to: $obs_path"
    
    experiment_results["results"]["data"] = Dict(
        "n_observations" => length(coin_data.observations),
        "true_theta" => coin_data.theta_real,
        "n_heads" => n_heads,
        "n_tails" => n_tails,
        "empirical_rate" => n_heads / length(coin_data.observations)
    )
    
    # ============================================================================
    # Step 2: Bayesian Inference with Diagnostics
    # ============================================================================
    @info ""
    @info "=" ^ 80
    @info "Step 2: Bayesian Inference with Advanced Diagnostics"
    @info "=" ^ 80
    
    inference_timer = CoinTossUtils.Timer("bayesian_inference_with_diagnostics")
    
    # Create diagnostic configuration
    diagnostic_config = CoinTossDiagnostics.DiagnosticConfig(
        enable_memory_addon = config["diagnostics"]["enable_memory_addon"],
        enable_callbacks = config["diagnostics"]["enable_callbacks"],
        enable_pipeline_logger = config["diagnostics"]["enable_pipeline_logger"],
        enable_benchmark = config["diagnostics"]["enable_benchmark"],
        verbose = config["diagnostics"]["verbose"]
    )
    
    @info "Running inference with diagnostics..."
    @info "  Observations: $(length(coin_data.observations))"
    @info "  Prior: Beta($(config["model"]["prior_a"]), $(config["model"]["prior_b"]))"
    @info "  Iterations: $(config["inference"]["iterations"])"
    @info "  Benchmark runs: $(config["diagnostics"]["n_benchmark_runs"])"
    
    # Create model with prior parameters
    model = coin_model(
        a = config["model"]["prior_a"],
        b = config["model"]["prior_b"]
    )
    
    # Run inference with diagnostics
    result, diagnostics = run_inference_with_diagnostics(
        model,
        coin_data.observations,
        config["model"]["prior_a"],
        config["model"]["prior_b"];
        config = diagnostic_config,
        iterations = config["inference"]["iterations"],
        n_benchmark_runs = config["diagnostics"]["n_benchmark_runs"]
    )
    
    CoinTossUtils.elapsed_time(inference_timer)
    
    # Extract posterior
    posterior_raw = result.posteriors[:θ]
    posterior_marginal = posterior_raw isa Vector ? posterior_raw[end] : posterior_raw
    # Extract the actual distribution from the Marginal wrapper
    α, β = params(posterior_marginal)
    posterior = Beta(α, β)
    
    @info "Inference completed"
    @info "  Posterior: $(typeof(posterior))"
    @info "  Parameters: $(params(posterior))"
    
    # ============================================================================
    # Step 3: Save Diagnostic Results
    # ============================================================================
    @info ""
    @info "=" ^ 80
    @info "Step 3: Processing Diagnostic Results"
    @info "=" ^ 80
    
    if config["diagnostics"]["save_diagnostics"]
        save_diagnostics(diagnostics, diagnostics_dir)
        visualize_message_trace(diagnostics, diagnostics_dir)
    end
    
    # Display benchmark statistics
    if diagnostics.benchmark_stats !== nothing
        @info ""
        @info "Performance Benchmark Statistics ($(config["diagnostics"]["n_benchmark_runs"]) runs):"
        pretty_table(
            diagnostics.benchmark_stats,
            show_subheader = false,
            formatters = ft_printf("%.2f", 2:7)
        )
    end
    
    # Display memory trace summary
    if diagnostics.memory_trace !== nothing
        @info ""
        @info "Message Trace Summary:"
        @info "  Variable: $(diagnostics.memory_trace["variable"])"
        @info "  Posterior Type: $(diagnostics.memory_trace["posterior_type"])"
        @info "  Posterior Parameters: $(diagnostics.memory_trace["posterior_params"])"
        if haskey(diagnostics.memory_trace, "full_trace")
            @info "  Full trace saved to: $diagnostics_dir/message_trace_report.txt"
        end
    end
    
    # Display callback trace summary
    if diagnostics.callback_trace !== nothing
        @info ""
        @info "Callback Trace Summary:"
        @info "  Total events: $(length(diagnostics.callback_trace))"
        
        # Count marginal updates
        marginal_updates = filter(x -> haskey(x, "variable"), diagnostics.callback_trace)
        @info "  Marginal updates: $(length(marginal_updates))"
        
        if !isempty(marginal_updates)
            @info "  Updated variables: $(unique([x["variable"] for x in marginal_updates]))"
        end
    end
    
    # ============================================================================
    # Step 4: Statistical Analysis
    # ============================================================================
    @info ""
    @info "=" ^ 80
    @info "Step 4: Statistical Analysis"
    @info "=" ^ 80
    
    analysis_timer = CoinTossUtils.Timer("statistical_analysis")
    
    # Compute posterior statistics
    stats = posterior_statistics(posterior)
    
    @info "Posterior Statistics:"
    @info "  Mean: $(round(stats["mean"], digits=4))"
    @info "  Mode: $(round(stats["mode"], digits=4))"
    @info "  Std: $(round(stats["std"], digits=4))"
    ci_lower, ci_upper = stats["credible_interval"]
    @info "  95% CI: [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]"
    
    # Compare with analytical solution
    analytical_post = analytical_posterior(coin_data.observations, 
                                          config["model"]["prior_a"], 
                                          config["model"]["prior_b"])
    a_analytical, b_analytical = params(analytical_post)
    a_rxinfer, b_rxinfer = params(posterior)
    
    @info "Analytical vs RxInfer comparison:"
    @info "  Analytical: α=$a_analytical, β=$b_analytical"
    @info "  RxInfer: α=$a_rxinfer, β=$b_rxinfer"
    
    # Log marginal likelihood
    lml = log_marginal_likelihood(coin_data.observations, 
                                  config["model"]["prior_a"], 
                                  config["model"]["prior_b"])
    @info "Log marginal likelihood: $(round(lml, digits=4))"
    
    CoinTossUtils.elapsed_time(analysis_timer)
    
    experiment_results["results"]["inference"] = Dict(
        "posterior_mean" => stats["mean"],
        "posterior_mode" => stats["mode"],
        "posterior_std" => stats["std"],
        "credible_interval" => collect(stats["credible_interval"]),
        "log_marginal_likelihood" => lml,
        "analytical_alpha" => a_analytical,
        "analytical_beta" => b_analytical,
        "rxinfer_alpha" => a_rxinfer,
        "rxinfer_beta" => b_rxinfer
    )
    
    # ============================================================================
    # Step 5: Visualization
    # ============================================================================
    @info ""
    @info "=" ^ 80
    @info "Step 5: Visualization"
    @info "=" ^ 80
    
    viz_timer = CoinTossUtils.Timer("visualization")
    
    theme = config["visualization"]["theme"]
    prior = Beta(config["model"]["prior_a"], config["model"]["prior_b"])
    
    # Create comprehensive dashboard
    @info "Creating comprehensive dashboard..."
    dashboard = plot_comprehensive_dashboard(
        prior, posterior, coin_data.observations,
        result.free_energy isa Vector ? Float64.(result.free_energy) : nothing;
        theta_real = coin_data.theta_real,
        theme = theme
    )
    save_plot(dashboard, joinpath(plots_dir, "comprehensive_dashboard.png"))
    
    # Create individual plots
    @info "Creating individual diagnostic plots..."
    
    p1 = plot_prior_posterior(prior, posterior; 
                              theta_real = coin_data.theta_real, 
                              theme = theme)
    save_plot(p1, joinpath(plots_dir, "prior_posterior.png"))
    
    p2 = plot_credible_interval(posterior; 
                                theta_real = coin_data.theta_real,
                                theme = theme)
    save_plot(p2, joinpath(plots_dir, "credible_interval.png"))
    
    p3 = plot_data_histogram(coin_data.observations; theme = theme)
    save_plot(p3, joinpath(plots_dir, "data_histogram.png"))
    
    p4 = plot_predictive(posterior, coin_data.observations; theme = theme)
    save_plot(p4, joinpath(plots_dir, "posterior_predictive.png"))
    
    if result.free_energy !== nothing
        p5 = plot_convergence(result.free_energy isa Vector ? Float64.(result.free_energy) : Float64[], 
                             theme = theme)
        save_plot(p5, joinpath(plots_dir, "free_energy_convergence.png"))
    end
    
    # Posterior evolution timeseries
    @info "Creating posterior evolution timeseries..."
    evolution_plot = plot_posterior_evolution(
        coin_data.observations,
        config["model"]["prior_a"],
        config["model"]["prior_b"];
        sample_increments = config["animation"]["sample_increments"],
        credible_level = config["analysis"]["credible_interval"],
        theme = theme
    )
    save_plot(evolution_plot, joinpath(plots_dir, "posterior_evolution.png"))
    
    # Comprehensive timeseries dashboard (ALL metrics through time)
    @info "Creating comprehensive timeseries dashboard (ALL metrics)..."
    timeseries_dir = joinpath("outputs", "timeseries")
    mkpath(timeseries_dir)
    
    comprehensive_dashboard, temporal_evolution = create_comprehensive_timeseries_dashboard(
        coin_data.observations,
        config["model"]["prior_a"],
        config["model"]["prior_b"];
        true_theta = coin_data.theta_real,
        theme = theme
    )
    save_plot(comprehensive_dashboard, joinpath(plots_dir, "comprehensive_timeseries_dashboard.png"))
    @info "Saved comprehensive timeseries dashboard (12 metrics)"
    
    # Individual timeseries plots for each metric
    @info "Creating individual timeseries plots for all metrics..."
    individual_plots, evolution_data = plot_all_metrics_through_time(
        coin_data.observations,
        config["model"]["prior_a"],
        config["model"]["prior_b"];
        true_theta = coin_data.theta_real,
        output_dir = timeseries_dir
    )
    @info "Saved $(length(individual_plots)) individual timeseries plots"
    
    # Export temporal evolution data
    evolution_df = DataFrame(evolution_data)
    evolution_csv_path = joinpath(timeseries_dir, "temporal_evolution_data.csv")
    CSV.write(evolution_csv_path, evolution_df)
    @info "Exported temporal evolution data to CSV" filepath=evolution_csv_path
    
    # Create Comprehensive Graphical Abstract (mega visualization)
    @info "Creating comprehensive graphical abstract (24-panel mega visualization)..."
    graphical_abstract = create_graphical_abstract(
        coin_data.observations,
        config["model"]["prior_a"],
        config["model"]["prior_b"],
        result,
        diagnostics,
        evolution_data;
        true_theta = coin_data.theta_real,
        benchmark_stats = diagnostics.benchmark_stats,
        theme = theme
    )
    save_plot(graphical_abstract, joinpath(plots_dir, "graphical_abstract.png"))
    @info "Saved comprehensive graphical abstract (2400×3600 px, 24 panels)"
    
    CoinTossUtils.elapsed_time(viz_timer)
    
    # ============================================================================
    # Step 6: Animation (Optional)
    # ============================================================================
    if !skip_animation
        @info ""
        @info "=" ^ 80
        @info "Step 6: Animation Generation"
        @info "=" ^ 80
        
        anim_timer = CoinTossUtils.Timer("animation_generation")
        
        @info "Creating inference animation..."
        animation = create_inference_animation(
            coin_data.observations,
            config["model"]["prior_a"],
            config["model"]["prior_b"];
            sample_increments = config["animation"]["sample_increments"],
            fps = config["animation"]["fps"],
            theme = theme
        )
        
        anim_path = joinpath(animations_dir, "bayesian_update.gif")
        @info "Animation saved to: $anim_path"
        
        CoinTossUtils.elapsed_time(anim_timer)
    end
    
    # ============================================================================
    # Step 7: Export Results
    # ============================================================================
    @info ""
    @info "=" ^ 80
    @info "Step 7: Export Results"
    @info "=" ^ 80
    
    export_timer = CoinTossUtils.Timer("results_export")
    
    # Save experiment results (save_experiment_results handles the timestamped directory)
    exp_results_dir = save_experiment_results(
        "coin_toss_diagnostic",
        experiment_results
    )
    
    @info "Results exported to: $exp_results_dir"
    
    CoinTossUtils.elapsed_time(export_timer)
    
    # ============================================================================
    # Final Summary
    # ============================================================================
    @info ""
    @info "=" ^ 80
    @info "EXPERIMENT SUMMARY"
    @info "=" ^ 80
    @info "Data: $(length(coin_data.observations)) observations, θ_true=$(coin_data.theta_real)"
    @info "Prior: Beta($(config["model"]["prior_a"]), $(config["model"]["prior_b"]))"
    @info "Posterior: β̂=$(round(stats["mean"], digits=4)) ± $(round(stats["std"], digits=4))"
    @info "95% CI: [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]"
    @info "True θ in credible interval: $(ci_lower <= coin_data.theta_real <= ci_upper)"
    @info ""
    @info "Outputs:"
    @info "  Plots: $plots_dir"
    @info "  Timeseries: $timeseries_dir (15+ plots + evolution data CSV)"
    @info "  Animations: $animations_dir"
    @info "  Results: $exp_results_dir"
    @info "  Diagnostics: $diagnostics_dir"
    @info "  Logs: $logs_dir"
    @info ""
    @info "=" ^ 80
    @info "Experiment completed successfully!"
    @info "=" ^ 80
    
    return experiment_results
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_args()
    
    try
        # Load configuration
        config = load_config(args["config_file"])
        
        # Validate configuration
        issues = validate_config(config)
        if !isempty(issues)
            @error "Configuration validation failed" issues
            exit(1)
        end
        
        # Run experiment
        results = run_experiment_with_diagnostics(
            config,
            skip_animation = args["skip_animation"]
        )
        
        exit(0)
    catch e
        @error "Experiment failed" exception=(e, catch_backtrace())
        exit(1)
    end
end

