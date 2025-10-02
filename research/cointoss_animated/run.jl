#!/usr/bin/env julia

# Main runner for Coin Toss Model research fork
# Comprehensive Bayesian inference experiment with full diagnostics

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

# Load configuration
include("config.jl")
using .Config

# Load all modules
include("src/model.jl")
include("src/inference.jl")
include("src/visualization.jl")
include("src/utils.jl")

using .CoinTossModel
using .CoinTossInference
using .CoinTossVisualization
using .CoinTossUtils

"""
    parse_args(args::Vector{String})

Parse command line arguments and merge with config.
"""
function parse_args(args::Vector{String})
    # Load base configuration
    config = load_config()
    
    # Override with command line arguments
    for arg in args
        if arg == "--help" || arg == "-h"
            return nothing, true
        elseif arg == "--verbose"
            config["logging"]["verbose"] = true
        elseif arg == "--quiet"
            config["logging"]["verbose"] = false
        elseif arg == "--no-animation"
            config["animation"]["enabled"] = false
        elseif arg == "--benchmark"
            config["benchmark"]["enabled"] = true
        elseif startswith(arg, "--n=")
            config["data"]["n_samples"] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--theta=")
            config["data"]["theta_real"] = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--seed=")
            config["data"]["seed"] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--theme=")
            config["visualization"]["theme"] = split(arg, "=")[2]
        end
    end
    
    return config, false
end

"""
    show_help()

Display help message.
"""
function show_help()
    println("""
    Coin Toss Model - Comprehensive Bayesian Inference Research Fork
    
    Usage: julia run.jl [options]
    
    Options:
      --help, -h          Show this help message
      --verbose           Enable detailed logging
      --quiet             Minimize logging output
      --no-animation      Disable animation generation
      --benchmark         Run performance benchmarks
      --n=N               Number of coin tosses (default: 500)
      --theta=θ           True coin bias (default: 0.75)
      --seed=S            Random seed (default: 42)
      --theme=THEME       Visualization theme: default, dark, colorblind
    
    Examples:
      julia run.jl                          # Run with default settings
      julia run.jl --verbose                # Detailed logging
      julia run.jl --n=1000 --theta=0.6    # Custom parameters
      julia run.jl --theme=dark             # Dark theme visualizations
      julia run.jl --benchmark              # Performance analysis
    
    Output:
      - Comprehensive plots in outputs/plots/
      - Animations in outputs/animations/
      - Data exports in outputs/data/
      - Results in outputs/results/
      - Logs in outputs/logs/
    
    Configuration:
      Edit config.toml to customize all parameters including:
      - Data generation parameters
      - Prior distributions
      - Inference settings
      - Visualization options
      - Export formats
    """)
end

"""
    run_experiment(config::Dict)

Run the complete coin toss inference experiment.
"""
function run_experiment(config::Dict)
    @info "="^80
    @info "Coin Toss Model - Bayesian Inference Experiment"
    @info "="^80
    
    # Validate configuration
    issues = validate_config(config)
    if !isempty(issues)
        @error "Configuration validation failed" issues=issues
        return
    end
    
    # Ensure all output directories exist
    ensure_directories(config)
    
    # Setup logging
    setup_logging(
        verbose = config["logging"]["verbose"],
        structured = config["logging"]["structured"],
        performance = config["logging"]["performance"],
        log_file = config["logging"]["log_to_file"] ? config["logging"]["log_file"] : nothing
    )
    
    # Log configuration summary
    @info "Configuration Summary:"
    @info "  Data: n=$(config["data"]["n_samples"]), θ=$(config["data"]["theta_real"]), seed=$(config["data"]["seed"])"
    @info "  Prior: Beta($(config["model"]["prior_a"]), $(config["model"]["prior_b"]))"
    @info "  Inference: $(config["inference"]["iterations"]) iterations"
    @info "  Theme: $(config["visualization"]["theme"])"
    
    # Initialize results dictionary
    experiment_results = Dict(
        "experiment_name" => "coin_toss_bayesian_inference",
        "timestamp" => string(now()),
        "config" => config,
        "results" => Dict{String, Any}()
    )
    
    #==========================================================================
    1. DATA GENERATION
    ==========================================================================#
    @info "\n" * "="^80
    @info "Step 1: Data Generation"
    @info "="^80
    
    data_timer = CoinTossUtils.Timer("data_generation")
    
    coin_data = generate_coin_data(
        n = config["data"]["n_samples"],
        theta_real = config["data"]["theta_real"],
        seed = config["data"]["seed"]
    )
    
    close(data_timer)
    
    @info "Generated $(length(coin_data.observations)) coin tosses"
    @info "  True θ: $(coin_data.theta_real)"
    @info "  Observed heads: $(sum(coin_data.observations)) ($(round(mean(coin_data.observations)*100, digits=1))%)"
    
    # Save data
    if config["export"]["comprehensive"]
        data_df = DataFrame(
            observation_id = 1:length(coin_data.observations),
            outcome = coin_data.observations
        )
        data_path = joinpath(config["output"]["data_dir"], "coin_toss_observations.csv")
        mkpath(dirname(data_path))
        CSV.write(data_path, data_df)
        @info "Saved observations to: $data_path"
    end
    
    #==========================================================================
    2. BAYESIAN INFERENCE
    ==========================================================================#
    @info "\n" * "="^80
    @info "Step 2: Bayesian Inference"
    @info "="^80
    
    inference_timer = CoinTossUtils.Timer("bayesian_inference")
    
    inference_result = run_inference(
        coin_data.observations,
        config["model"]["prior_a"],
        config["model"]["prior_b"];
        iterations = config["inference"]["iterations"],
        track_fe = config["inference"]["free_energy_tracking"],
        convergence_check = config["inference"]["convergence_check"],
        convergence_tol = config["inference"]["convergence_tolerance"],
        showprogress = config["inference"]["showprogress"]
    )
    
    close(inference_timer)
    
    # Log inference results
    @info "Inference completed in $(round(inference_result.execution_time, digits=4))s"
    @info "  Converged: $(inference_result.converged)"
    if inference_result.convergence_iteration !== nothing
        @info "  Convergence iteration: $(inference_result.convergence_iteration)"
    end
    
    # Log free energy trace
    if inference_result.free_energy !== nothing
        @info "Free Energy Trace (all iterations):"
        for (i, fe) in enumerate(inference_result.free_energy)
            @info "  Iteration $i: FE = $(round(fe, digits=6))"
        end
    end
    
    # Log posterior statistics
    post_stats = posterior_statistics(inference_result.posterior, 
                                       credible_level=config["analysis"]["credible_interval"])
    @info "Posterior Statistics:"
    @info "  Mean: $(round(post_stats["mean"], digits=4))"
    @info "  Mode: $(round(post_stats["mode"], digits=4))"
    @info "  Std: $(round(post_stats["std"], digits=4))"
    @info "  $(Int(config["analysis"]["credible_interval"]*100))% CI: [$(round(post_stats["credible_interval"][1], digits=4)), $(round(post_stats["credible_interval"][2], digits=4))]"
    
    # Store results
    experiment_results["results"]["inference"] = Dict(
        "execution_time" => inference_result.execution_time,
        "iterations" => inference_result.iterations,
        "converged" => inference_result.converged,
        "convergence_iteration" => inference_result.convergence_iteration,
        "posterior_statistics" => post_stats,
        "diagnostics" => inference_result.diagnostics,
        "free_energy" => inference_result.free_energy  # Add free energy trace
    )
    
    #==========================================================================
    3. STATISTICAL ANALYSIS
    ==========================================================================#
    @info "\n" * "="^80
    @info "Step 3: Statistical Analysis"
    @info "="^80
    
    analysis_timer = CoinTossUtils.Timer("statistical_analysis")
    
    # Compute analytical posterior (for validation)
    analytical_post = analytical_posterior(
        coin_data.observations,
        config["model"]["prior_a"],
        config["model"]["prior_b"]
    )
    
    # Compare with RxInfer result
    @info "Analytical vs RxInfer comparison:"
    @info "  Analytical: α=$(params(analytical_post)[1]), β=$(params(analytical_post)[2])"
    @info "  RxInfer: α=$(params(inference_result.posterior)[1]), β=$(params(inference_result.posterior)[2])"
    
    # Log marginal likelihood
    log_ml = log_marginal_likelihood(
        coin_data.observations,
        config["model"]["prior_a"],
        config["model"]["prior_b"]
    )
    @info "Log marginal likelihood: $(round(log_ml, digits=4))"
    
    # Posterior predictive check
    if config["analysis"]["posterior_predictive"]
        pp_check = posterior_predictive_check(
            inference_result.posterior,
            config["analysis"]["n_posterior_samples"]
        )
        @info "Posterior Predictive Check:"
        @info "  PP probability of heads: $(round(pp_check["pp_prob_heads"], digits=4))"
        @info "  Empirical probability: $(round(mean(coin_data.observations), digits=4))"
        
        experiment_results["results"]["posterior_predictive"] = pp_check
    end
    
    # Diagnostic information
    @info "Diagnostic Information:"
    log_dict(inference_result.diagnostics, prefix="  ")
    
    close(analysis_timer)
    
    experiment_results["results"]["analysis"] = Dict(
        "analytical_posterior" => Dict(
            "alpha" => params(analytical_post)[1],
            "beta" => params(analytical_post)[2]
        ),
        "log_marginal_likelihood" => log_ml
    )
    
    #==========================================================================
    4. VISUALIZATION
    ==========================================================================#
    @info "\n" * "="^80
    @info "Step 4: Visualization"
    @info "="^80
    
    viz_timer = CoinTossUtils.Timer("visualization")
    
    theme = config["visualization"]["theme"]
    plots_dir = config["output"]["plots_dir"]
    mkpath(plots_dir)
    
    # Create comprehensive dashboard
    @info "Creating comprehensive dashboard..."
    dashboard = plot_comprehensive_dashboard(
        inference_result.prior,
        inference_result.posterior,
        coin_data.observations,
        inference_result.free_energy;
        theta_real = coin_data.theta_real,
        theme = theme
    )
    
    if config["visualization"]["save_plots"]
        dashboard_path = joinpath(plots_dir, "comprehensive_dashboard.png")
        save_plot(dashboard, dashboard_path)
    end
    
    # Create individual plots
    @info "Creating individual diagnostic plots..."
    
    # Prior-Posterior comparison
    pp_plot = plot_prior_posterior(
        inference_result.prior,
        inference_result.posterior;
        theta_real = coin_data.theta_real,
        theme = theme
    )
    save_plot(pp_plot, joinpath(plots_dir, "prior_posterior.png"))
    
    # Credible interval
    ci_plot = plot_credible_interval(
        inference_result.posterior;
        level = config["analysis"]["credible_interval"],
        theta_real = coin_data.theta_real,
        theme = theme
    )
    save_plot(ci_plot, joinpath(plots_dir, "credible_interval.png"))
    
    # Data histogram
    data_plot = plot_data_histogram(
        coin_data.observations;
        theta_real = coin_data.theta_real,
        theme = theme
    )
    save_plot(data_plot, joinpath(plots_dir, "data_histogram.png"))
    
    # Posterior predictive
    pred_plot = plot_predictive(
        inference_result.posterior,
        coin_data.observations;
        theme = theme
    )
    save_plot(pred_plot, joinpath(plots_dir, "posterior_predictive.png"))
    
    # Free energy convergence
    if inference_result.free_energy !== nothing
        fe_plot = plot_convergence(inference_result.free_energy; theme=theme)
        save_plot(fe_plot, joinpath(plots_dir, "free_energy_convergence.png"))
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
    
    close(viz_timer)
    
    #==========================================================================
    5. ANIMATION
    ==========================================================================#
    if config["animation"]["enabled"]
        @info "\n" * "="^80
        @info "Step 5: Animation Generation"
        @info "="^80
        
        anim_timer = CoinTossUtils.Timer("animation_generation")
        
        animations_dir = config["output"]["animations_dir"]
        mkpath(animations_dir)
        
        @info "Creating inference animation..."
        anim = create_inference_animation(
            coin_data.observations,
            config["model"]["prior_a"],
            config["model"]["prior_b"],
            config["animation"]["sample_increments"];
            theta_real = coin_data.theta_real,
            fps = config["animation"]["fps"],
            theme = theme
        )
        
        anim_path = joinpath(animations_dir, "bayesian_update.gif")
        gif(anim, anim_path, fps=config["animation"]["fps"])
        @info "Animation saved to: $anim_path"
        
        close(anim_timer)
    end
    
    #==========================================================================
    6. EXPORT RESULTS
    ==========================================================================#
    @info "\n" * "="^80
    @info "Step 6: Export Results"
    @info "="^80
    
    export_timer = CoinTossUtils.Timer("results_export")
    
    # Save comprehensive results
    results_dir = save_experiment_results(
        experiment_results["experiment_name"],
        experiment_results
    )
    
    @info "Results exported to: $results_dir"
    
    close(export_timer)
    
    #==========================================================================
    SUMMARY
    ==========================================================================#
    @info "\n" * "="^80
    @info "EXPERIMENT SUMMARY"
    @info "="^80
    @info "Data: $(length(coin_data.observations)) observations, θ_true=$(coin_data.theta_real)"
    @info "Prior: Beta($(config["model"]["prior_a"]), $(config["model"]["prior_b"]))"
    @info "Posterior: β̂=$(round(post_stats["mean"], digits=4)) ± $(round(post_stats["std"], digits=4))"
    @info "$(Int(config["analysis"]["credible_interval"]*100))% CI: [$(round(post_stats["credible_interval"][1], digits=4)), $(round(post_stats["credible_interval"][2], digits=4))]"
    
    # Check if true value is in credible interval
    in_ci = post_stats["credible_interval"][1] <= coin_data.theta_real <= post_stats["credible_interval"][2]
    @info "True θ in credible interval: $in_ci"
    
    @info "\nOutputs:"
    @info "  Plots: $(config["output"]["plots_dir"])"
    @info "  Animations: $(config["output"]["animations_dir"])"
    @info "  Results: $results_dir"
    @info "  Logs: $(config["output"]["logs_dir"])"
    
    @info "\n" * "="^80
    @info "Experiment completed successfully!"
    @info "="^80
    
    return experiment_results
end

"""
    main()

Main entry point.
"""
function main()
    config, show_help_flag = parse_args(ARGS)
    
    if show_help_flag
        show_help()
        return
    end
    
    try
        results = run_experiment(config)
        return results
    catch e
        @error "Experiment failed with error: $e"
        if config !== nothing && get(config["logging"], "verbose", false)
            showerror(stderr, e, catch_backtrace())
        end
        rethrow(e)
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end

