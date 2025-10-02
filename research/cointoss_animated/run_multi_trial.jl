#!/usr/bin/env julia

"""
Multi-trial coin toss analysis with parameter sweeps and comprehensive visualization

This script demonstrates:
1. Parameter sweeps across multiple dimensions
2. Multi-trial analysis and comparison
3. Advanced visualization and animation
4. Comprehensive output generation

Usage:
    julia run_multi_trial.jl [options]

Options:
    --help, -h              Show this help
    --config=FILE          Use custom config file
    --n-trials=N           Number of trials in sweep
    --output-dir=DIR       Output directory
    --skip-animation       Skip animation generation
    --verbose              Enable verbose logging
    --quiet                Minimal output
"""

using ArgParse
using Logging
using Dates
using JSON

# Include required modules
include("src/model.jl")
include("src/inference.jl")
include("src/visualization.jl")
include("src/utils.jl")
include("src/parameter_sweep.jl")
include("src/multi_trial_analysis.jl")
include("config.jl")

# Make functions available
using .CoinTossUtils: setup_logging, ensure_directories
using .Config: load_config
using .ParameterSweep: ParameterSweepConfig, run_parameter_sweep, generate_parameter_combinations
using .MultiTrialAnalysis: MultiTrialConfig, load_trial_results, analyze_trials, create_comparison_plots, create_performance_dashboard, create_trial_dataframe
using .CoinTossVisualization: create_multi_trial_animation, create_parameter_sweep_animation

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--config"
            help = "Configuration file"
            arg_type = String
            default = "config.toml"
        "--n-trials"
            help = "Number of trials in parameter sweep"
            arg_type = Int
            default = 10
        "--output-dir"
            help = "Output directory"
            arg_type = String
            default = "outputs"
        "--skip-animation"
            help = "Skip animation generation"
            action = :store_true
        "--verbose"
            help = "Enable verbose logging"
            action = :store_true
        "--quiet"
            help = "Minimal output"
            action = :store_true
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    if haskey(args, "help") && args["help"]
        println("Multi-trial Coin Toss Analysis")
        println("Usage: julia run_multi_trial.jl [options]")
        exit(0)
    end

    # Setup logging
    setup_logging(verbose=args["verbose"])

    @info "Starting Multi-Trial Coin Toss Analysis"

    # Load configuration
    config = load_config(args["config"])
    @info "Configuration loaded from $(args["config"])"

    # Setup output directories
    output_dir = args["output-dir"]
    config = Dict("output" => Dict("output_dir" => output_dir, "data_dir" => joinpath(output_dir, "data"), "plots_dir" => joinpath(output_dir, "plots"), "results_dir" => joinpath(output_dir, "results"), "logs_dir" => joinpath(output_dir, "logs")))
    ensure_directories(config)
    sweep_output_dir = joinpath(output_dir, "parameter_sweep")
    analysis_output_dir = joinpath(output_dir, "multi_trial_analysis")

    # Ensure analysis directory exists
    if !isdir(analysis_output_dir)
        mkpath(analysis_output_dir)
        @info "Created directory" path=analysis_output_dir
    end

        # Define parameter ranges for sweep
        parameter_ranges = Dict(
            "n_samples" => [50, 100, 200, 500, 1000, 10000],
            "theta_real" => [0.3, 0.5, 0.7, 0.8],
            "prior_a" => [1.0, 2.0, 4.0, 8.0],
            "prior_b" => [1.0, 2.0, 4.0, 8.0]
        )

    # Create parameter sweep configuration
    sweep_config = ParameterSweepConfig(
        parameter_ranges,
        args["n-trials"],
        sweep_output_dir,
        false  # parallel
    )

    # Run parameter sweep
    @info "Running parameter sweep with $(length(generate_parameter_combinations(sweep_config))) combinations"
    sweep_results = run_parameter_sweep(sweep_config)

    # Analyze multi-trial results
    @info "Analyzing multi-trial results"
    analysis_config = MultiTrialConfig(
        sweep_output_dir,
        analysis_output_dir,
        ["posterior_mean", "kl_divergence", "execution_time"],
        ["theta_real", "n_samples", "prior_a"],
        0.05
    )

    trial_results = load_trial_results(sweep_output_dir)
    analysis_results = analyze_trials(trial_results, analysis_config)

    # Create visualizations
    @info "Creating multi-trial visualizations"
    df = create_trial_dataframe(filter(r -> !haskey(r, "error"), trial_results))

    # Create comparison plots
    comparison_plots = create_comparison_plots(df, analysis_config)

    # Create performance dashboard
    dashboard = create_performance_dashboard(df, analysis_config)

    # Create multi-trial animation (if not skipped)
    if !args["skip-animation"]
        @info "Creating multi-trial animations"
        multi_trial_anim = create_multi_trial_animation(trial_results, joinpath(output_dir, "animations"))
        param_sweep_anim = create_parameter_sweep_animation(trial_results, "n_samples", "posterior_mean", joinpath(output_dir, "animations"))
    end

    # Save final summary
    summary = Dict(
        "experiment_type" => "multi_trial_parameter_sweep",
        "timestamp" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
        "sweep_config" => parameter_ranges,
        "analysis_results" => analysis_results,
        "total_trials" => length(trial_results),
        "successful_trials" => count(r -> !haskey(r, "error"), trial_results),
        "outputs" => Dict(
            "sweep_results" => sweep_output_dir,
            "analysis_results" => analysis_output_dir,
            "visualizations" => output_dir
        )
    )

    summary_file = joinpath(output_dir, "multi_trial_summary.json")
    open(summary_file, "w") do f
        JSON.print(f, summary, 2)
    end

    @info "Multi-trial analysis completed successfully!"
    @info "Outputs saved to $output_dir"
    @info "Summary: $(summary["successful_trials"])/$(summary["total_trials"]) trials successful"
end

# Run main function
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
