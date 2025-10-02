module ParameterSweep

using Distributions
using DataFrames
using ProgressMeter
using JSON
using CSV
using Dates

# Import from other modules
include("../src/utils.jl")
include("../src/model.jl")
include("../src/inference.jl")
using .CoinTossUtils: ensure_directories
using .CoinTossModel: generate_coin_data
using .CoinTossInference: run_inference, posterior_statistics, log_marginal_likelihood, kl_divergence

export ParameterSweepConfig, generate_parameter_combinations, run_parameter_sweep

"""
Configuration for parameter sweeps
"""
struct ParameterSweepConfig
    parameter_ranges::Dict{String, Vector{Any}}
    n_trials::Int
    output_dir::String
    parallel::Bool
end

"""
Generate all combinations of parameters for sweep
"""
function generate_parameter_combinations(config::ParameterSweepConfig)
    param_names = keys(config.parameter_ranges)
    param_values = values(config.parameter_ranges)

    # Generate all combinations
    combinations = []
    if length(param_names) > 0
        # Use recursion to generate combinations
        function generate_combos(current_combo::Dict, remaining_params::Vector{String}, remaining_values::Vector{Vector{Any}})
            if length(remaining_params) == 0
                push!(combinations, copy(current_combo))
                return
            end

            param = remaining_params[1]
            values = remaining_values[1]
            for value in values
                current_combo[param] = value
                generate_combos(current_combo, remaining_params[2:end], remaining_values[2:end])
                delete!(current_combo, param)
            end
        end

        generate_combos(Dict(), collect(String, param_names), collect(Vector{Any}, param_values))
    else
        push!(combinations, Dict())
    end

    return combinations
end

"""
Run a single trial with given parameters
"""
function run_single_trial(params::Dict)
    # Extract parameters
    n_samples = get(params, "n_samples", 500)
    theta_real = get(params, "theta_real", 0.75)
    prior_a = get(params, "prior_a", 4.0)
    prior_b = get(params, "prior_b", 8.0)
    seed = get(params, "seed", 42)

    # Generate data
    data = generate_coin_data(n=n_samples, theta_real=theta_real, seed=seed)

    # Run inference
    result = run_inference(data.observations, prior_a, prior_b)

    # Return results
    return Dict(
        "params" => params,
        "data" => Dict(
            "n_samples" => n_samples,
            "theta_real" => theta_real,
            "n_heads" => sum(data.observations),
            "n_tails" => n_samples - sum(data.observations),
            "empirical_rate" => sum(data.observations) / n_samples
        ),
        "inference" => Dict(
            "posterior_mean" => mean(result.posterior),
            "posterior_std" => std(result.posterior),
            "converged" => result.converged,
            "iterations" => result.iterations,
            "execution_time" => result.execution_time
        )
    )
end

"""
Run complete parameter sweep
"""
function run_parameter_sweep(config::ParameterSweepConfig)
    println("Starting parameter sweep with $(length(generate_parameter_combinations(config))) parameter combinations")

    # Generate all parameter combinations
    param_combinations = generate_parameter_combinations(config)

    # Ensure output directory exists
    ensure_directories(Dict("output" => Dict(
        "output_dir" => config.output_dir,
        "data_dir" => config.output_dir,
        "plots_dir" => config.output_dir,
        "results_dir" => config.output_dir,
        "logs_dir" => config.output_dir
    )))

    # Run trials
    results = []
    @showprogress "Running parameter sweep..." for params in param_combinations
        try
            trial_result = run_single_trial(params)
            push!(results, trial_result)

            # Save intermediate results
            timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
            filename = "sweep_trial_$(length(results))_$(timestamp).json"
            filepath = joinpath(config.output_dir, filename)
            open(filepath, "w") do f
                JSON.print(f, trial_result, 2)
            end

        catch e
            @warn "Trial failed: $e"
            push!(results, Dict(
                "params" => params,
                "error" => string(e),
                "status" => "failed"
            ))
        end
    end

    # Save complete results
    complete_results = Dict(
        "config" => Dict(
            "parameter_ranges" => config.parameter_ranges,
            "n_trials" => config.n_trials,
            "parallel" => config.parallel
        ),
        "results" => results,
        "summary" => Dict(
            "total_trials" => length(results),
            "successful_trials" => count(r -> !haskey(r, "error"), results),
            "failed_trials" => count(r -> haskey(r, "error"), results)
        )
    )

    output_file = joinpath(config.output_dir, "parameter_sweep_results.json")
    open(output_file, "w") do f
        JSON.print(f, complete_results, 2)
    end

    # Save as CSV for easier analysis
    df = create_sweep_dataframe(results)
    csv_file = joinpath(config.output_dir, "parameter_sweep_results.csv")
    CSV.write(csv_file, df)

    println("Parameter sweep completed. Results saved to $config.output_dir")
    return complete_results
end

"""
Create DataFrame from sweep results for easier analysis
"""
function create_sweep_dataframe(results)
    df_data = []

    for result in results
        if haskey(result, "error")
            push!(df_data, Dict(
                "status" => "failed",
                "error" => result["error"],
                "params" => JSON.json(result["params"])
            ))
        else
            row = Dict(
                "status" => "success",
                "n_samples" => result["data"]["n_samples"],
                "theta_real" => result["data"]["theta_real"],
                "prior_a" => result["params"]["prior_a"],
                "prior_b" => result["params"]["prior_b"],
                "n_heads" => result["data"]["n_heads"],
                "empirical_rate" => result["data"]["empirical_rate"],
                "posterior_mean" => result["inference"]["posterior_mean"],
                "posterior_std" => result["inference"]["posterior_std"],
                "converged" => result["inference"]["converged"],
                "iterations" => result["inference"]["iterations"],
                "execution_time" => result["inference"]["execution_time"]
            )
            push!(df_data, row)
        end
    end

    return DataFrame(df_data)
end

end # module
