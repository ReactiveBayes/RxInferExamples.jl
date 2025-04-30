#!/usr/bin/env julia
# Financial Latent Vector Autoregressive Model (FLVAR)
# Last updated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))

# Include dependencies
using Pkg

# Function to ensure all required packages are installed
function ensure_dependencies()
    required_packages = ["RxInfer", "Random", "LinearAlgebra", "Dates", "Printf", 
                         "Statistics", "Plots", "DelimitedFiles", "StatsBase", "JSON", 
                         "Distributions", "SpecialFunctions", "HTTP", "ArgParse", "JLD2"]
    
    # Check which packages need to be installed
    missing_packages = filter(pkg -> !haskey(Pkg.project().dependencies, pkg), required_packages)
    
    if !isempty(missing_packages)
        println("Installing missing dependencies: $(join(missing_packages, ", "))")
        Pkg.add(missing_packages)
    end
end

# Ensure all dependencies are installed
ensure_dependencies()

# Now load all required packages
using RxInfer, Random, LinearAlgebra, Dates, Printf, Statistics, ArgParse, JSON, JLD2
using Statistics: mean
using RxInfer: KeepLast

# Parse command line arguments
function parse_commandline()
    s = ArgParseSettings()
    
    @add_arg_table s begin
        "--data-source"
            help = "Source of financial data: 'synthetic_returns' (VAR-based log returns), 'synthetic' (complex price sim), 'alphavantage', or 'yahoo'"
            default = "synthetic_returns"
        "--symbols"
            help = "Comma-separated list of ticker symbols to analyze"
            default = "AAPL,MSFT,GOOG,AMZN,META"
        "--start-date"
            help = "Start date for data in YYYY-MM-DD format"
            default = "2020-01-01"
        "--end-date"
            help = "End date for data in YYYY-MM-DD format"
            default = string(Dates.today())
        "--api-key"
            help = "API key for data source (if required)"
            default = nothing
        "--ar-orders"
            help = "Comma-separated list of AR process orders"
            default = "5,5,5,5,5"
        "--iterations"
            help = "Number of iterations for inference"
            arg_type = Int
            default = 30
        "--output-dir"
            help = "Directory for output files"
            default = joinpath("output", "financial_lva", Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS"))
    end
    
    return parse_args(s)
end

# Include visualization module and data sources
include("flva_visualization.jl")
using .FLVAVisualization

include("data_sources.jl")
using .FinancialDataSources

# Set up logging functionality
function log_message(message; level="INFO")
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    if level == "INFO"
        println("[$timestamp] [INFO] $message")
    elseif level == "WARNING"
        println("[$timestamp] [WARNING] $message")
    elseif level == "ERROR"
        println("[$timestamp] [ERROR] $message")
    elseif level == "DEBUG"
        println("[$timestamp] [DEBUG] $message")
    end
end

# Enhanced error handling wrapper function
function with_error_handling(f, error_message)
    try
        return f()
    catch e
        log_message("$error_message: $e", level="ERROR")
        # Print stack trace for debugging but only in non-production environments
        if get(ENV, "JULIA_ENV", "development") == "development"
            log_message("Stack trace: $(stacktrace())", level="DEBUG")
        end
        return nothing
    end
end

# Function to measure execution time
function measure_time(f, description)
    log_message("Starting: $description")
    start_time = Dates.now()
    result = f()
    execution_time = Dates.value(Dates.now() - start_time) / 1000.0
    log_message(@sprintf("Completed: %s in %.2f seconds", description, execution_time))
    return result, execution_time
end

log_message("Starting Financial Latent Vector Autoregressive Model (FLVAR) script")

# Function to prepare financial data for the model
function prepare_financial_data(financial_data, dates, test_ratio=0.2)
    log_message("Preparing financial data for the model (using log returns)")
    
    symbols = collect(keys(financial_data))
    n_symbols = length(symbols)
    n_samples_raw = length(dates)
    
    if n_samples_raw < 2
        log_message("Need at least 2 data points to calculate returns.", level="ERROR")
        return nothing
    end

    # Calculate log returns: log(price[t] / price[t-1])
    log_returns_data = Vector{Vector{Float64}}(undef, n_samples_raw - 1)
    valid_indices = Int[] # Keep track of indices with valid returns for all symbols

    for i in 2:n_samples_raw
        current_returns = Vector{Float64}(undef, n_symbols)
        all_valid = true
        for (k, symbol) in enumerate(symbols)
            price_t = financial_data[symbol][i]
            price_t_minus_1 = financial_data[symbol][i-1]
            
            # Check for valid prices (non-zero, finite)
            if !isfinite(price_t) || !isfinite(price_t_minus_1) || price_t_minus_1 <= 0 || price_t <= 0
                log_message("Invalid price data for symbol $symbol at index $i or $(i-1). Skipping return calculation.", level="WARNING")
                all_valid = false
                break # Skip this time step if any symbol has invalid data
            end
            current_returns[k] = log(price_t / price_t_minus_1)
        end
        
        if all_valid
            log_returns_data[i-1] = current_returns
            push!(valid_indices, i-1)
        end
    end

    # Filter out any time steps where returns couldn't be calculated for all symbols
    log_returns_data = log_returns_data[valid_indices]
    n_samples = length(log_returns_data) # Number of valid return samples
    
    if n_samples < 2
        log_message("Insufficient valid log returns data after processing ($n_samples points).", level="ERROR")
        return nothing
    end

    # Calculate the number of test samples based on valid returns
    n_test = max(1, round(Int, n_samples * test_ratio))
    n_train = n_samples - n_test
    
    log_message("Using $n_train log return samples for training and $n_test samples for testing (Total valid returns: $n_samples)")
    
    # Create the true data array (log returns)
    true_data = log_returns_data # Already in the correct format
    
    # Observations are the log returns themselves (no added noise needed typically)
    observations = deepcopy(true_data)
    
    # Create training set by marking test data as missing
    training_set = Vector{Any}(undef, n_samples)
    
    # Add known observations for training
    for i in 1:n_train
        training_set[i] = observations[i]
    end
    
    # Extend observations with missing values for the test set
    for i in (n_train + 1):n_samples
        training_set[i] = missing
    end
    
    log_message("Log return data preparation complete")
    
    # Note: Return the dates corresponding to the *returns*, which start from the second original date
    return true_data, observations, training_set, symbols, n_train, n_test, dates[2:n_samples_raw][valid_indices]
end

# Model definition functions (similar to LVAR, but adapted for financial data)
function form_priors(orders)
    log_message("Forming priors for the model", level="DEBUG")
    priors = (x = [], γ = [], θ = [])
    for k in 1:length(orders)
        push!(priors[:γ], GammaShapeRate(1.0, 1.0))
        push!(priors[:x], MvNormalMeanPrecision(zeros(orders[k]), diageye(orders[k])))
        push!(priors[:θ], MvNormalMeanPrecision(zeros(orders[k]), diageye(orders[k])))
    end
    return priors
end

function form_c_b(y, orders)
    log_message("Forming coefficients for the model", level="DEBUG")
    c = Any[]
    b = Any[]
    for k in 1:length(orders)
        _c = ReactiveMP.ar_unit(Multivariate, orders[k])
        _b = zeros(length(y[1])); _b[k] = 1.0
        push!(c, _c)
        push!(b, _b)
    end
    return c, b
end

@model function AR_sequence(x, index, length, priors, order)
    γ ~ priors[:γ][index]
    θ ~ priors[:θ][index]
    x_prev ~ priors[:x][index]
    for i in 1:length
        x[index, i] ~ AR(x_prev, θ, γ) where {
            meta = ARMeta(Multivariate, order, ARsafe())
        }
        x_prev = x[index, i]
    end
end

@model function dot_sequence(out, k, i, orders, x, c, b)
    if k === length(orders)
        out ~ b[k] * dot(c[k], x[k, i])
    else 
        next ~ dot_sequence(k = k + 1, i = i, orders = orders, x = x, c = c, b = b)
        out  ~ b[k] * dot(c[k], x[k, i]) + next
    end
end

@model function FLVAR(y, orders)
    priors   = form_priors(orders)
    c, b     = form_c_b(y, orders)
    y_length = length(y)
    
    local x # `x` is being initialized in the loop within submodels
    for k in 1:length(orders)
        x ~ AR_sequence(index = k, length = y_length, priors = priors, order = orders[k])
    end

    τ ~ GammaShapeRate(1.0, 1.0)
    for i in 1:y_length
        μ[i] ~ dot_sequence(k = 1, i = i, orders = orders, x = x, c = c, b = b)
        y[i] ~ MvNormalMeanScalePrecision(μ[i], τ)
    end
end

@constraints function flvar_constraints()
    for q in AR_sequence
        q(x, x_prev, γ, θ) = q(x, x_prev)q(γ)q(θ)
    end
    q(μ, τ) = q(μ)q(τ)
end

@initialization function flvar_init(orders)
    for init in AR_sequence
        q(γ) = GammaShapeRate(1.0, 1.0) 
        q(θ) = MvNormalMeanPrecision(zeros(orders[1]), diageye(orders[1])) # `orders[1]` is a limitation
    end
    q(τ) = GammaShapeRate(1.0, 1.0)
    μ(x) = MvNormalMeanPrecision(zeros(orders[1]), diageye(orders[1]))
end

# Function to run inference
function run_inference(training_set, orders, iterations=30)
    log_message("Starting model inference with $iterations iterations")
    
    # Run the inference using built-in progress tracking
    mresult, inference_time = measure_time(
        () -> infer(
            model          = FLVAR(orders = orders), 
            data           = (y = training_set, ), 
            constraints    = flvar_constraints(), 
            initialization = flvar_init(orders), 
            returnvars     = KeepLast(), 
            options        = (limit_stack_depth = 100, ), 
            showprogress   = true,
            iterations     = iterations,
        ),
        "Model inference"
    )

    # Add additional inference statistics
    inference_stats = Dict(
        "iterations" => iterations,
        "execution_time_seconds" => inference_time,
        "iterations_per_second" => inference_time > 0 ? iterations / inference_time : NaN
    )
    
    log_message(@sprintf("Average inference speed: %.2f iterations/second", 
                         inference_stats["iterations_per_second"]))
    
    return mresult, inference_stats
end

# Calculate prediction metrics (Adapted from LVAR_Refactored.jl)
function calculate_metrics(mresult, true_data, n_train, n_test, n_symbols)
    log_message("Calculating prediction metrics")
    
    if isempty(mresult.predictions[:y])
        log_message("No predictions found in mresult. Skipping metrics calculation.", level="WARNING")
        return Dict("error" => "No predictions available")
    end

    predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    actual_values = getindex.(true_data, :)
    n_samples = n_train + n_test

    # Calculate metrics for test period only
    test_indices = (n_train + 1):n_samples
    
    if isempty(test_indices)
        log_message("No test data available (n_test = 0). Skipping metrics calculation.", level="WARNING")
        return Dict("error" => "No test data")
    end

    test_rmse_by_symbol = zeros(n_symbols)
    test_mae_by_symbol = zeros(n_symbols)
    test_mape_by_symbol = zeros(n_symbols) .+ NaN # Initialize with NaN
    
    for symbol_idx in 1:n_symbols
        # Ensure indices are valid
        if isempty(predicted_means) || length(predicted_means) < maximum(test_indices) || symbol_idx > length(predicted_means[first(test_indices)])
             log_message("Prediction data structure mismatch or insufficient length. Symbol index: $symbol_idx", level="WARNING")
             continue # Skip this symbol
        end
         if isempty(actual_values) || length(actual_values) < maximum(test_indices) || symbol_idx > length(actual_values[first(test_indices)])
             log_message("Actual data structure mismatch or insufficient length. Symbol index: $symbol_idx", level="WARNING")
             continue # Skip this symbol
        end

        proc_pred = [predicted_means[i][symbol_idx] for i in test_indices]
        proc_actual = [actual_values[i][symbol_idx] for i in test_indices]
        
        # Calculate different metrics
        test_rmse_by_symbol[symbol_idx] = sqrt(mean((proc_pred .- proc_actual).^2))
        test_mae_by_symbol[symbol_idx] = mean(abs.(proc_pred .- proc_actual))
        
        # MAPE with protection against division by zero and NaN/Inf propagation
        nonzero_indices = findall(x -> !iszero(x) && isfinite(x), proc_actual)
        if !isempty(nonzero_indices)
            mape_values = abs.((proc_pred[nonzero_indices] .- proc_actual[nonzero_indices]) ./ proc_actual[nonzero_indices])
            finite_mape_values = filter(isfinite, mape_values)
            if !isempty(finite_mape_values)
                test_mape_by_symbol[symbol_idx] = 100 * mean(finite_mape_values)
            end # Keep NaN if all MAPE values are non-finite
        end # Keep NaN if no valid non-zero actuals
    end

    # Filter out NaNs before calculating averages/min/max
    finite_rmse = filter(!isnan, test_rmse_by_symbol)
    finite_mae = filter(!isnan, test_mae_by_symbol)
    finite_mape = filter(!isnan, test_mape_by_symbol)

    avg_test_rmse = !isempty(finite_rmse) ? mean(finite_rmse) : NaN
    avg_test_mae = !isempty(finite_mae) ? mean(finite_mae) : NaN
    avg_test_mape = !isempty(finite_mape) ? mean(finite_mape) : NaN
    
    log_message(@sprintf("Average test RMSE across all symbols: %.4f", avg_test_rmse))
    log_message(@sprintf("Average test MAE across all symbols: %.4f", avg_test_mae))
    if !isnan(avg_test_mape)
        log_message(@sprintf("Average test MAPE across all symbols: %.2f%%", avg_test_mape))
    end
    if !isempty(finite_rmse)
        log_message(@sprintf("Min symbol RMSE: %.4f, Max symbol RMSE: %.4f", 
                            minimum(finite_rmse), maximum(finite_rmse)))
    end
                        
    return Dict(
        "rmse" => test_rmse_by_symbol,
        "mae" => test_mae_by_symbol,
        "mape" => test_mape_by_symbol,
        "avg_rmse" => avg_test_rmse,
        "avg_mae" => avg_test_mae,
        "avg_mape" => avg_test_mape
    )
end

# Function to save results to files
function save_results(mresult, test_metrics, execution_metrics, symbols, orders, parsed_args, output_dir)
    log_message("Saving results to JSON files")
    saved_files = Dict{String, String}()

    # Ensure output directories exist
    metrics_dir = joinpath(output_dir, "metrics")
    raw_dir = joinpath(output_dir, "raw_data")
    mkpath(metrics_dir)
    mkpath(raw_dir)

    # Save test metrics
    try
        metrics_path = joinpath(metrics_dir, "test_metrics.json")
        open(metrics_path, "w") do io
            JSON.print(io, test_metrics, 4)
        end
        saved_files["test_metrics_path"] = metrics_path
        log_message("Test metrics saved to: $(basename(metrics_path))")
    catch e
        log_message("Failed to save test metrics: $e", level="ERROR")
    end

    # Save execution summary (including inference stats and timing)
    try
        summary_path = joinpath(output_dir, "results_summary.json")
        # Combine execution metrics with run parameters for a full summary
        summary_data = Dict(
            "run_parameters" => parsed_args,
            "model_details" => Dict("symbols" => symbols, "orders" => orders),
            "execution_metrics" => execution_metrics,
            "test_metrics_summary" => Dict(k => test_metrics[k] for k in keys(test_metrics) if occursin("avg", k))
        )
        open(summary_path, "w") do io
            JSON.print(io, summary_data, 4)
        end
        saved_files["results_summary_path"] = summary_path
        log_message("Results summary saved to: $(basename(summary_path))")
    catch e
        log_message("Failed to save results summary: $e", level="ERROR")
    end

    # Save selected raw inference results (predictions and posteriors)
    try
        raw_results_path = joinpath(raw_dir, "inference_result.jld2") # Requires JLD2 package
        
        # Create a dictionary to hold the parts of mresult we want to save
        results_to_save = Dict{Symbol, Any}()
        
        # Save predictions if available
        if isdefined(mresult, :predictions) && !isempty(mresult.predictions)
            results_to_save[:predictions] = mresult.predictions
            log_message("Included predictions in raw results.", level="DEBUG")
        else
            log_message("No predictions found in mresult to save.", level="WARNING")
        end
        
        # Save posteriors if available
        if isdefined(mresult, :posteriors) && !isempty(mresult.posteriors)
            results_to_save[:posteriors] = mresult.posteriors
             log_message("Included posteriors in raw results.", level="DEBUG")
        else
            log_message("No posteriors found in mresult to save.", level="WARNING")
        end
        
        # Save the selected results dictionary using JLD2
        if !isempty(results_to_save)
            save_object(raw_results_path, results_to_save) 
            saved_files["raw_results_path"] = raw_results_path
            log_message("Selected raw inference results (predictions, posteriors) saved to: $(basename(raw_results_path))")
        else
             log_message("No raw results (predictions or posteriors) were available to save.", level="WARNING")
        end
        
    catch e
        log_message("Failed to save selected raw inference results: $e", level="WARNING")
        # Print stack trace for debugging if in development environment
        if get(ENV, "JULIA_ENV", "development") == "development"
            log_message("Stack trace: $(stacktrace())", level="DEBUG")
        end
    end

    return saved_files
end

# Main function
function main()
    # Store overall execution metrics
    execution_metrics = Dict{String, Any}()
    total_start_time = Dates.now()
    
    log_message("Parsing command line arguments")
    parsed_args = parse_commandline()
    
    # Determine output directory
    output_dir = abspath(parsed_args["output-dir"])
    if !isdir(dirname(output_dir))
        mkpath(dirname(output_dir))
    end
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    log_message("Output directory set to: $output_dir")
    execution_metrics["output_directory"] = output_dir

    # Set random seed for reproducibility
    Random.seed!(42)
    log_message("Random seed set to 42 for reproducibility")

    # --- Data Loading --- 
    data_source_type = parsed_args["data-source"]
    local financial_data, dates, true_log_returns_dict, observed_log_returns_dict, return_dates
    data_load_result, data_load_time = measure_time(
        () -> begin
            log_message("Loading data using source: $data_source_type")
            
            # Call the appropriate data loading function
            load_func = () -> FinancialDataSources.get_financial_data(
                split(parsed_args["symbols"], ','),       
                parsed_args["start-date"],      
                parsed_args["end-date"],        
                api_key = parsed_args["api-key"], 
                data_source = data_source_type 
            )
            
            # Handle different return types based on data source
            data_result = with_error_handling(load_func, "Error loading data")
            
            if data_result === nothing
                return false # Indicate failure
            end

            if data_source_type == "synthetic_returns"
                # Expected return: true_log_returns_dict, observed_log_returns_dict, return_dates
                if length(data_result) != 3
                    log_message("Unexpected return structure from generate_synthetic_log_returns.", level="ERROR")
                    return false
                end
                true_log_returns_dict, observed_log_returns_dict, return_dates = data_result
                # We don't have original 'financial_data' (prices) or 'dates' in this case
                financial_data = nothing 
                dates = nothing
                return !isnothing(true_log_returns_dict) # Check if log returns were loaded
            else 
                # Expected return for other sources: financial_data (prices), dates
                 if length(data_result) != 2
                    log_message("Unexpected return structure from data source '$data_source_type'.", level="ERROR")
                    return false
                end
                financial_data, dates = data_result
                # Log returns will be calculated later in prepare_financial_data
                true_log_returns_dict = nothing
                observed_log_returns_dict = nothing
                return_dates = nothing
                return !isnothing(financial_data) # Check if price data was loaded
            end
        end,
         "Data loading"
    )

    if !data_load_result
         log_message("Data loading failed. Exiting.", level="ERROR")
         return nothing
    end
    execution_metrics["data_loading_time"] = data_load_time
    # --- End Data Loading ---

    # --- Data Preparation ---
    # If using synthetic_returns, data is already prepared. Otherwise, calculate log returns from prices.
    local true_data, observations, training_set, symbols, n_train, n_test 
    data_prep_time = 0.0 # Initialize

    if data_source_type == "synthetic_returns"
        log_message("Using pre-generated synthetic log returns. Skipping price-to-return conversion.")
        
        symbols = collect(keys(true_log_returns_dict))
        n_symbols = length(symbols)
        n_samples = length(return_dates) # Should be same for true and observed

        if n_samples == 0 || n_symbols == 0
             log_message("No synthetic log return data found after loading.", level="ERROR")
             return nothing
        end

        # Convert dictionary data to Vector{Vector{Float64}} format
        true_data = [ [true_log_returns_dict[s][t] for s in symbols] for t in 1:n_samples ]
        obs_data_vec = [ [observed_log_returns_dict[s][t] for s in symbols] for t in 1:n_samples ]

        # Create training/test split (default 80/20 split)
        test_ratio = 0.2 
        n_test = max(1, round(Int, n_samples * test_ratio))
        n_train = n_samples - n_test
        log_message("Using $n_train synthetic log return samples for training and $n_test samples for testing (Total: $n_samples)")

        # Create training set with missing values for test period
        training_set = Vector{Any}(undef, n_samples)
        for i in 1:n_train
            training_set[i] = obs_data_vec[i]
        end
        for i in (n_train + 1):n_samples
            training_set[i] = missing
        end

        observations = obs_data_vec # Keep the full observed data (with noise)

    else # Handle price-based sources (synthetic prices, alphavantage, yahoo)
        log_message("Preparing financial data (calculating log returns from prices)")
        prep_result, prep_time = measure_time(
            () -> begin
                if financial_data === nothing || dates === nothing
                    log_message("Price data not available for preparation.", level="ERROR")
                    return nothing
                end
                prepare_financial_data(financial_data, dates) # Use loaded price data
            end,
             "Data preparation (log returns from prices)"
        )
        data_prep_time = prep_time

        if prep_result === nothing
            log_message("Data preparation failed.", level="ERROR")
            return nothing
        end
        # Unpack results from prepare_financial_data
        true_data, observations, training_set, symbols, n_train, n_test, return_dates = prep_result
        n_samples = n_train + n_test
        n_symbols = length(symbols)
    end

    execution_metrics["data_preparation_time"] = data_prep_time
    # --- End Data Preparation ---
    
    # Parse AR orders
    orders = tryparse.(Int, split(parsed_args["ar-orders"], ','))
    if any(isnothing, orders) || length(orders) != n_symbols
        log_message("Error: Number of AR orders must match the number of symbols ($(n_symbols)). Provided: $(parsed_args["ar-orders"])", level="ERROR")
        return nothing
    end
    orders = Int.(orders) # Convert to Int
    
    log_message(@sprintf("Model configuration: %d symbols/processes with orders %s, %d samples (%d training, %d testing)",
                        n_symbols, join(string.(orders), ", "), n_samples, n_train, n_test))

    # Run inference
    inference_result, inference_stats = with_error_handling(
        () -> run_inference(training_set, orders, parsed_args["iterations"]),
        "Error during model inference"
    )
    
    if inference_result === nothing
        log_message("Inference failed. Exiting.", level="ERROR")
        return nothing
    end
    
    mresult = inference_result
    execution_metrics["inference"] = inference_stats
    
    # Calculate metrics
    metrics_result, metrics_time = measure_time(
        () -> calculate_metrics(mresult, true_data, n_train, n_test, n_symbols),
        "Metrics calculation"
    )
    test_metrics = metrics_result
    execution_metrics["metrics_calculation_time"] = metrics_time

    # Generate visualizations and save results
    visualization_time = 0.0 # Initialize
    save_time = 0.0 # Initialize
    
    vis_save_result, vis_save_time = measure_time(
        () -> begin
            # Generate Visualizations using the adapted module
            log_message("Generating comprehensive visualizations and data exports")
            local vis_result # Use local scope for visualization results
            local save_result = Dict{String, Any}() # Initialize save_result dictionary
            try
                # Use the new wrapper function 
                vis_export_results = FLVAVisualization.visualize_and_export_all(
                    mresult, 
                    true_data,      # Pass true log returns 
                    return_dates,   # Use dates corresponding to returns
                    symbols,
                    n_train, n_test, # Pass train/test counts
                    test_metrics,   # Pass calculated test metrics
                    output_dir,     # Pass output dir for saving plots
                    # Pass metadata (train/test split index refers to log return samples)
                    Dict("train_test_split" => n_train) 
                )
                # Extract visualization and export paths
                vis_result = Dict(k => v for (k,v) in vis_export_results if occursin("_plot", k) || occursin("_heatmap", k))
                save_result = Dict(k => v for (k,v) in vis_export_results if occursin("_file", k))
                
                log_message("Comprehensive visualization and export completed successfully")
            catch e
                log_message("Error during visualization/export: $e", level="ERROR")
                # Optionally: provide more details in debug mode
                if get(ENV, "JULIA_ENV", "development") == "development"
                    log_message("Stack trace: $(stacktrace())", level="DEBUG")
                end
                vis_result = Dict("error" => "Visualization failed")
            end

            # Save execution summary and raw results (metrics/stats now saved by visualize_and_export_all)
            log_message("Saving execution summary and raw results")
            try
                # Pass only necessary info now
                saved_summary_raw = save_results(mresult, test_metrics, execution_metrics, symbols, orders, parsed_args, output_dir)
                # Merge the file paths 
                merge!(save_result, saved_summary_raw)
                log_message("Execution summary and raw results saved successfully")
            catch e
                log_message("Error saving execution summary/raw results: $e", level="ERROR")
                 if get(ENV, "JULIA_ENV", "development") == "development"
                    log_message("Stack trace: $(stacktrace())", level="DEBUG")
                end
               save_result["error_saving_summary"] = "Failed to save summary/raw results"
            end
            
            return (visualizations = vis_result, saved_files = save_result)
        end,
        "Comprehensive Visualization and Export"
    )

    execution_metrics["visualization_and_save_time"] = vis_save_time

    log_message("All outputs saved to: $output_dir")
    if haskey(vis_save_result.saved_files, "results_summary_path")
         log_message("Results summary saved to: $(basename(vis_save_result.saved_files["results_summary_path"]))")
    end
     if haskey(vis_save_result.saved_files, "raw_results_path")
         log_message("Raw inference results saved to: $(basename(vis_save_result.saved_files["raw_results_path"]))")
    end
     if haskey(vis_save_result.visualizations, "error")
         log_message("Visualization status: $(vis_save_result.visualizations["error"])", level="WARNING")
     elseif !isempty(vis_save_result.visualizations)
         log_message("Generated visualizations:")
         for (key, path) in vis_save_result.visualizations
             if path !== nothing && isa(path, String)
                log_message("  - $key: $(basename(path))")
             end
         end
    end


    # Calculate and log total execution time
    total_execution_time = Dates.value(Dates.now() - total_start_time) / 1000.0
    log_message(@sprintf("Total script execution time: %.2f seconds", total_execution_time))
    execution_metrics["total_execution_time"] = total_execution_time

     # Add timing breakdown
    timing_breakdown = Dict(
        "data_preparation" => data_prep_time,
        "inference" => inference_stats["execution_time_seconds"],
        "metrics_calculation" => metrics_time,
        "visualization_and_save" => vis_save_time
    )
    execution_metrics["timing_breakdown"] = timing_breakdown

    log_message("Script execution completed successfully")

    # Return key results (optional, for interactive use)
    return (
        mresult = mresult,
        test_metrics = test_metrics,
        execution_metrics = execution_metrics,
        output_dir = output_dir
    )
end

# Run the main function if the script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    result = main()
    
    # If running interactively, print a summary of performance metrics
    if isinteractive() && result !== nothing
        println("\n=== Performance Summary ===")
        metrics = result.execution_metrics
        println("Output Directory: $(result.output_dir)")
        println("Total execution time: $(round(metrics["total_execution_time"], digits=2)) seconds")
            
        # Print timing breakdown as percentages
        if haskey(metrics, "timing_breakdown")
            println("\nTime breakdown:")
            breakdown = metrics["timing_breakdown"]
            total_time = metrics["total_execution_time"] # Use total time for percentage calculation
             if total_time > 0 # Avoid division by zero
                for (key, value) in breakdown
                    value_sec = round(value, digits=2)
                    percent = round(100 * value / total_time, digits=1)
                    println("  $(replace(key, "_"=>" ")): $(value_sec)s ($(percent)%)")
                end
            else
                 for (key, value) in breakdown
                     println("  $(replace(key, "_"=>" ")): $(round(value, digits=2))s")
                 end
            end
        end
            
        # Print inference metrics
        if haskey(metrics, "inference")
            println("\nInference performance:")
            inf = metrics["inference"]
            println("  Iterations: $(inf["iterations"])")
            println("  Speed: $(round(inf["iterations_per_second"], digits=2)) iterations/second")
        end

        # Print key metrics
        if haskey(result, :test_metrics) && !haskey(result.test_metrics, "error")
            println("\nTest Set Metrics:")
            tm = result.test_metrics
             println(@sprintf("  Avg RMSE: %.4f", tm["avg_rmse"]))
             println(@sprintf("  Avg MAE:  %.4f", tm["avg_mae"]))
             if !isnan(tm["avg_mape"])
                println(@sprintf("  Avg MAPE: %.2f%%", tm["avg_mape"]))
             else
                 println("  Avg MAPE: N/A")
             end
        elseif haskey(result, :test_metrics) && haskey(result.test_metrics, "error")
             println("\nTest Set Metrics: $(result.test_metrics["error"])")
        end
    elseif result === nothing
         println("\nScript execution failed. Check logs for details.")
    end
end 