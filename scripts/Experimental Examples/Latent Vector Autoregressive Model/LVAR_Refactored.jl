#!/usr/bin/env julia
# Latent Vector Autoregressive Model (LVAR) - Refactored Version
# Last updated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))

# Include dependencies
using Pkg

# Function to ensure all required packages are installed
function ensure_dependencies()
    required_packages = ["RxInfer", "Random", "LinearAlgebra", "Dates", "Printf", 
                         "Statistics", "Plots", "DelimitedFiles", "StatsBase", "JSON", 
                         "Distributions", "SpecialFunctions", "Logging"]
    
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
using RxInfer, Random, LinearAlgebra, Dates, Printf, Statistics
using Statistics: mean
using RxInfer: KeepLast, KeepAll # Add explicit import for KeepAll

# Include visualization module
include("lva_visualization.jl")
using .LVAVisualization

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

log_message("Starting Latent Vector Autoregressive Model (LVAR) script - Refactored Version")

# Data generation functions
function generate_ar_process(order, θ, n_samples; σ²=1.0)
    x = zeros(n_samples)
    # Initialize with random noise
    x[1:order] = randn(order) * sqrt(σ²)
    
    for t in (order+1):n_samples
        # AR equation: x[t] = θ₁x[t-1] + θ₂x[t-2] + ... + θₚx[t-p] + ε[t]
        x[t] = sum(θ[i] * x[t-i] for i in 1:order) + randn() * sqrt(σ²)
    end
    return x
end

# Function to generate synthetic data
function generate_data(orders, n_samples, n_missing)
    log_message("Generating synthetic AR processes")
    processes = []
    n_ar_processes = length(orders)

    # Initialize progress tracking
    total_processes = length(orders)
    log_message("Generating AR parameters and data for $total_processes processes")

    # Generate AR parameters and data for each process
    for (i, order) in enumerate(orders)
        if i % 5 == 0 || i == 1 || i == total_processes
            log_message(@sprintf("Generating process %d/%d (%.1f%%)", i, total_processes, 100.0*i/total_processes))
        end
        
        # Generate stable AR parameters (using a simple method)
        θ = 0.5 .^ (1:order)  # This ensures stability by having decreasing coefficients
        
        # Generate the AR process
        x = generate_ar_process(order, θ, n_samples)
        push!(processes, x)
    end

    log_message("Data generation complete")

    # Convert to the format needed for the model
    log_message("Preparing data for the model")
    true_data = [[processes[j][i] for j in 1:n_ar_processes] for i in 1:n_samples]
    observations = Any[[true_data[i][j] .+ randn() for j in 1:n_ar_processes] for i in 1:n_samples]

    log_message("Creating training set with missing values for prediction")
    training_set = deepcopy(observations[1:n_samples-n_missing])

    # Extend observations with missing values for the test set
    for i in n_samples-n_missing+1:n_samples
        push!(training_set, missing)
    end

    log_message("Data preparation complete")
    
    return true_data, observations, training_set
end

# Model definition functions
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

@model function LVAR(y, orders)
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

@constraints function lvar_constraints()
    for q in AR_sequence
        # This requires patch in GraphPPL though, see https://github.com/ReactiveBayes/GraphPPL.jl/issues/262
        # A workaround is to use `constraints = MeanField()` in the `infer` function and initializing `q(x)` instead of `μ(x)`
        q(x, x_prev, γ, θ) = q(x, x_prev)q(γ)q(θ)
    end
    q(μ, τ) = q(μ)q(τ)
end

@initialization function lvar_init(orders)
    # Note: There's a limitation here that could be addressed in future versions
    for init in AR_sequence
        q(γ) = GammaShapeRate(1.0, 1.0) 
        q(θ) = MvNormalMeanPrecision(zeros(orders[1]), diageye(orders[1])) # `orders[1]` is sad... needs to be fixed
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
            model          = LVAR(orders = orders), 
            data           = (y = training_set, ), 
            constraints    = lvar_constraints(), 
            initialization = lvar_init(orders), 
            returnvars     = KeepLast(), 
            # Use KeepAll() or equivalent for historyvars if available in your RxInfer version
            # historyvars    = (θ = KeepAll(), γ = KeepAll(), τ = KeepAll()),
            options        = (limit_stack_depth = 100, ), 
            showprogress   = true,  # Use built-in progress tracking
            iterations     = iterations,
        ),
        "Model inference"
    )

    # Add additional inference statistics
    inference_stats = Dict(
        "iterations" => iterations,
        "execution_time_seconds" => inference_time,
        "iterations_per_second" => iterations / inference_time
    )
    
    log_message(@sprintf("Average inference speed: %.2f iterations/second", 
                         inference_stats["iterations_per_second"]))
    
    return mresult, inference_stats
end

# Calculate prediction metrics
function calculate_metrics(mresult, true_data, n_samples, n_missing, n_ar_processes)
    log_message("Calculating prediction metrics")
    predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    actual_values = getindex.(true_data, :)

    # Calculate metrics for test period only
    test_indices = (n_samples-n_missing+1):n_samples
    test_rmse_by_process = zeros(n_ar_processes)
    test_mae_by_process = zeros(n_ar_processes)
    test_mape_by_process = zeros(n_ar_processes)

    for proc_idx in 1:n_ar_processes
        proc_pred = [predicted_means[i][proc_idx] for i in test_indices]
        proc_actual = [actual_values[i][proc_idx] for i in test_indices]
        
        # Calculate different metrics
        test_rmse_by_process[proc_idx] = sqrt(mean((proc_pred .- proc_actual).^2))
        test_mae_by_process[proc_idx] = mean(abs.(proc_pred .- proc_actual))
        
        # MAPE with protection against division by zero
        nonzero_indices = findall(!iszero, proc_actual)
        if !isempty(nonzero_indices)
            test_mape_by_process[proc_idx] = 100 * mean(abs.((proc_pred[nonzero_indices] .- proc_actual[nonzero_indices]) ./ proc_actual[nonzero_indices]))
        else
            test_mape_by_process[proc_idx] = NaN
        end
    end

    avg_test_rmse = mean(test_rmse_by_process)
    avg_test_mae = mean(test_mae_by_process)
    avg_test_mape = mean(filter(!isnan, test_mape_by_process))
    
    log_message(@sprintf("Average test RMSE across all processes: %.4f", avg_test_rmse))
    log_message(@sprintf("Average test MAE across all processes: %.4f", avg_test_mae))
    if !isnan(avg_test_mape)
        log_message(@sprintf("Average test MAPE across all processes: %.2f%%", avg_test_mape))
    end
    log_message(@sprintf("Min process RMSE: %.4f, Max process RMSE: %.4f", 
                        minimum(test_rmse_by_process), maximum(test_rmse_by_process)))
                        
    return Dict(
        "rmse" => test_rmse_by_process,
        "mae" => test_mae_by_process,
        "mape" => test_mape_by_process,
        "avg_rmse" => avg_test_rmse,
        "avg_mae" => avg_test_mae,
        "avg_mape" => avg_test_mape
    )
end

# Main function to run the entire workflow
function main()
    # Store overall execution metrics
    execution_metrics = Dict{String, Any}()
    total_start_time = Dates.now()
    
    # Set random seed for reproducibility
    Random.seed!(42)
    log_message("Random seed set to 42 for reproducibility")
    
    # Define model parameters
    log_message("Initializing model parameters")
    orders = 5 .* ones(Int, 20)
    n_samples = 120
    n_missing = 20
    n_ar_processes = length(orders)
    
    log_message(@sprintf("Model configuration: %d AR processes with order %d, %d samples (%d for training, %d for testing)",
                        n_ar_processes, Int(orders[1]), n_samples, n_samples-n_missing, n_missing))
    
    # Generate synthetic data
    data_result, data_time = measure_time(
        () -> generate_data(orders, n_samples, n_missing),
        "Data generation"
    )
    true_data, observations, training_set = data_result
    execution_metrics["data_generation_time"] = data_time
    
    # Run inference
    inference_result, inference_stats = with_error_handling(
        () -> run_inference(training_set, orders),
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
        () -> calculate_metrics(mresult, true_data, n_samples, n_missing, n_ar_processes),
        "Metrics calculation"
    )
    test_metrics = metrics_result
    execution_metrics["metrics_calculation_time"] = metrics_time
    
    # Generate visualizations and export data using the enhanced visualization module
    log_message("Generating enhanced visualizations")
    vis_result, vis_time = measure_time(
        () -> LVAVisualization.visualize_and_export(
            mresult, true_data, observations, n_samples, n_missing, test_metrics["rmse"]),
        "Visualization and data export"
    )
    visualization_results = vis_result
    execution_metrics["visualization_time"] = vis_time
    
    # Display summary of created visualizations
    output_dir = visualization_results["output_dir"]
    log_message("All visualizations have been saved to: $output_dir")
    log_message("Generated outputs:")
    
    # List all visualization outputs
    for (key, value) in visualization_results
        if key != "output_dir" && value !== nothing
            if typeof(value) <: AbstractString
                log_message("  - $key: $(basename(value))")
            else
                log_message("  - $key: generated successfully")
            end
        end
    end
    
    # Calculate and log total execution time
    total_execution_time = Dates.value(Dates.now() - total_start_time) / 1000.0
    log_message(@sprintf("Total script execution time: %.2f seconds", total_execution_time))
    execution_metrics["total_execution_time"] = total_execution_time
    
    # Add timing breakdown
    timing_breakdown = Dict(
        "data_generation" => data_time,
        "inference" => inference_stats["execution_time_seconds"],
        "metrics_calculation" => metrics_time,
        "visualization" => vis_time
    )
    execution_metrics["timing_breakdown"] = timing_breakdown
    
    log_message("Script execution completed successfully")
    
    # Return key results
    return (
        mresult = mresult,
        test_metrics = test_metrics,
        visualizations = visualization_results,
        execution_metrics = execution_metrics
    )
end

# Run the main function if this script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    result = main()
    
    # If running interactively, print a summary of performance metrics
    if isinteractive()
        if result !== nothing && haskey(result, :execution_metrics)
            println("\n=== Performance Summary ===")
            metrics = result.execution_metrics
            println("Total execution time: $(round(metrics["total_execution_time"], digits=2)) seconds")
            
            # Print timing breakdown as percentages
            if haskey(metrics, "timing_breakdown")
                println("\nTime breakdown:")
                breakdown = metrics["timing_breakdown"]
                total = sum(values(breakdown))
                for (key, value) in breakdown
                    percent = round(100 * value / total, digits=1)
                    println("  $key: $(round(value, digits=2))s ($(percent)%)")
                end
            end
            
            # Print inference metrics
            if haskey(metrics, "inference")
                println("\nInference performance:")
                inf = metrics["inference"]
                println("  Iterations: $(inf["iterations"])")
                println("  Speed: $(round(inf["iterations_per_second"], digits=2)) iterations/second")
            end
        end
    end
end 