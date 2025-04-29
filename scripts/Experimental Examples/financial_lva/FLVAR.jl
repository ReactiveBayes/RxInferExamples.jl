#!/usr/bin/env julia
# Financial Latent Vector Autoregressive Model (FLVAR)
# Last updated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))

# Include dependencies
using Pkg

# Function to ensure all required packages are installed
function ensure_dependencies()
    required_packages = ["RxInfer", "Random", "LinearAlgebra", "Dates", "Printf", 
                         "Statistics", "Plots", "DelimitedFiles", "StatsBase", "JSON", 
                         "Distributions", "SpecialFunctions", "HTTP", "ArgParse"]
    
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
using RxInfer, Random, LinearAlgebra, Dates, Printf, Statistics, ArgParse, JSON
using Statistics: mean
using RxInfer: KeepLast

# Parse command line arguments
function parse_commandline()
    s = ArgParseSettings()
    
    @add_arg_table s begin
        "--data-source"
            help = "Source of financial data: 'synthetic', 'alphavantage', or 'yahoo'"
            default = "synthetic"
        "--symbols"
            help = "Comma-separated list of ticker symbols to analyze"
            default = "AAPL,MSFT,GOOG,AMZN,META"
        "--start-date"
            help = "Start date for data in YYYY-MM-DD format"
            default = "2022-01-01"
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
    log_message("Preparing financial data for the model")
    
    # Convert Dict of time series to the format needed for the model
    symbols = collect(keys(financial_data))
    n_symbols = length(symbols)
    n_samples = length(dates)
    
    # Calculate the number of test samples
    n_test = max(1, round(Int, n_samples * test_ratio))
    n_train = n_samples - n_test
    
    log_message("Using $n_train samples for training and $n_test samples for testing")
    
    # Create the true data array for the model
    true_data = [[financial_data[symbol][i] for symbol in symbols] for i in 1:n_samples]
    
    # Add observation noise
    # For financial data, we'll use a very small noise since we want to denoise
    # the existing noise in the data, not add more
    observations = deepcopy(true_data)
    
    # Create training set by marking test data as missing
    training_set = Any[]
    
    # Add known observations for training
    for i in 1:n_train
        push!(training_set, observations[i])
    end
    
    # Extend observations with missing values for the test set
    for i in n_train+1:n_samples
        push!(training_set, missing)
    end
    
    log_message("Data preparation complete")
    
    return true_data, observations, training_set, symbols, n_train, n_test
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
        "iterations_per_second" => iterations / inference_time
    )
    
    log_message(@sprintf("Average inference speed: %.2f iterations/second", 
                         inference_stats["iterations_per_second"]))
    
    return mresult, inference_stats
end

# Calculate prediction metrics
function calculate_metrics(mresult, true_data, n_samples, n_train, n_symbols, metadata)
    log_message("Calculating prediction metrics")
    
    # Extract the predictions
    predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    predicted_covs = cov.(mresult.predictions[:y][end])
    
    # Prepare containers for metrics
    test_rmse_by_symbol = zeros(n_symbols)
    test_mae_by_symbol = zeros(n_symbols)
    
    # Calculate RMSE and MAE for each symbol (for test period only)
    for symbol_idx in 1:n_symbols
        # Calculate test RMSE
        test_predictions = [p[symbol_idx] for p in predicted_means[(n_train+1):end]]
        test_actuals = [d[symbol_idx] for d in true_data[(n_train+1):end]]
        
        squared_errors = (test_predictions .- test_actuals).^2
        absolute_errors = abs.(test_predictions .- test_actuals)
        
        test_rmse_by_symbol[symbol_idx] = sqrt(mean(squared_errors))
        test_mae_by_symbol[symbol_idx] = mean(absolute_errors)
    end
    
    # Calculate overall metrics
    avg_test_rmse = mean(test_rmse_by_symbol)
    avg_test_mae = mean(test_mae_by_symbol)
    
    # Log the metrics
    log_message(@sprintf("Average Test RMSE: %.4f", avg_test_rmse))
    log_message(@sprintf("Average Test MAE: %.4f", avg_test_mae))
    
    # Also calculate metrics relevant to financial data
    
    # Direction accuracy (whether the model correctly predicts up/down movement)
    direction_accuracy = zeros(n_symbols)
    
    for symbol_idx in 1:n_symbols
        # Get predictions and actuals for all data points except the first one
        full_predictions = [p[symbol_idx] for p in predicted_means[2:end]]
        full_actuals = [d[symbol_idx] for d in true_data[2:end]]
        
        # Calculate directions (true = up, false = down)
        pred_directions = diff([predicted_means[1][symbol_idx]; full_predictions]) .> 0
        actual_directions = diff([true_data[1][symbol_idx]; full_actuals]) .> 0
        
        # Calculate accuracy for test period only
        test_indices = n_train:(n_samples-1)  # -1 because we're using diffs
        test_pred_directions = pred_directions[test_indices]
        test_actual_directions = actual_directions[test_indices]
        
        direction_accuracy[symbol_idx] = mean(test_pred_directions .== test_actual_directions)
    end
    
    avg_direction_accuracy = mean(direction_accuracy)
    log_message(@sprintf("Average Direction Accuracy: %.2f%%", avg_direction_accuracy * 100))
    
    # Compile all metrics
    metrics = Dict(
        "overall_statistics" => Dict(
            "avg_test_rmse" => avg_test_rmse,
            "avg_test_mae" => avg_test_mae,
            "avg_direction_accuracy" => avg_direction_accuracy
        ),
        "symbol_statistics" => Dict(
            "test_rmse_by_symbol" => test_rmse_by_symbol,
            "test_mae_by_symbol" => test_mae_by_symbol,
            "direction_accuracy" => direction_accuracy
        ),
        "model_parameters" => Dict(
            "n_symbols" => n_symbols,
            "n_samples" => n_samples,
            "test_samples" => n_samples - n_train,
            "training_samples" => n_train
        ),
        "metadata" => metadata  # Include the metadata in the metrics
    )
    
    return metrics, predicted_means, predicted_covs
end

# Main function
function main()
    # Parse command line arguments
    args = parse_commandline()
    
    # Extract parameters
    data_source = args["data-source"]
    symbols = split(args["symbols"], ",")
    start_date = args["start-date"]
    end_date = args["end-date"]
    api_key = args["api-key"]
    ar_orders = parse.(Int, split(args["ar-orders"], ","))
    iterations = args["iterations"]
    output_dir = args["output-dir"]
    
    # Ensure AR orders match the number of symbols
    if length(ar_orders) != length(symbols)
        log_message("Number of AR orders doesn't match number of symbols, using default order of 5", level="WARNING")
        ar_orders = [5 for _ in 1:length(symbols)]
    end
    
    # Create output directory structure
    mkpath(output_dir)
    mkpath(joinpath(output_dir, "visualizations"))
    mkpath(joinpath(output_dir, "metrics"))
    mkpath(joinpath(output_dir, "raw_data"))
    
    log_message("Output will be saved to $output_dir")
    log_message("Created output directory structure:")
    log_message("  - Main output directory: $(abspath(output_dir))")
    log_message("  - Visualizations directory: $(abspath(joinpath(output_dir, "visualizations")))")
    log_message("  - Metrics directory: $(abspath(joinpath(output_dir, "metrics")))")
    log_message("  - Raw data directory: $(abspath(joinpath(output_dir, "raw_data")))")
    
    # Fetch or generate financial data
    log_message("Fetching financial data for $(length(symbols)) symbols from $start_date to $end_date")
    financial_data, dates = get_financial_data(symbols, start_date, end_date, 
                                              api_key=api_key, data_source=data_source)
    
    # Save raw data
    raw_data_file = joinpath(output_dir, "raw_data", "financial_data.json")
    open(raw_data_file, "w") do io
        # Convert dates to strings for JSON serialization
        serializable_data = Dict{String, Any}()
        for (symbol, prices) in financial_data
            serializable_data[symbol] = prices
        end
        date_strings = string.(dates)
        
        JSON.print(io, Dict("data" => serializable_data, "dates" => date_strings), 4)
    end
    log_message("Raw financial data saved to: $(abspath(raw_data_file))")
    
    # Prepare data for the model
    true_data, observations, training_set, symbols, n_train, n_test = 
        prepare_financial_data(financial_data, dates)
    
    # Store metadata for visualizations
    metadata = Dict(
        "symbols" => symbols,
        "start_date" => start_date,
        "end_date" => end_date,
        "data_source" => data_source,
        "ar_orders" => ar_orders,
        "train_test_split" => n_train,
        "time_generated" => Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    )
    
    # Save metadata
    metadata_file = joinpath(output_dir, "metadata.json")
    open(metadata_file, "w") do io
        JSON.print(io, metadata, 4)
    end
    log_message("Metadata saved to: $(abspath(metadata_file))")
    
    # Run inference
    mresult, inference_stats = run_inference(training_set, ar_orders, iterations)
    
    # Calculate metrics
    metrics, predicted_means, predicted_covs = 
        calculate_metrics(mresult, true_data, length(dates), n_train, length(symbols), metadata)
    
    # Add inference stats to metrics
    metrics["inference"] = inference_stats
    
    # Save metrics to file
    metrics_file = joinpath(output_dir, "metrics", "model_statistics.json")
    open(metrics_file, "w") do io
        JSON.print(io, metrics, 4)  # 4 spaces of indentation
    end
    log_message("Model statistics saved to: $(abspath(metrics_file))")
    
    # Create visualizations
    log_message("Generating visualizations")
    
    # Generate output directories for each type of visualization
    vis_dir = joinpath(output_dir, "visualizations")
    price_vis_dir = joinpath(vis_dir, "price_trends")
    indicators_dir = joinpath(vis_dir, "technical_indicators")
    
    mkpath(price_vis_dir)
    mkpath(indicators_dir)
    
    log_message("Created visualization sub-directories:")
    log_message("  - Price trends: $(abspath(price_vis_dir))")
    log_message("  - Technical indicators: $(abspath(indicators_dir))")
    
    # Basic denoised price visualizations
    price_viz_files = create_price_visualization(mresult, financial_data, dates, symbols, price_vis_dir, metadata)
    
    # Log all generated price visualization files
    log_message("Generated price trend visualizations:")
    for (i, file) in enumerate(price_viz_files)
        log_message("  $(i). $(basename(file)): $(abspath(file))")
    end
    
    # Calculate financial metrics
    financial_metrics_file = joinpath(output_dir, "metrics", "financial_metrics.json")
    financial_metrics = calculate_financial_metrics(mresult, financial_data, dates, symbols, financial_metrics_file)
    log_message("Financial metrics saved to: $(abspath(financial_metrics_file))")
    
    # Create technical indicators for each symbol
    indicator_files = String[]
    for symbol in symbols
        indicator_file = create_technical_indicators(mresult, financial_data, dates, symbols, indicators_dir, symbol_to_plot=symbol, metadata=metadata)
        push!(indicator_files, indicator_file)
    end
    
    # Log all generated technical indicator files
    log_message("Generated technical indicator visualizations:")
    for (i, file) in enumerate(indicator_files)
        log_message("  $(i). $(basename(file)): $(abspath(file))")
    end
    
    # Create index HTML file to easily view all visualizations
    index_html = joinpath(output_dir, "index.html")
    open(index_html, "w") do io
        write(io, """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Financial LVA Analysis Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                .container { display: flex; flex-wrap: wrap; }
                .img-container { margin: 10px; text-align: center; }
                img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <h1>Financial LVA Analysis Results</h1>
            <p>Analysis run on: $(metadata["time_generated"])</p>
            <p>Symbols: $(join(symbols, ", "))</p>
            <p>Period: $(metadata["start_date"]) to $(metadata["end_date"])</p>
            
            <h2>Price Trend Visualizations</h2>
            <div class="container">
        """)
        
        for file in price_viz_files
            rel_path = relpath(file, output_dir)
            write(io, """
                <div class="img-container">
                    <img src="$(rel_path)" alt="$(basename(file))">
                    <p>$(basename(file))</p>
                </div>
            """)
        end
        
        write(io, """
            </div>
            
            <h2>Technical Indicators</h2>
            <div class="container">
        """)
        
        for file in indicator_files
            rel_path = relpath(file, output_dir)
            write(io, """
                <div class="img-container">
                    <img src="$(rel_path)" alt="$(basename(file))">
                    <p>$(basename(file))</p>
                </div>
            """)
        end
        
        write(io, """
            </div>
        </body>
        </html>
        """)
    end
    
    log_message("Created HTML index page at: $(abspath(index_html))")
    log_message("All outputs have been saved to: $(abspath(output_dir))")
    log_message("FLVAR processing complete")
    
    return mresult, metrics, output_dir
end

# Execute main function if this script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 