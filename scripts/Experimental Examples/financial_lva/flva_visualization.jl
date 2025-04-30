module FLVAVisualization

using Plots, DelimitedFiles, Printf, Dates, Statistics, LinearAlgebra, StatsBase, JSON, Distributions, SpecialFunctions
using Statistics: mean, median, std, quantile, cor
using LinearAlgebra: diag, tr, norm, svd
using Plots: Animation, @animate, gif
using Distributions: Normal
using SpecialFunctions: erfinv

# Define log_message function used in this module
function log_message(message; level="INFO")
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    println("[$timestamp] [$level] $message")
end

# Function to create standard plot theme for consistent styling
function set_financial_theme()
    theme(:dark)  # Dark theme popular for financial charts
    default(
        fontfamily="Computer Modern",
        linewidth=2,
        framestyle=:box,
        label=nothing,
        grid=true,
        palette=:darktest,
        legendfontsize=8,
        tickfontsize=8,
        guidefontsize=10,
        titlefontsize=12,
        margin=5Plots.mm,
        size=(800, 600)
    )
end

# Function to create candlestick chart
function create_candlestick_chart(ohlc_data, dates, ticker, output_dir)
    log_message("Generating candlestick chart for $ticker")
    
    set_financial_theme()
    
    # Create candlestick plot
    p = plot(dates, 
             ohlc_data, 
             seriestype=:candlestick,
             title="$ticker Price History",
             xlabel="Date",
             ylabel="Price",
             size=(800, 400),
             bar_width=0.7)
    
    # Save plot to file
    mkpath(output_dir)
    plot_filename = joinpath(output_dir, "$(ticker)_candlestick.png")
    savefig(p, plot_filename)
    
    log_message("Candlestick chart saved to: $plot_filename")
    return plot_filename
end

# Function to create basic price and denoised visualizations
function create_logreturn_visualization(mresult, true_log_returns, return_dates, symbols, output_dir, metadata=nothing)
    log_message("Generating log return visualization plots")
    
    set_financial_theme()
    
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Extract predictions (which are already log returns)
    predicted_means_logret = getindex.(mean.(mresult.predictions[:y][end]), :)
    predicted_stds_logret = [sqrt.(diag(cov.(mresult.predictions[:y][end])[i])) for i in 1:length(predicted_means_logret)]
    
    # Plotting options
    ribbon_alpha = 0.3
    
    # Create a plot for each symbol
    plot_filenames = []
    n_samples_logret = length(return_dates) # Number of log return points
    
    # Select a subset of symbols to plot if too many
    symbols_to_plot = length(symbols) > 5 ? symbols[1:5] : symbols 
    log_message("Plotting log returns for symbols: $(join(symbols_to_plot, ", "))")

    for symbol in symbols_to_plot
        symbol_idx = findfirst(s -> s == symbol, symbols)
        if symbol_idx === nothing
            log_message("Symbol $symbol not found in results, skipping plot.", level="WARNING")
            continue
        end

        p = plot(size=(800, 500), 
                title="$symbol: Actual vs. Predicted Log Returns",
                xlabel="Date",
                ylabel="Log Return",
                legend=:topleft)
        
        # Get true log returns for this symbol
        actual_log_returns_symbol = [ret[symbol_idx] for ret in true_log_returns]
        
        # Plot true log returns
        plot!(p, return_dates, actual_log_returns_symbol, 
             label="Actual Log Returns",
             color=:lightblue,
             alpha=0.8,
             linewidth=1.5)
        
        # Plot predicted log returns with uncertainty
        pred_values_logret = [pred[symbol_idx] for pred in predicted_means_logret]
        pred_stds = [std[symbol_idx] for std in predicted_stds_logret]
        
        # Ensure lengths match return_dates before plotting
        if length(pred_values_logret) != n_samples_logret || length(pred_stds) != n_samples_logret
             log_message("Mismatch in prediction length for $symbol. Expected $n_samples_logret, got $(length(pred_values_logret)). Skipping plot.", level="ERROR")
             continue
        end

        plot!(p, return_dates, pred_values_logret,
             ribbon=pred_stds,
             fillalpha=ribbon_alpha,
             label="Predicted Log Returns",
             color=:orange,
             linewidth=2)
        
        # Add training/test split line if applicable
        if metadata !== nothing && haskey(metadata, "train_test_split")
            split_idx = metadata["train_test_split"] # This index refers to the log return samples
            if 1 <= split_idx < n_samples_logret # Ensure index is valid for return_dates
                 vline!(p, [return_dates[split_idx]], 
                      label="Train/Test Split", 
                      linestyle=:dash, 
                      color=:white)
            else
                 log_message("Train/test split index $split_idx is out of bounds for return dates (length $n_samples_logret).", level="WARNING")
            end
        end
        
        # Save individual plot to file
        plot_filename = joinpath(output_dir, "$(symbol)_logreturns.png")
        savefig(p, plot_filename)
        push!(plot_filenames, plot_filename)
        
        log_message("Created log return visualization for $symbol")
    end
    
    log_message("All log return visualization plots saved to: $output_dir")
    return plot_filenames
end

# Function to calculate financial metrics
function calculate_financial_metrics(mresult, financial_data, dates, symbols, metrics_file)
    log_message("Calculating financial metrics")
    
    # Extract predictions
    predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    
    # Calculate metrics for each symbol
    metrics = Dict{String, Dict{String, Any}}()
    
    for (i, symbol) in enumerate(symbols)
        log_message("Calculating metrics for $symbol", level="DEBUG")
        
        # Get actual price data
        original_prices = financial_data[symbol]
        
        # Get denoised values
        denoised_values = [pred[i] for pred in predicted_means]
        
        # Calculate trend direction accuracy
        trend_directions_actual = diff(original_prices) .> 0
        trend_directions_pred = diff(denoised_values) .> 0
        direction_accuracy = mean(trend_directions_actual .== trend_directions_pred)
        
        # Calculate volatility reduction
        original_returns = diff(log.(max.(original_prices, 1e-10)))
        denoised_returns = diff(log.(max.(denoised_values, 1e-10)))
        
        original_volatility = std(original_returns)
        denoised_volatility = std(denoised_returns)
        
        # Ensure we don't divide by zero or have negative volatilities
        if original_volatility > 1e-10
            volatility_reduction = 1.0 - (denoised_volatility / original_volatility)
        else
            volatility_reduction = 0.0
            log_message("Warning: Original volatility near zero for $symbol", level="WARNING")
        end
        
        # Calculate metrics commonly used in finance
        # Daily returns
        daily_returns_actual = original_returns
        daily_returns_denoised = denoised_returns
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        sharpe_actual = sqrt(252) * mean(daily_returns_actual) / std(daily_returns_actual)
        sharpe_denoised = sqrt(252) * mean(daily_returns_denoised) / std(daily_returns_denoised)
        
        # Maximum drawdown
        function calc_max_drawdown(prices)
            max_drawdown = 0.0
            peak_value = prices[1]
            
            for price in prices
                if price > peak_value
                    peak_value = price
                else
                    drawdown = (peak_value - price) / peak_value
                    max_drawdown = max(max_drawdown, drawdown)
                end
            end
            
            return max_drawdown
        end
        
        max_drawdown_actual = calc_max_drawdown(original_prices)
        max_drawdown_denoised = calc_max_drawdown(denoised_values)
        
        # Store metrics
        metrics[symbol] = Dict(
            "direction_accuracy" => direction_accuracy,
            "volatility_reduction" => volatility_reduction,
            "sharpe_ratio" => Dict(
                "actual" => sharpe_actual,
                "denoised" => sharpe_denoised
            ),
            "max_drawdown" => Dict(
                "actual" => max_drawdown_actual,
                "denoised" => max_drawdown_denoised
            )
        )
        
        log_message(@sprintf("Metrics for %s: Direction accuracy: %.2f%%, Volatility reduction: %.2f%%", 
                          symbol, direction_accuracy * 100, volatility_reduction * 100))
    end
    
    # Ensure output directory exists
    mkpath(dirname(metrics_file))
    
    # Save metrics to file
    open(metrics_file, "w") do io
        JSON.print(io, metrics, 4)  # 4 spaces of indentation
    end
    
    log_message("Financial metrics saved to: $metrics_file")
    return metrics
end

# Function to generate technical indicators visualization
function create_technical_indicators(mresult, financial_data, dates, symbols, output_dir; symbol_to_plot=nothing, metadata=nothing)
    log_message("Generating technical indicators visualization")
    
    # If no specific symbol selected, use the first one
    if isnothing(symbol_to_plot) && !isempty(symbols)
        symbol_to_plot = symbols[1]
    elseif isnothing(symbol_to_plot)
        log_message("No symbols available for technical indicator visualization", level="WARNING")
        return nothing
    end
    
    log_message("Creating technical indicators for $symbol_to_plot")
    
    set_financial_theme()
    
    # Extract data for selected symbol
    prices = financial_data[symbol_to_plot]
    
    # Extract denoised prices
    predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    symbol_idx = findfirst(s -> s == symbol_to_plot, symbols)
    denoised_prices = [pred[symbol_idx] for pred in predicted_means]
    
    # Calculate technical indicators
    
    # Simple Moving Averages (SMA)
    function calc_sma(prices, window)
        result = similar(prices)
        for i in 1:length(prices)
            if i < window
                result[i] = NaN
            else
                result[i] = mean(prices[i-window+1:i])
            end
        end
        return result
    end
    
    # Relative Strength Index (RSI)
    function calc_rsi(prices, window=14)
        gains = zeros(length(prices))
        losses = zeros(length(prices))
        rsi = similar(prices)
        
        # Calculate gains and losses
        for i in 2:length(prices)
            change = prices[i] - prices[i-1]
            if change > 0
                gains[i] = change
            else
                losses[i] = -change
            end
        end
        
        # Calculate RSI
        for i in 1:length(prices)
            if i <= window
                rsi[i] = NaN
            else
                avg_gain = mean(gains[i-window+1:i])
                avg_loss = mean(losses[i-window+1:i])
                
                if avg_loss == 0
                    rsi[i] = 100.0
                else
                    rs = avg_gain / avg_loss
                    rsi[i] = 100.0 - (100.0 / (1.0 + rs))
                end
            end
        end
        
        return rsi
    end
    
    # MACD (Moving Average Convergence Divergence)
    function calc_macd(prices, fast=12, slow=26, signal=9)
        fast_ema = similar(prices)
        slow_ema = similar(prices)
        macd_line = similar(prices)
        signal_line = similar(prices)
        histogram = similar(prices)
        
        # Simple implementation of EMA
        function calc_ema(prices, window)
            ema = similar(prices)
            multiplier = 2.0 / (window + 1.0)
            
            ema[1] = prices[1]
            for i in 2:length(prices)
                ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
            end
            
            return ema
        end
        
        fast_ema = calc_ema(prices, fast)
        slow_ema = calc_ema(prices, slow)
        
        # Calculate MACD line
        macd_line = fast_ema .- slow_ema
        
        # Calculate signal line
        signal_line = calc_ema(macd_line, signal)
        
        # Calculate histogram
        histogram = macd_line .- signal_line
        
        return (macd_line, signal_line, histogram)
    end
    
    # Calculate indicators for both original and denoised prices
    sma20_orig = calc_sma(prices, 20)
    sma50_orig = calc_sma(prices, 50)
    rsi_orig = calc_rsi(prices)
    macd_orig, signal_orig, hist_orig = calc_macd(prices)
    
    sma20_denoised = calc_sma(denoised_prices, 20)
    sma50_denoised = calc_sma(denoised_prices, 50)
    rsi_denoised = calc_rsi(denoised_prices)
    macd_denoised, signal_denoised, hist_denoised = calc_macd(denoised_prices)
    
    # Create visualization plots
    # Price plot with SMAs
    p1 = plot(dates, prices, 
             label="Original Price",
             color=:lightblue,
             alpha=0.7,
             title="$symbol_to_plot Price with Moving Averages",
             ylabel="Price",
             legend=:topleft)
    
    plot!(p1, dates, denoised_prices,
         label="Denoised Price",
         color=:red,
         linewidth=2)
    
    plot!(p1, dates, sma20_orig,
         label="SMA(20) Original",
         color=:orange,
         linewidth=1.5,
         linestyle=:dash)
    
    plot!(p1, dates, sma50_orig,
         label="SMA(50) Original",
         color=:purple,
         linewidth=1.5,
         linestyle=:dash)
    
    plot!(p1, dates, sma20_denoised,
         label="SMA(20) Denoised",
         color=:orange,
         linewidth=1.5)
    
    plot!(p1, dates, sma50_denoised,
         label="SMA(50) Denoised",
         color=:purple,
         linewidth=1.5)
    
    # Add training/test split line if applicable
    if metadata !== nothing && haskey(metadata, "train_test_split")
        split_idx = metadata["train_test_split"]
        vline!(p1, [dates[split_idx]], 
             label="Train/Test Split", 
             linestyle=:dash, 
             color=:white)
    end
    
    # RSI plot
    p2 = plot(dates, rsi_orig,
             label="RSI Original",
             color=:lightblue,
             alpha=0.7,
             title="Relative Strength Index (RSI)",
             ylabel="RSI",
             ylim=(0, 100))
    
    plot!(p2, dates, rsi_denoised,
         label="RSI Denoised",
         color=:red,
         linewidth=2)
    
    # Add RSI reference lines at 30 and 70
    hline!(p2, [30, 70], color=:gray, linestyle=:dash, label=nothing)
    
    # Add training/test split line if applicable
    if metadata !== nothing && haskey(metadata, "train_test_split")
        split_idx = metadata["train_test_split"]
        vline!(p2, [dates[split_idx]], 
             label="Train/Test Split", 
             linestyle=:dash, 
             color=:white)
    end
    
    # MACD plot
    p3 = plot(dates, macd_orig,
             label="MACD Original",
             color=:lightblue,
             alpha=0.7,
             title="Moving Average Convergence Divergence (MACD)",
             ylabel="MACD",
             legend=:topleft)
    
    plot!(p3, dates, signal_orig,
         label="Signal Original",
         color=:orange,
         alpha=0.7,
         linestyle=:dash)
    
    plot!(p3, dates, macd_denoised,
         label="MACD Denoised",
         color=:red,
         linewidth=2)
    
    plot!(p3, dates, signal_denoised,
         label="Signal Denoised",
         color=:orange,
         linewidth=2)
    
    # Add training/test split line if applicable
    if metadata !== nothing && haskey(metadata, "train_test_split")
        split_idx = metadata["train_test_split"]
        vline!(p3, [dates[split_idx]], 
             label="Train/Test Split", 
             linestyle=:dash, 
             color=:white)
    end
    
    # Combine plots into a single figure
    p = plot(p1, p2, p3, layout=(3,1), size=(800, 1200))
    
    # Save plot to file
    mkpath(output_dir)
    filename = joinpath(output_dir, "$(symbol_to_plot)_technical_indicators.png")
    savefig(p, filename)
    
    log_message("Technical indicators visualization saved to: $filename")
    return filename
end

# Function to export prediction data to CSV (Ported from LVA)
function export_prediction_data(mresult, true_log_returns, return_dates, symbols, output_dir)
    log_message("Saving prediction results (log returns) to CSV")
    n_samples = length(return_dates)
    n_symbols = length(symbols)
    
    # Select a subset of symbols if too many, matching visualization
    symbols_to_export = length(symbols) > 5 ? symbols[1:5] : symbols 
    log_message("Exporting data for symbols: $(join(symbols_to_export, ", "))")

    results_data = Array{Any}(undef, n_samples, 3*length(symbols_to_export) + 1)
    results_data[:, 1] = string.(return_dates)  # Time index as dates

    # Extract required data once
    predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    predicted_covs = cov.(mresult.predictions[:y][end])

    for (i, symbol) in enumerate(symbols_to_export)
        symbol_idx = findfirst(s -> s == symbol, symbols)
        if symbol_idx === nothing; continue; end

        # True log returns
        results_data[:, 3*i-1] = [ret[symbol_idx] for ret in true_log_returns]
        
        # Predicted means (log returns)
        results_data[:, 3*i] = [pred[symbol_idx] for pred in predicted_means]
        
        # Prediction standard deviations (log returns)
        results_data[:, 3*i+1] = [sqrt(predicted_covs[t][symbol_idx, symbol_idx]) for t in 1:n_samples]
    end
    
    # Create header
    header = ["date"]
    for symbol in symbols_to_export
        push!(header, "true_logret_$symbol", "pred_mean_logret_$symbol", "pred_std_logret_$symbol")
    end
    
    # Write to CSV
    csv_filename = joinpath(output_dir, "flvar_predictions.csv")
    open(csv_filename, "w") do io
        writedlm(io, [header], ',')
        writedlm(io, results_data, ',')
    end
    log_message("Prediction data saved to: $csv_filename")
    
    return csv_filename
end

# Export detailed statistics to JSON (Ported from LVA)
function export_detailed_statistics(mresult, true_data, test_metrics, n_train, n_test, n_symbols, symbols, output_dir)
    log_message("Exporting detailed statistics")
    
    # Calculate additional statistics from test_metrics if available
    avg_rmse = haskey(test_metrics, "avg_rmse") ? test_metrics["avg_rmse"] : NaN
    rmse_by_symbol = haskey(test_metrics, "rmse") ? test_metrics["rmse"] : fill(NaN, n_symbols)
    
    median_rmse = median(filter(!isnan, rmse_by_symbol))
    min_rmse = minimum(filter(!isnan, rmse_by_symbol))
    max_rmse = maximum(filter(!isnan, rmse_by_symbol))
    rmse_std = std(filter(!isnan, rmse_by_symbol))
    
    # Calculate average uncertainty (prediction standard deviation) over test set
    n_samples = n_train + n_test
    test_indices = (n_train + 1):n_samples
    avg_uncertainty = try
        mean([mean(sqrt.(diag(cov.(mresult.predictions[:y][end])[t]))) for t in test_indices])
    catch e
        log_message("Could not calculate average uncertainty: $e", level="WARNING")
        NaN
    end
    
    # Prepare statistics in JSON format
    stats = Dict(
        "overall_statistics" => Dict(
            "avg_rmse" => avg_rmse,
            "median_rmse" => median_rmse,
            "min_rmse" => min_rmse,
            "max_rmse" => max_rmse,
            "rmse_std" => rmse_std,
            "avg_uncertainty_test" => avg_uncertainty
        ),
        "symbol_statistics" => Dict(
             symbol => Dict("rmse" => rmse_by_symbol[i]) for (i, symbol) in enumerate(symbols) if !isnan(rmse_by_symbol[i])
        ),
        "model_run_details" => Dict(
            "n_symbols" => n_symbols,
            "n_samples_total" => n_samples,
            "n_samples_test" => n_test,
            "n_samples_train" => n_train
        )
    )
    
    # Write to JSON file
    json_filename = joinpath(output_dir, "model_statistics.json")
    mkpath(dirname(json_filename)) # Ensure directory exists
    open(json_filename, "w") do io
        JSON.print(io, stats, 4) # Pretty print 
    end
    log_message("Detailed statistics saved to: $json_filename")
    
    return json_filename
end

# Create heatmap of prediction errors (Ported from LVA)
function create_error_heatmap(mresult, true_data, return_dates, n_train, n_test, n_symbols, symbols, output_dir)
    log_message("Generating log return error heatmap")
    set_financial_theme()
    n_samples = n_train + n_test

    # Extract predictions and true values (log returns)
    predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    actual_values = getindex.(true_data, :) # true_data should be log returns
    
    # Calculate errors 
    errors = zeros(n_samples, n_symbols)
    for t in 1:n_samples
        for p in 1:n_symbols
             # Ensure indices are valid before accessing
             if t <= length(predicted_means) && p <= length(predicted_means[t]) &&
                t <= length(actual_values) && p <= length(actual_values[t])
                 errors[t, p] = predicted_means[t][p] - actual_values[t][p]
             else
                 errors[t, p] = NaN # Mark as NaN if data is missing
                 log_message("Data mismatch at t=$t, p=$p for error heatmap.", level="WARNING")
             end
        end
    end
    
    # Create heatmap
    p = heatmap(1:n_symbols, 1:n_samples, errors', # Transpose errors for correct orientation
                xticks=(1:n_symbols, symbols), # Use symbols as x-ticks
                yticks=(round.(Int, range(1, n_samples, length=10)), string.(return_dates[round.(Int, range(1, n_samples, length=10))])),
                color=:RdBu, 
                aspect_ratio=:auto,
                xlabel="Symbol",
                ylabel="Time Step (Date)",
                title="Log Return Prediction Error Heatmap (Pred - True)",
                colorbar_title="Error",
                size=(800, 600))
                
    # Add a horizontal line to separate train and test regions
    hline!(p, [n_train + 0.5], color=:white, linewidth=1.5, linestyle=:dash, label="Train-Test Split")
    
    # Save figure
    heatmap_filename = joinpath(output_dir, "logreturn_error_heatmap.png")
    savefig(p, heatmap_filename)
    log_message("Error heatmap saved to: $heatmap_filename")
    
    return heatmap_filename
end

# Create correlation plots between symbols (Ported from LVA)
function create_correlation_plots(mresult, true_data, n_symbols, symbols, output_dir)
    log_message("Generating log return correlation plots")
    set_financial_theme()

    # Extract predicted means (log returns)
    predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    n_samples = length(predicted_means)

    # Calculate correlation matrix of predicted log returns
    pred_data = zeros(n_samples, n_symbols)
    for t in 1:n_samples
        if length(predicted_means[t]) == n_symbols
            pred_data[t, :] = predicted_means[t]
        else
            pred_data[t, :] .= NaN # Handle potential size mismatch
        end
    end
    
    # Remove rows with NaN before calculating correlation
    valid_rows_pred = all(!isnan, pred_data, dims=2)[:]
    if !any(valid_rows_pred); log_message("No valid prediction data for correlation.", level="WARNING"); return nothing; end
    pred_corr = cor(pred_data[valid_rows_pred, :])
    
    # Create true log return matrix
    true_matrix = zeros(length(true_data), n_symbols)
     for t in 1:length(true_data)
        if length(true_data[t]) == n_symbols
            true_matrix[t, :] = true_data[t]
         else
             true_matrix[t, :] .= NaN
         end
    end

    valid_rows_true = all(!isnan, true_matrix, dims=2)[:]
    if !any(valid_rows_true); log_message("No valid true data for correlation.", level="WARNING"); return nothing; end
    true_corr = cor(true_matrix[valid_rows_true, :])
    
    # Create correlation heatmaps
    tick_labels = (1:n_symbols, symbols)
    p1 = heatmap(pred_corr, 
                title="Predicted Log Return Correlations", 
                xlabel="Symbol", ylabel="Symbol",
                xticks=tick_labels, yticks=tick_labels,
                color=:viridis, aspect_ratio=:equal, clim=(-1,1), size=(600, 600))
                
    p2 = heatmap(true_corr, 
                title="True Log Return Correlations", 
                xlabel="Symbol", ylabel="Symbol",
                xticks=tick_labels, yticks=tick_labels,
                color=:viridis, aspect_ratio=:equal, clim=(-1,1), size=(600, 600))
                
    # Create correlation difference heatmap
    p3 = heatmap(pred_corr - true_corr, 
                title="Correlation Difference (Pred - True)", 
                xlabel="Symbol", ylabel="Symbol",
                xticks=tick_labels, yticks=tick_labels,
                color=:RdBu, aspect_ratio=:equal, clim=(-0.5,0.5), size=(600, 600))
    
    # Combine plots
    p = plot(p1, p2, p3, layout=(1,3), size=(1800, 600))
    
    # Save figure
    corr_filename = joinpath(output_dir, "logreturn_correlation_plots.png")
    savefig(p, corr_filename)
    log_message("Correlation plots saved to: $corr_filename")
    
    return corr_filename
end

# Create uncertainty visualization (Ported from LVA)
function create_uncertainty_plots(mresult, true_data, n_train, n_test, n_symbols, output_dir)
    log_message("Generating uncertainty visualization")
    set_financial_theme()
    n_samples = n_train + n_test

    # Calculate the uncertainty (standard deviation) 
    uncertainties = []
    predicted_means = []
    try 
        uncertainties = [sqrt.(diag(cov.(mresult.predictions[:y][end])[t])) for t in 1:n_samples]
        predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    catch e
         log_message("Could not extract predictions/uncertainties: $e", level="ERROR")
         return nothing
    end

    # Flatten for plotting
    all_uncertainties = vcat(uncertainties...)
    
    # Create histogram of uncertainties
    p1 = histogram(all_uncertainties, 
                  bins=30, 
                  title="Distribution of Prediction Uncertainties (Std Dev)",
                  xlabel="Standard Deviation (Log Return)", ylabel="Count",
                  legend=false, alpha=0.7, size=(600, 400))
    
    # Calculate prediction errors for test set
    test_indices = (n_train + 1):n_samples
    actual_values = getindex.(true_data, :) # true_data should be log returns
    
    # Collect errors and corresponding uncertainties for test set
    test_errors = Float64[]
    test_uncertainties = Float64[]
    
    for t_idx in 1:length(test_indices)
        t = test_indices[t_idx]
        if t > length(predicted_means) || t > length(actual_values) || t > length(uncertainties)
             log_message("Index $t out of bounds for uncertainty/error calculation.", level="WARNING")
             continue
        end
        for p in 1:n_symbols
             if p > length(predicted_means[t]) || p > length(actual_values[t]) || p > length(uncertainties[t])
                 log_message("Symbol index $p out of bounds at time $t.", level="WARNING")
                 continue
             end
            err = predicted_means[t][p] - actual_values[t][p]
            unc = uncertainties[t][p]
            push!(test_errors, err)
            push!(test_uncertainties, unc)
        end
    end
    
    # Scatter plot of errors vs uncertainties
    p2 = scatter(test_uncertainties, abs.(test_errors),
                xlabel="Predicted Uncertainty (Std Dev)", ylabel="Absolute Error (Log Return)",
                title="Uncertainty vs. Error (Test Set)",
                legend=false, alpha=0.4, size=(600, 400))
    
    # Add a trend line if possible
    if length(test_uncertainties) > 1
        try
            X = [ones(length(test_uncertainties)) test_uncertainties]
            valid_indices = findall(isfinite.(abs.(test_errors))) # Handle potential NaNs/Infs
            if length(valid_indices) > 1
                β = X[valid_indices,:] \ abs.(test_errors[valid_indices])
                x_range = range(minimum(filter(isfinite, test_uncertainties)), maximum(filter(isfinite, test_uncertainties)), length=100)
                y_pred = [β[1] + β[2]*x for x in x_range]
                plot!(p2, x_range, y_pred, linewidth=2, color=:orange, label="Trend")
                
                corr_val = cor(test_uncertainties[valid_indices], abs.(test_errors[valid_indices]))
                annotate!(p2, maximum(filter(isfinite, test_uncertainties))*0.8, maximum(filter(isfinite, abs.(test_errors)))*0.9, 
                         text(@sprintf("Corr: %.3f", corr_val), 10))
            end
        catch e
             log_message("Could not compute trend line for uncertainty plot: $e", level="WARNING")
        end
    end
    
    # Combine plots
    p = plot(p1, p2, layout=(1,2), size=(1200, 400))
    
    # Save figure
    uncertainty_filename = joinpath(output_dir, "uncertainty_analysis.png")
    savefig(p, uncertainty_filename)
    log_message("Uncertainty plots saved to: $uncertainty_filename")
    
    return uncertainty_filename
end


# Create residual analysis plots (Ported from LVA)
function create_residual_analysis(mresult, true_data, n_train, n_test, n_symbols, output_dir)
    log_message("Generating residual analysis plots")
    set_financial_theme()
    n_samples = n_train + n_test

    # Calculate residuals for test set
    test_indices = (n_train + 1):n_samples
    residuals = Float64[]
    
    try
        predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
        actual_values = getindex.(true_data, :) # true_data should be log returns

        for t_idx in 1:length(test_indices)
            t = test_indices[t_idx]
            if t > length(predicted_means) || t > length(actual_values); continue; end
            for p in 1:n_symbols
                if p > length(predicted_means[t]) || p > length(actual_values[t]); continue; end
                res = predicted_means[t][p] - actual_values[t][p]
                if isfinite(res); push!(residuals, res); end
            end
        end
    catch e
        log_message("Could not calculate residuals: $e", level="ERROR")
        return nothing
    end

    if isempty(residuals)
        log_message("No valid residuals found for analysis.", level="WARNING")
        return nothing
    end

    # Create histogram of residuals
    p1 = histogram(residuals, 
                  bins=30, normalize=:pdf,
                  title="Distribution of Residuals (Test Set, Log Returns)",
                  xlabel="Residual (Pred - True)", ylabel="Density",
                  label="Residuals", alpha=0.7, size=(600, 400))
    
    # Add a normal distribution fit
    μ = mean(residuals); σ = std(residuals)
    x_range = range(minimum(residuals), maximum(residuals), length=100)
    y_normal = pdf.(Normal(μ, σ), x_range)
    plot!(p1, x_range, y_normal, linewidth=2, color=:orange, label="Normal Fit")
    
    # Create QQ plot
    # Calculate quantiles manually for QQ plot
    sort_residuals = sort(residuals)
    n_res = length(residuals)
    theo_quantiles = quantile.(Normal(μ, σ), (1:n_res) ./ (n_res + 1))
    
    p2 = scatter(theo_quantiles, sort_residuals,
                xlabel="Theoretical Normal Quantiles", ylabel="Sample Quantiles",
                title="Q-Q Plot of Residuals vs Normal",
                legend=false, size=(600, 400))
    # Add reference line
    plot!(p2, theo_quantiles, theo_quantiles, color=:orange, linestyle=:dash, label="y=x")

    # Combine plots
    p = plot(p1, p2, layout=(1,2), size=(1200, 400))
    
    # Save figure
    residual_filename = joinpath(output_dir, "residual_analysis.png")
    savefig(p, residual_filename)
    log_message("Residual analysis plots saved to: $residual_filename")
    
    return residual_filename
end


# Combined function to handle all enhanced visualization and data export (New Wrapper)
function visualize_and_export_all(mresult, true_data, return_dates, symbols, n_train, n_test, test_metrics, output_dir, metadata=nothing)
    
    log_message("Starting comprehensive visualization and data export process")
    mkpath(output_dir) # Ensure base output dir exists
    
    n_symbols = length(symbols)
    
    results = Dict{String, Any}(
        "output_dir" => output_dir
    )
    
    # Use try-catch blocks for robustness
    
    # Basic Log Return Plots
    try
        results["logreturn_plots"] = create_logreturn_visualization(mresult, true_data, return_dates, symbols, output_dir, metadata)
    catch e; log_message("Error generating log return plots: $e", level="ERROR"); end
    
    # Error heatmap
    try
        results["error_heatmap"] = create_error_heatmap(mresult, true_data, return_dates, n_train, n_test, n_symbols, symbols, output_dir)
    catch e; log_message("Error generating error heatmap: $e", level="ERROR"); end
    
    # Correlation plots
    try
        results["correlation_plots"] = create_correlation_plots(mresult, true_data, n_symbols, symbols, output_dir)
    catch e; log_message("Error generating correlation plots: $e", level="ERROR"); end
    
    # Uncertainty plots
    try
        results["uncertainty_plots"] = create_uncertainty_plots(mresult, true_data, n_train, n_test, n_symbols, output_dir)
    catch e; log_message("Error generating uncertainty plots: $e", level="ERROR"); end
    
    # Residual analysis
    try
        results["residual_plot"] = create_residual_analysis(mresult, true_data, n_train, n_test, n_symbols, output_dir)
    catch e; log_message("Error generating residual analysis: $e", level="ERROR"); end
    
    # --- Data Exports ---
    
    # Export prediction data to CSV
    try
        results["csv_file"] = export_prediction_data(mresult, true_data, return_dates, symbols, output_dir)
    catch e; log_message("Error exporting prediction data to CSV: $e", level="ERROR"); end
    
    # Export detailed statistics as JSON
    try
        # Ensure test_metrics is passed correctly
        results["stats_file"] = export_detailed_statistics(mresult, true_data, test_metrics, n_train, n_test, n_symbols, symbols, output_dir)
    catch e; log_message("Error exporting detailed statistics: $e", level="ERROR"); end
    
    log_message("All visualizations and data exports completed for FLVAR")
    
    return results
end


# Export all the visualization functions
export set_financial_theme, create_candlestick_chart, create_logreturn_visualization,
       calculate_financial_metrics, create_technical_indicators,
       # Newly added/ported functions
       export_prediction_data, export_detailed_statistics, create_error_heatmap,
       create_correlation_plots, create_uncertainty_plots, create_residual_analysis,
       visualize_and_export_all

end # module