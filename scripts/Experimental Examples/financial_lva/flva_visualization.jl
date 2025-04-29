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
function create_price_visualization(mresult, financial_data, dates, symbols, output_dir, metadata=nothing)
    log_message("Generating price visualization plots")
    
    set_financial_theme()
    
    # Create output directory if it doesn't exist
    mkpath(output_dir)
    
    # Extract predictions
    predicted_means = getindex.(mean.(mresult.predictions[:y][end]), :)
    predicted_stds = [sqrt.(diag(cov.(mresult.predictions[:y][end])[i])) for i in 1:length(predicted_means)]
    
    # Plotting options
    ribbon_alpha = 0.3
    marker_alpha = 0.7
    marker_size = 3
    
    # Create a plot for each symbol
    plot_filenames = []
    
    for (i, symbol) in enumerate(symbols)
        p = plot(size=(800, 500), 
                title="$symbol: Denoised Price Trend",
                xlabel="Date",
                ylabel="Price",
                legend=:topleft)
        
        # Get actual price data
        original_prices = financial_data[symbol]
        
        # Plot original price
        plot!(p, dates, original_prices, 
             label="Actual Price",
             color=:lightblue,
             alpha=0.7)
        
        # Plot denoised prediction with uncertainty
        pred_values = [pred[i] for pred in predicted_means]
        pred_stds = [std[i] for std in predicted_stds]
        
        plot!(p, dates, pred_values,
             ribbon=pred_stds,
             fillalpha=ribbon_alpha,
             label="Denoised Trend",
             color=:red,
             linewidth=2)
        
        # Add training/test split line if applicable
        if metadata !== nothing && haskey(metadata, "train_test_split")
            split_idx = metadata["train_test_split"]
            vline!(p, [dates[split_idx]], 
                 label="Train/Test Split", 
                 linestyle=:dash, 
                 color=:white)
        end
        
        # Save individual plot to file
        plot_filename = joinpath(output_dir, "$(symbol)_denoised.png")
        savefig(p, plot_filename)
        push!(plot_filenames, plot_filename)
        
        log_message("Created denoised visualization for $symbol")
    end
    
    log_message("All price visualization plots saved to: $output_dir")
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

# Export all the visualization functions
export set_financial_theme, create_candlestick_chart, create_price_visualization,
       calculate_financial_metrics, create_technical_indicators

end # module