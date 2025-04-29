module FinancialDataSources

using HTTP, JSON, Dates, Random, Statistics, DelimitedFiles, Printf
using LinearAlgebra: diag, cholesky, eigen, I

export get_financial_data, generate_synthetic_financial_data, generate_synthetic_log_returns

# Function to log messages
function log_message(message; level="INFO")
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    println("[$timestamp] [$level] $message")
end

"""
    get_financial_data(symbols, start_date, end_date; api_key=nothing, data_source="synthetic_returns")

Retrieve financial data for the given symbols from the specified data source.
If data_source is "synthetic", generate synthetic data instead of fetching from an API.

# Arguments
- `symbols`: Array of ticker symbols to fetch
- `start_date`: Start date in "YYYY-MM-DD" format
- `end_date`: End date in "YYYY-MM-DD" format
- `api_key`: API key for the data source (optional)
- `data_source`: "synthetic", "alphavantage", or "yahoo" (default: "synthetic_returns")

# Returns
- `data`: Dictionary with symbol keys and time series values
- `dates`: Array of dates corresponding to the data points
"""
function get_financial_data(symbols, start_date, end_date; 
                           api_key=nothing, data_source="synthetic_returns")
    
    if data_source == "synthetic"
        log_message("Generating synthetic financial PRICE data for $(length(symbols)) symbols")
        return generate_synthetic_financial_data(symbols, start_date, end_date)
    elseif data_source == "synthetic_returns"
        log_message("Generating synthetic LOG RETURN data for $(length(symbols)) symbols")
        return generate_synthetic_log_returns(symbols, start_date, end_date)
    elseif data_source == "alphavantage"
        if isnothing(api_key)
            error("API key is required for Alpha Vantage data source")
        end
        return fetch_alphavantage_data(symbols, start_date, end_date, api_key)
    elseif data_source == "yahoo"
        return fetch_yahoo_data(symbols, start_date, end_date)
    else
        error("Unsupported data source: $data_source")
    end
end

"""
    generate_synthetic_financial_data(symbols, start_date, end_date)

Generate synthetic financial time series data with realistic properties.

# Arguments
- `symbols`: Array of ticker symbols to generate data for
- `start_date`: Start date in "YYYY-MM-DD" format
- `end_date`: End date in "YYYY-MM-DD" format

# Returns
- `data`: Dictionary with symbol keys and time series values
- `dates`: Array of dates corresponding to the data points
"""
function generate_synthetic_financial_data(symbols, start_date, end_date)
    # Parse dates
    start = Date(start_date)
    end_date = Date(end_date)
    
    # Generate business days between start and end dates
    dates = collect(start:Day(1):end_date)
    # Filter to keep only business days (Mon-Fri)
    dates = filter(date -> Dates.dayofweek(date) ∉ [6, 7], dates)
    
    n_days = length(dates)
    log_message("Generating $n_days days of synthetic data from $start_date to $end_date")
    
    # Parameters for synthetic data generation
    n_assets = length(symbols)
    
    # Define correlation structure between assets
    # Create a valid correlation matrix
    Random.seed!(42) # For reproducibility
    
    # Create a more robust positive definite correlation matrix
    # Using the nearest correlation matrix approach
    base_corr = rand(n_assets, n_assets) * 0.5  # Lower correlations
    base_corr = (base_corr + base_corr') / 2    # Make it symmetric
    # Add diagonal dominance to ensure positive definiteness
    for i in 1:n_assets
        base_corr[i,i] = 1.0  # Set diagonal to 1
    end
    
    # Ensure positive definiteness by adding a small constant to eigenvalues if needed
    eigen_vals = eigen(base_corr).values
    min_eigen = minimum(eigen_vals)
    if min_eigen <= 0
        # Add a small value to the diagonal to make it positive definite
        base_corr += (abs(min_eigen) + 0.01) * I(n_assets)
        # Rescale to make sure diagonal is 1
        for i in 1:n_assets
            for j in 1:n_assets
                if i != j
                    base_corr[i,j] = base_corr[i,j] / sqrt(base_corr[i,i] * base_corr[j,j])
                end
            end
        end
        # Reset diagonal to 1
        for i in 1:n_assets
            base_corr[i,i] = 1.0
        end
    end
    
    correlation_matrix = base_corr
    
    # Parameters for different types of financial series
    # Each asset can behave differently
    asset_params = []
    for i in 1:n_assets
        # Randomly choose parameters for this asset
        drift = rand() * 0.001 - 0.0003  # Small daily drift, slightly negative bias
        volatility = 0.005 + rand() * 0.02  # Daily volatility between 0.5% and 2.5%
        jump_prob = rand() * 0.05        # Jump probability up to 5% per day
        jump_size_mean = 0               # Jump mean (can be 0 for symmetric jumps)
        jump_size_std = volatility * 5   # Jump size related to volatility
        
        # Mean reversion or momentum parameters
        mean_reversion = rand() > 0.7    # 30% chance of being mean-reverting
        reversion_strength = mean_reversion ? 0.05 + rand() * 0.15 : -0.05 - rand() * 0.1
        
        # Seasonal/cyclical component parameters
        has_seasonality = rand() > 0.7   # 30% chance of having seasonality
        seasonal_period = has_seasonality ? rand([5, 10, 20, 60]) : 0  # Different periods (week, 2 weeks, month, quarter)
        seasonal_amplitude = has_seasonality ? volatility * (0.5 + rand()) : 0
        
        # Volatility clustering parameters
        vol_clustering = rand() > 0.3    # 70% chance of having volatility clustering
        vol_persistence = vol_clustering ? 0.7 + rand() * 0.25 : 0  # How persistent is volatility
        
        push!(asset_params, (
            drift = drift,
            volatility = volatility,
            jump_prob = jump_prob,
            jump_size_mean = jump_size_mean,
            jump_size_std = jump_size_std,
            mean_reversion = mean_reversion,
            reversion_strength = reversion_strength,
            has_seasonality = has_seasonality,
            seasonal_period = seasonal_period,
            seasonal_amplitude = seasonal_amplitude,
            vol_clustering = vol_clustering,
            vol_persistence = vol_persistence
        ))
    end
    
    # Initialize data structure
    data = Dict{String, Vector{Float64}}()
    for symbol in symbols
        data[symbol] = ones(n_days)  # Start at price 1.0
    end
    
    # Generate time series with correlated noise
    for t in 2:n_days
        # Generate correlated random numbers
        Z = randn(n_assets)  # Standard normal samples
        # Apply correlation structure (Cholesky decomposition would be more efficient)
        L = cholesky(correlation_matrix).L
        eps = L * Z  # Correlated noise
        
        for (i, symbol) in enumerate(symbols)
            params = asset_params[i]
            price = data[symbol][t-1]
            
            # Current volatility (for volatility clustering)
            current_vol = params.volatility
            if params.vol_clustering && t > 2
                # GARCH-like effect: volatility depends on previous return
                prev_return = log(data[symbol][t-1] / data[symbol][t-2])
                current_vol = sqrt((1 - params.vol_persistence) * params.volatility^2 + 
                                   params.vol_persistence * prev_return^2)
            end
            
            # Mean reversion / momentum component
            reversion_component = 0.0
            if t > 5  # Need some history for this
                # Calculate recent average
                recent_avg = mean(data[symbol][max(1, t-20):t-1])
                # Mean reversion pulls toward average, momentum pushes away
                reversion_component = params.reversion_strength * (recent_avg - price) / price
            end
            
            # Seasonal component
            seasonal_component = 0.0
            if params.has_seasonality
                seasonal_component = params.seasonal_amplitude * 
                                    sin(2π * t / params.seasonal_period)
            end
            
            # Jump component
            jump_component = 0.0
            if rand() < params.jump_prob
                jump_component = randn() * params.jump_size_std + params.jump_size_mean
            end
            
            # Combine all components for the log return
            log_return = params.drift + reversion_component + seasonal_component + 
                        jump_component + current_vol * eps[i]
            
            # Update price
            data[symbol][t] = price * exp(log_return)
        end
    end

    # Ensure all data starts at reasonably different price levels
    # This is just for better visualization
    for (i, symbol) in enumerate(symbols)
        base_price = 50.0 + i * 20.0  # Different starting prices
        data[symbol] = data[symbol] .* base_price
    end
    
    log_message("Successfully generated synthetic financial data for $(length(symbols)) symbols")
    
    return data, dates
end

"""
    generate_synthetic_log_returns(symbols, start_date, end_date; orders=5, noise_level=0.1)

Generate synthetic financial log returns based on a simple VAR process, analogous to LVAR.

# Arguments
- `symbols`: Array of ticker symbols (used for keys in the output dict)
- `start_date`: Start date in "YYYY-MM-DD" format
- `end_date`: End date in "YYYY-MM-DD" format
- `orders`: The order P for the VAR(P) process (can be a single Int or a vector)
- `noise_level`: Standard deviation of the observation noise added to the true log returns.

# Returns
- `true_log_returns`: Dictionary {symbol: [log_returns...]} representing the underlying VAR process.
- `observations`: Dictionary {symbol: [observed_log_returns...]} with added noise.
- `dates`: Array of dates corresponding to the log return points (N-1 dates).
"""
function generate_synthetic_log_returns(symbols, start_date, end_date; orders=5, noise_level=0.1)
    start = Date(start_date)
    end_dt = Date(end_date)
    
    # Generate business days (these correspond to PRICE dates)
    price_dates = collect(start:Day(1):end_dt)
    price_dates = filter(date -> Dates.dayofweek(date) ∉ [6, 7], price_dates)
    n_price_days = length(price_dates)
    
    if n_price_days < 2
        error("Need at least 2 price days to generate log returns.")
    end
    
    # Log return dates start from the second price date
    return_dates = price_dates[2:end]
    n_samples = length(return_dates) # Number of log return samples
    log_message("Generating $n_samples days of synthetic log returns from $(return_dates[1]) to $(return_dates[end])")

    n_assets = length(symbols)
    
    # Determine AR orders for each process
    if isa(orders, Int)
        ar_orders = fill(orders, n_assets)
    elseif length(orders) == n_assets
        ar_orders = orders
    else
        error("Length of orders must match number of symbols or be a single integer.")
    end
    max_order = maximum(ar_orders)

    # --- Generate Stable VAR(P) parameters --- 
    # Simplified approach: Generate stable parameters for individual AR processes
    # and assume diagonal transition matrix for simplicity (no cross-asset dependencies in this synthetic data)
    # This matches the structure assumed by the LVAR/FLVAR model itself.
    Random.seed!(42) # For reproducibility
    Theta = [zeros(ar_orders[k]) for k in 1:n_assets] # List of AR coefficient vectors
    for k in 1:n_assets
        # Generate stable AR parameters (simple method with decreasing coeffs)
        Theta[k] = 0.5 .^ (1:ar_orders[k]) 
    end

    # --- Simulate the VAR process (as independent AR processes) --- 
    log_returns_matrix = zeros(n_samples, n_assets)
    
    # Initialize first max_order points with noise
    for t in 1:max_order
        log_returns_matrix[t, :] = randn(n_assets) * 0.01 # Small initial noise
    end

    # Generate subsequent points
    for t in (max_order + 1):n_samples
        for k in 1:n_assets
            order = ar_orders[k]
            if t > order # Ensure we have enough past data
                past_values = log_returns_matrix[t-order : t-1, k]
                 # AR equation: x[t] = sum(θ[i] * x[t-i] for i=1:p) + noise
                log_returns_matrix[t, k] = sum(Theta[k][i] * past_values[order - i + 1] for i in 1:order) + randn() * 0.01 # Process noise
            else 
                 # If t <= order, just use noise (already initialized)
            end
        end
    end

    # --- Convert to dictionary format and add observation noise --- 
    true_log_returns_dict = Dict{String, Vector{Float64}}()
    observed_log_returns_dict = Dict{String, Vector{Float64}}()
    
    for (i, symbol) in enumerate(symbols)
        true_returns = log_returns_matrix[:, i]
        true_log_returns_dict[symbol] = true_returns
        observed_log_returns_dict[symbol] = true_returns .+ randn(n_samples) .* noise_level
    end

    log_message("Successfully generated synthetic log returns for $(length(symbols)) symbols")
    
    # Return true returns, observed returns, and the dates corresponding to the returns
    return true_log_returns_dict, observed_log_returns_dict, return_dates
end

"""
    fetch_alphavantage_data(symbols, start_date, end_date, api_key)

Fetch financial data from Alpha Vantage API.

# Arguments
- `symbols`: Array of ticker symbols to fetch
- `start_date`: Start date in "YYYY-MM-DD" format
- `end_date`: End date in "YYYY-MM-DD" format
- `api_key`: Alpha Vantage API key

# Returns
- `data`: Dictionary with symbol keys and time series values
- `dates`: Array of dates corresponding to the data points
"""
function fetch_alphavantage_data(symbols, start_date, end_date, api_key)
    data = Dict{String, Vector{Float64}}()
    all_dates = Set{Date}()
    
    for symbol in symbols
        log_message("Fetching Alpha Vantage data for $symbol")
        
        url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED" *
              "&symbol=$symbol&apikey=$api_key&outputsize=full"
        
        try
            response = HTTP.get(url)
            parsed = JSON.parse(String(response.body))
            
            if haskey(parsed, "Error Message")
                log_message("Error from Alpha Vantage: $(parsed["Error Message"])", level="ERROR")
                continue
            end
            
            time_series = parsed["Time Series (Daily)"]
            symbol_data = Dict{Date, Float64}()
            
            for (date_str, values) in time_series
                date = Date(date_str)
                if start_date <= date_str <= end_date
                    close_price = parse(Float64, values["4. close"])
                    symbol_data[date] = close_price
                    push!(all_dates, date)
                end
            end
            
            # Convert to arrays later after we have all dates
            data[symbol] = symbol_data
            
        catch e
            log_message("Error fetching data for $symbol: $e", level="ERROR")
        end
    end
    
    # Convert all_dates to a sorted array
    sorted_dates = sort(collect(all_dates))
    
    # Convert each symbol's data to an array aligned with sorted_dates
    for symbol in symbols
        if haskey(data, symbol) && data[symbol] isa Dict
            symbol_dict = data[symbol]
            symbol_array = Float64[]
            
            for date in sorted_dates
                if haskey(symbol_dict, date)
                    push!(symbol_array, symbol_dict[date])
                else
                    # Use the previous value or NaN if no previous value
                    prev_value = isempty(symbol_array) ? NaN : symbol_array[end]
                    push!(symbol_array, prev_value)
                end
            end
            
            data[symbol] = symbol_array
        else
            # If we didn't get data for this symbol, fill with NaNs
            data[symbol] = fill(NaN, length(sorted_dates))
        end
    end
    
    return data, sorted_dates
end

"""
    fetch_yahoo_data(symbols, start_date, end_date)

Fetch financial data from Yahoo Finance (simplified implementation).
This is a placeholder. In a real implementation, you'd use a proper Yahoo Finance API.

# Arguments
- `symbols`: Array of ticker symbols to fetch
- `start_date`: Start date in "YYYY-MM-DD" format
- `end_date`: End date in "YYYY-MM-DD" format

# Returns
- `data`: Dictionary with symbol keys and time series values
- `dates`: Array of dates corresponding to the data points
"""
function fetch_yahoo_data(symbols, start_date, end_date)
    log_message("Yahoo Finance API not implemented. Using synthetic data instead.", level="WARNING")
    return generate_synthetic_financial_data(symbols, start_date, end_date)
end

end # module 