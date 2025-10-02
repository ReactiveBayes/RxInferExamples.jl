# Utilities module for Coin Toss research fork
# Logging, data export, statistical analysis, and helper functions

module CoinTossUtils

using Logging
using Dates
using DataFrames
using CSV
using JSON
using Statistics
using ProgressMeter
using Printf
using Distributions: Normal, quantile

export setup_logging, Timer, export_to_csv, export_to_json, 
       save_experiment_results, ensure_directories, ProgressBar,
       log_dict, format_time, format_bytes, compute_summary_statistics, 
       bernoulli_confidence_interval, update!, finish!, flatten_dict, elapsed_time

"""
    setup_logging(; verbose::Bool=true, structured::Bool=false, 
                   performance::Bool=false, log_file::Union{String, Nothing}=nothing)

Configure logging system with multiple output formats.

# Arguments
- `verbose::Bool`: Enable detailed console logging
- `structured::Bool`: Enable structured JSON logging to file
- `performance::Bool`: Enable performance metrics logging to CSV
- `log_file::String`: Optional log file path
"""
# Global file handle for logging
const LOG_FILE_HANDLE = Ref{Union{IO, Nothing}}(nothing)

function setup_logging(; verbose::Bool=true, structured::Bool=false, 
                        performance::Bool=false, log_file::Union{String, Nothing}=nothing)
    # Set console log level
    log_level = verbose ? Logging.Info : Logging.Warn
    global_logger(ConsoleLogger(stderr, log_level))
    
    # Setup file logging if requested
    if log_file !== nothing
        # Ensure directory exists
        log_dir = dirname(log_file)
        if !isdir(log_dir) && log_dir != ""
            mkpath(log_dir)
        end
        
        # Open file for writing and store handle
        LOG_FILE_HANDLE[] = open(log_file, "w")
        
        # Create custom logger that writes to both console and file
        # For now, we'll use a simple approach: redirect with tee-like behavior
        # Note: This is a simplified approach; production code might use LoggingExtras.jl
    end
    
    @info "Logging configured" verbose=verbose structured=structured performance=performance
    
    # If file logging enabled, also write to file
    if LOG_FILE_HANDLE[] !== nothing
        println(LOG_FILE_HANDLE[], "Logging configured: verbose=$verbose, structured=$structured, performance=$performance")
        flush(LOG_FILE_HANDLE[])
    end
end

# Custom logging macro that writes to both console and file
macro logboth(msg)
    quote
        @info $(esc(msg))
        if LOG_FILE_HANDLE[] !== nothing
            println(LOG_FILE_HANDLE[], $(esc(msg)))
            flush(LOG_FILE_HANDLE[])
        end
    end
end

"""
    Timer

Utility for timing code blocks with automatic logging.
"""
mutable struct Timer
    name::String
    start_time::Float64
    end_time::Union{Float64, Nothing}
    
    Timer(name::String) = new(name, time(), nothing)
end

"""
    close(timer::Timer)

Stop timer and log elapsed time.
"""
function Base.close(timer::Timer)
    timer.end_time = time()
    elapsed = timer.end_time - timer.start_time
    @info "PERF $(timer.name)" elapsed_seconds=round(elapsed, digits=4)
    return elapsed
end

"""
    elapsed_time(timer::Timer)

Get elapsed time (works for both running and stopped timers).
"""
function elapsed_time(timer::Timer)
    end_t = timer.end_time === nothing ? time() : timer.end_time
    return end_t - timer.start_time
end

"""
    ProgressBar

Simple progress bar wrapper.
"""
struct ProgressBar
    progress::Progress
    
    function ProgressBar(total::Int; desc::String="Progress")
        return new(Progress(total; desc=desc, dt=0.1))
    end
end

"""
    update!(pb::ProgressBar, n::Int)

Update progress bar to step n.
"""
function update!(pb::ProgressBar, n::Int)
    update!(pb.progress, n)
end

"""
    finish!(pb::ProgressBar)

Finish and close progress bar.
"""
function finish!(pb::ProgressBar)
    finish!(pb.progress)
end

"""
    export_to_csv(data::Dict, filepath::String)

Export dictionary to CSV file (flattening nested structures).
"""
function export_to_csv(data::Dict, filepath::String)
    # Ensure directory exists
    dir = dirname(filepath)
    if !isdir(dir) && dir != ""
        mkpath(dir)
    end
    
    # Flatten dictionary and convert to DataFrame
    flat_data = flatten_dict(data)
    
    # Convert to DataFrame with single row
    df = DataFrame(
        key = collect(keys(flat_data)),
        value = [string(v) for v in values(flat_data)]
    )
    
    CSV.write(filepath, df)
    @info "Exported data to CSV" filepath=filepath
end

"""
    export_to_json(data::Dict, filepath::String; indent::Int=2)

Export dictionary to JSON file.
"""
function export_to_json(data::Dict, filepath::String; indent::Int=2)
    # Ensure directory exists
    dir = dirname(filepath)
    if !isdir(dir) && dir != ""
        mkpath(dir)
    end
    
    # Write JSON with pretty printing
    open(filepath, "w") do io
        JSON.print(io, data, indent)
    end
    
    @info "Exported data to JSON" filepath=filepath
end

"""
    flatten_dict(d::Dict, prefix::String="")

Recursively flatten nested dictionary with dot notation.
"""
function flatten_dict(d::Dict, prefix::String="")
    result = Dict{String, Any}()
    
    for (key, value) in d
        full_key = prefix == "" ? string(key) : "$(prefix).$(key)"
        
        if isa(value, Dict)
            merge!(result, flatten_dict(value, full_key))
        elseif isa(value, Array) && !isempty(value) && isa(value[1], Dict)
            # Handle array of dicts
            for (i, item) in enumerate(value)
                merge!(result, flatten_dict(item, "$(full_key)[$i]"))
            end
        else
            result[full_key] = value
        end
    end
    
    return result
end

"""
    save_experiment_results(experiment_name::String, results::Dict)

Save comprehensive experiment results with timestamped directory.

# Returns
- String: Path to results directory
"""
function save_experiment_results(experiment_name::String, results::Dict)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    results_dir = joinpath("outputs", "results", "$(experiment_name)_$(timestamp)")
    mkpath(results_dir)
    
    # Save as JSON
    json_path = joinpath(results_dir, "results.json")
    export_to_json(results, json_path)
    
    # Save as CSV (flattened)
    csv_path = joinpath(results_dir, "results.csv")
    export_to_csv(results, csv_path)
    
    # Save metadata
    metadata = Dict(
        "experiment_name" => experiment_name,
        "timestamp" => timestamp,
        "julia_version" => string(VERSION)
    )
    metadata_path = joinpath(results_dir, "metadata.json")
    export_to_json(metadata, metadata_path)
    
    @info "Saved experiment results" results_dir=results_dir
    
    return results_dir
end

"""
    ensure_directories(config::Dict)

Ensure all output directories specified in config exist.
"""
function ensure_directories(config::Dict)
    if !haskey(config, "output")
        @warn "No output configuration found"
        return
    end
    
    output_config = config["output"]
    
    for (key, path) in output_config
        if isa(path, String) && endswith(key, "_dir")
            if !isdir(path)
                mkpath(path)
                @info "Created directory" path=path
            end
        end
    end
end

"""
    log_dict(dict::Dict; prefix::String="")

Log dictionary contents in a structured format.
"""
function log_dict(dict::Dict; prefix::String="")
    for (key, value) in dict
        full_key = prefix == "" ? string(key) : "$(prefix).$(key)"
        
        if isa(value, Dict)
            log_dict(value, prefix=full_key)
        else
            @info full_key value=value
        end
    end
end

"""
    format_time(seconds::Float64)

Format time duration in human-readable format.
"""
function format_time(seconds::Float64)
    if seconds < 60
        return @sprintf("%.2fs", seconds)
    elseif seconds < 3600
        minutes = floor(seconds / 60)
        secs = seconds - minutes * 60
        return @sprintf("%dm %.2fs", minutes, secs)
    else
        hours = floor(seconds / 3600)
        remainder = seconds - hours * 3600
        minutes = floor(remainder / 60)
        secs = remainder - minutes * 60
        return @sprintf("%dh %dm %.2fs", hours, minutes, secs)
    end
end

"""
    format_bytes(bytes::Int)

Format byte count in human-readable format.
"""
function format_bytes(bytes::Int)
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(bytes)
    unit_idx = 1
    
    while size >= 1024 && unit_idx < length(units)
        size /= 1024
        unit_idx += 1
    end
    
    return @sprintf("%.2f %s", size, units[unit_idx])
end

"""
    compute_summary_statistics(data::Vector{Float64})

Compute comprehensive summary statistics for a data vector.
"""
function compute_summary_statistics(data::Vector{Float64})
    return Dict(
        "mean" => mean(data),
        "median" => median(data),
        "std" => std(data),
        "var" => var(data),
        "min" => minimum(data),
        "max" => maximum(data),
        "q25" => quantile(data, 0.25),
        "q75" => quantile(data, 0.75),
        "n" => length(data)
    )
end

"""
    bernoulli_confidence_interval(n_successes::Int, n_trials::Int; confidence::Float64=0.95)

Compute Wilson score confidence interval for Bernoulli parameter.
"""
function bernoulli_confidence_interval(n_successes::Int, n_trials::Int; confidence::Float64=0.95)
    p̂ = n_successes / n_trials
    z = quantile(Normal(), 1 - (1 - confidence) / 2)
    
    denominator = 1 + z^2 / n_trials
    center = (p̂ + z^2 / (2 * n_trials)) / denominator
    margin = z * sqrt(p̂ * (1 - p̂) / n_trials + z^2 / (4 * n_trials^2)) / denominator
    
    return (
        lower = max(0.0, center - margin),
        upper = min(1.0, center + margin),
        estimate = p̂
    )
end

end # module CoinTossUtils

