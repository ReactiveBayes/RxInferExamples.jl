# Utilities module for Active Inference Mountain Car example
# Provides enhanced logging, data export, performance metrics, and helper functions

@doc """
Utilities module for the Active Inference Mountain Car example.

This module provides:
- Enhanced logging with structured output
- Data export capabilities (CSV, JSON, structured logs)
- Performance metrics and profiling
- Statistical analysis utilities
- File I/O helpers
- Timing and benchmarking tools
"""
module Utils

using Logging
using Logging: SimpleLogger, ConsoleLogger, MultiLogger
using Dates
using JSON
using DataFrames
using CSV
using Statistics
using Printf
import ..Config: PHYSICS, WORLD, TARGET, AGENT, OUTPUTS, NUMERICAL

# Global timing variables for performance tracking
const TIMING_DATA = Dict{String, Dict{String, Any}}()

@doc """
Enhanced logging setup with multiple output formats and structured data.

Args:
- verbose: Enable verbose logging
- structured: Enable structured JSON logging
- performance: Enable performance logging
- filename: Log file name (optional)
"""
function setup_logging(; verbose::Bool = false,
                      structured::Bool = false,
                      performance::Bool = false,
                      filename::Union{String, Nothing} = nothing)

    # Set default filename if not provided
    log_file = filename !== nothing ? filename : OUTPUTS.log_file

    # Create output directory if it doesn't exist
    output_dir = OUTPUTS.output_dir
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    # Configure logging level
    level = verbose ? Logging.Info : Logging.Warn

    # Create formatters
    console_formatter = verbose ?
        (level, message, _module, group, id, file, line) ->
            "$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS")) [$level] $(_module): $message" :
        (level, message, _module, group, id, file, line) ->
            "$message"

    # Setup console logger
    console_logger = ConsoleLogger(stderr, level)

    # Setup file logger if file is specified
    file_logger = if !isnothing(log_file)
        try
            SimpleLogger(open(log_file, "w"), level)
        catch e
            @warn "Could not create log file: $e"
            nothing
        end
    else
        nothing
    end

    # Setup structured logger
    structured_logger = if structured
        try
            structured_file = OUTPUTS.structured_log_file
            structured_io = open(structured_file, "w")
            StructuredLogger(structured_io, level)
        catch e
            @warn "Could not create structured log file: $e"
            nothing
        end
    else
        nothing
    end

    # Setup performance logger
    performance_logger = if performance
        try
            performance_file = OUTPUTS.performance_log_file
            perf_io = open(performance_file, "w")
            PerformanceLogger(perf_io, Logging.Info)
        catch e
            @warn "Could not create performance log file: $e"
            nothing
        end
    else
        nothing
    end

    # For now, just use the console logger
    # Advanced logging features can be added later
    global_logger(console_logger)

    @info "Logging initialized" log_file structured performance
end

@doc """
Custom logger for structured JSON logging.
"""
struct StructuredLogger <: AbstractLogger
    io::IOStream
    level::Logging.LogLevel
end

function Logging.handle_message(logger::StructuredLogger, level, message, _module, group, id, file, line)
    if level >= logger.level
        log_entry = Dict(
            "timestamp" => string(now()),
            "level" => string(level),
            "message" => message,
            "module" => string(_module),
            "group" => group,
            "id" => id,
            "file" => basename(file),
            "line" => line
        )

        println(logger.io, JSON.json(log_entry))
        flush(logger.io)
    end
end

Logging.shouldlog(::StructuredLogger, level, _module, group, id) = true
Logging.min_enabled_level(::StructuredLogger) = Logging.Debug

@doc """
Custom logger for performance metrics in CSV format.
"""
struct PerformanceLogger <: AbstractLogger
    io::IOStream
    level::Logging.LogLevel
end

function Logging.handle_message(logger::PerformanceLogger, level, message, _module, group, id, file, line)
    if level >= logger.level && occursin("PERF", message)
        # Extract performance data from message
        parts = split(message, " ")
        if length(parts) >= 3
            metric = parts[2]
            value_str = parts[3]
            try
                value = parse(Float64, value_str)
                timestamp = now()
                println(logger.io, "$timestamp,$metric,$value")
                flush(logger.io)
            catch e
                @warn "Could not parse performance data: $e"
            end
        end
    end
end

Logging.shouldlog(::PerformanceLogger, level, _module, group, id) = true
Logging.min_enabled_level(::PerformanceLogger) = Logging.Info

@doc """
Timer context manager for performance measurement.
"""
struct Timer
    name::String
    start_time::Float64
    start_memory::Float64
end

function Timer(name::String)
    Timer(name, time(), memory_usage())
end

function Base.close(timer::Timer)
    end_time = time()
    end_memory = memory_usage()
    elapsed = end_time - timer.start_time
    memory_delta = end_memory - timer.start_memory

    @info "PERF $(timer.name) $(@sprintf("%.4f", elapsed)) memory_delta=$(memory_delta)MB"
end

@doc """
Get current memory usage in MB.
"""
function memory_usage()
    try
        # This is a simplified memory usage check
        # In a real implementation, you might use more sophisticated methods
        0.0  # Placeholder - would need proper memory monitoring
    catch
        0.0
    end
end

@doc """
Export data to CSV format.
"""
function export_to_csv(data::Dict{String, Any}, filename::String)
    # Convert nested dictionaries to flat structure
    flat_data = flatten_dict(data)

    # Convert all values to strings for DataFrame compatibility
    string_data = Dict{String, String}()
    for (k, v) in flat_data
        string_data[k] = string(v)
    end

    # Create DataFrame
    df = DataFrame(string_data)

    # Write to CSV
    CSV.write(filename, df)

    @info "Data exported to CSV: $filename"
end

@doc """
Export data to JSON format.
"""
function export_to_json(data::Dict{String, Any}, filename::String)
    # Create directory if it doesn't exist
    dir = dirname(filename)
    if !isdir(dir)
        mkpath(dir)
    end

    # Write to JSON
    open(filename, "w") do f
        JSON.print(f, data, 2)
    end

    @info "Data exported to JSON: $filename"
end

@doc """
Flatten nested dictionary for CSV export.
"""
function flatten_dict(d::Dict{String, Any}, prefix::String = "")
    flat = Dict{String, Any}()

    for (k, v) in d
        new_key = prefix == "" ? k : "$(prefix)_$(k)"

        if v isa Dict
            merge!(flat, flatten_dict(v, new_key))
        elseif v isa Vector && all(x -> x isa Number, v)
            # Convert numeric vectors to comma-separated strings
            flat[new_key] = join(v, ",")
        else
            flat[new_key] = v
        end
    end

    return flat
end

# Generic overload for any nested Dict types
function flatten_dict(d::Dict, prefix::String = "")
    flat = Dict{String, Any}()

    for (k, v) in d
        new_key = prefix == "" ? k : "$(prefix)_$(k)"

        if v isa Dict
            merge!(flat, flatten_dict(v, new_key))
        elseif v isa Vector && all(x -> x isa Number, v)
            flat[new_key] = join(v, ",")
        else
            flat[new_key] = v
        end
    end

    return flat
end

@doc """
Calculate statistics for a vector of numbers.
"""
function calculate_stats(data::Vector{T} where T <: Number)
    if isempty(data)
        return Dict(
            "mean" => NaN,
            "std" => NaN,
            "min" => NaN,
            "max" => NaN,
            "median" => NaN,
            "length" => 0
        )
    else
        Dict(
            "mean" => mean(data),
            "std" => std(data),
            "min" => minimum(data),
            "max" => maximum(data),
            "median" => median(data),
            "length" => length(data)
        )
    end
end

@doc """
Performance benchmarking decorator.
"""
macro benchmark(name::String, expr)
    quote
        timer = Timer($name)
        try
            result = $(expr)
            close(timer)
            result
        catch e
            close(timer)
            rethrow(e)
        end
    end
end

@doc """
Save experiment results in multiple formats.
"""
function save_experiment_results(experiment_name::String, results::Dict{String, Any})
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

    # Create results directory
    results_dir = "$(OUTPUTS.results_dir)/$(experiment_name)_$(timestamp)"
    if !isdir(results_dir)
        mkpath(results_dir)
    end

    # Save to JSON
    json_file = joinpath(results_dir, "results.json")
    export_to_json(results, json_file)

    # Save to CSV
    csv_file = joinpath(results_dir, "results.csv")
    export_to_csv(results, csv_file)

    @info "Results saved to: $results_dir"

    return results_dir
end

@doc """
Validate configuration parameters.
"""
function validate_config()
    issues = String[]

    # Physics validation
    if PHYSICS.engine_force_limit <= 0
        push!(issues, "Engine force limit must be positive")
    end
    if PHYSICS.friction_coefficient < 0
        push!(issues, "Friction coefficient must be non-negative")
    end

    # World validation
    if WORLD.initial_position >= WORLD.target_position
        push!(issues, "Initial position should be less than target position")
    end

    # Agent validation
    if AGENT.transition_precision <= 0
        push!(issues, "Transition precision must be positive")
    end
    if AGENT.observation_variance <= 0
        push!(issues, "Observation variance must be positive")
    end

    return issues
end

@doc """
Progress bar for long-running operations.
"""
mutable struct ProgressBar
    total::Int
    current::Int
    width::Int
    start_time::Float64
end

function ProgressBar(total::Int; width::Int = 50)
    ProgressBar(total, 0, width, time())
end

function update!(pb::ProgressBar, current::Int = pb.current + 1)
    pb.current = current

    elapsed = time() - pb.start_time
    if pb.total > 0
        progress = pb.current / pb.total
        bar_length = round(Int, pb.width * progress)

        bar = "█" ^ bar_length * "░" ^ (pb.width - bar_length)
        percentage = round(progress * 100, digits=1)

        eta = if progress > 0
            elapsed / progress * (1 - progress)
        else
            0.0
        end

        print("\r[$bar] $(percentage)% ($(pb.current)/$(pb.total)) ETA: $(round(eta, digits=1))s")
    end
end

function finish!(pb::ProgressBar)
    update!(pb, pb.total)
    println()  # New line
end

# Export public functions
export setup_logging, Timer, export_to_csv, export_to_json,
       calculate_stats, save_experiment_results, validate_config,
       ProgressBar, memory_usage, update!, finish!

end # module Utils
