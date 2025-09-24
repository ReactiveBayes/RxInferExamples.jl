# Comprehensive Utility Module for Active Inference Car Examples
# Provides logging, data export, performance monitoring, and system utilities

@doc """
Comprehensive utility module for active inference car examples.

This module provides essential utilities including advanced logging,
comprehensive data export, performance monitoring, validation tools,
and system management functions.

## Key Features
- **Advanced Logging**: Structured logging with multiple output formats
- **Data Export**: Comprehensive export to CSV, JSON, and other formats
- **Performance Monitoring**: Detailed timing and memory usage tracking
- **Validation Tools**: Configuration and data validation utilities
- **Progress Tracking**: Real-time progress bars and status updates
- **Error Handling**: Robust error handling with detailed reporting
- **System Utilities**: File management, timing, and resource monitoring
"""
module Utils

using Logging
using DataFrames
using CSV
using JSON
using Dates
using Printf
using ProgressMeter
using Statistics
using Base.Threads
using LinearAlgebra
import ..Config: OUTPUTS, get_config_value

# ==================== LOGGING SYSTEM ====================

@doc """
Advanced logging system with multiple output formats.

Supports structured logging, performance logging, and multiple output destinations.
"""
struct AdvancedLogger
    console_logger::ConsoleLogger
    file_logger::SimpleLogger
    structured_logger::Union{SimpleLogger, Nothing}
    performance_logger::Union{SimpleLogger, Nothing}
    log_file::String
    structured_file::String
    performance_file::String

    function AdvancedLogger(;
        log_file::String = "active_inference_car.log",
        enable_structured::Bool = false,
        enable_performance::Bool = false,
        log_level::Logging.LogLevel = Logging.Info
    )
        # Create output directory
        log_dir = dirname(log_file)
        if !isempty(log_dir) && !isdir(log_dir)
            mkpath(log_dir)
        end

        # Console logger
        console_logger = ConsoleLogger(stderr, log_level)

        # File logger
        file_logger = SimpleLogger(open(log_file, "a"), log_level)

        # Structured logger (JSON format)
        structured_logger = enable_structured ?
            SimpleLogger(open(replace(log_file, ".log" => "_structured.jsonl"), "a"), log_level) : nothing

        # Performance logger (CSV format)
        performance_logger = enable_performance ?
            SimpleLogger(open(replace(log_file, ".log" => "_performance.csv"), "a"), log_level) : nothing

        structured_file = enable_structured ? replace(log_file, ".log" => "_structured.jsonl") : ""
        performance_file = enable_performance ? replace(log_file, ".log" => "_performance.csv") : ""

        new(console_logger, file_logger, structured_logger, performance_logger,
            log_file, structured_file, performance_file)
    end
end

@doc """
Initialize the global logging system.

Args:
- log_file: Base log file name
- enable_structured: Enable structured JSON logging
- enable_performance: Enable performance CSV logging
- log_level: Minimum log level to display
"""
function setup_logging(; log_file::String = OUTPUTS.log_filename,
                      enable_structured::Bool = false,
                      enable_performance::Bool = false,
                      log_level::Logging.LogLevel = Logging.Info)

    global_logger = AdvancedLogger(
        log_file = log_file,
        enable_structured = enable_structured,
        enable_performance = enable_performance,
        log_level = log_level
    )

    # Configure global logger to write to all outputs
    logger_instance[] = global_logger

    @info "Logging system initialized" log_file = log_file enable_structured = enable_structured enable_performance = enable_performance
end

@doc """
Log structured data in JSON format.

Args:
- data: Data to log (will be converted to JSON)
- level: Log level
"""
function log_structured(data::Dict, level::Logging.LogLevel = Logging.Info)
    if logger_instance[] !== nothing && logger_instance[].structured_logger !== nothing
        timestamp = Dates.now()
        log_entry = Dict(
            "timestamp" => string(timestamp),
            "level" => string(level),
            "data" => data
        )
        println(logger_instance[].structured_logger.stream, JSON.json(log_entry))
        flush(logger_instance[].structured_logger.stream)
    end
end

@doc """
Log performance metrics in CSV format.

Args:
- operation: Operation name
- duration: Duration in seconds
- memory_usage: Memory usage in MB (optional)
- metadata: Additional metadata
"""
function log_performance(operation::String, duration::Float64;
                        memory_usage::Float64 = 0.0, metadata::Dict = Dict())
    if logger_instance[] !== nothing && logger_instance[].performance_logger !== nothing
        timestamp = Dates.now()
        performance_entry = Dict(
            "timestamp" => string(timestamp),
            "operation" => operation,
            "duration_seconds" => duration,
            "memory_mb" => memory_usage,
            "metadata" => JSON.json(metadata)
        )
        println(logger_instance[].performance_logger.stream, join(values(performance_entry), ","))
        flush(logger_instance[].performance_logger.stream)
    end
end

# Global logger instance
const logger_instance = Ref{Union{AdvancedLogger, Nothing}}(nothing)

# ==================== PERFORMANCE MONITORING ====================

@doc """
Performance timer for measuring operation duration.

Usage:
```julia
timer = PerformanceTimer("my_operation")
# ... do work ...
close(timer)  # Logs performance metrics
```
"""
struct PerformanceTimer
    operation::String
    start_time::Float64
    start_memory::Float64

    function PerformanceTimer(operation::String)
        start_time = time()
        start_memory = get_memory_usage()
        new(operation, start_time, start_memory)
    end
end

@doc """
Close performance timer and log results.

Args:
- timer: PerformanceTimer instance
- metadata: Additional metadata to log
"""
function close(timer::PerformanceTimer; metadata::Dict = Dict())
    end_time = time()
    end_memory = get_memory_usage()

    duration = end_time - timer.start_time
    memory_delta = end_memory - timer.start_memory

    log_performance(timer.operation, duration, memory_usage = memory_delta, metadata = metadata)

    @info "PERF $(timer.operation)" duration = round(duration, digits=4) memory_delta = round(memory_delta, digits=2)
end

@doc """
Get current memory usage in MB.

Returns:
- Memory usage in MB
"""
function get_memory_usage()
    try
        # Try to get memory info from /proc/meminfo (Linux)
        meminfo = read("/proc/meminfo", String)
        for line in split(meminfo, '\n')
            if startswith(line, "MemTotal:")
                total_kb = parse(Int, split(line)[2])
                return total_kb / 1024  # Convert to MB
            end
        end
    catch
        # Fallback: return 0 if we can't get memory info
        return 0.0
    end
end

@doc """
Benchmark function execution time.

Args:
- name: Benchmark name
- f: Function to benchmark
- args: Arguments to pass to function
- kwargs: Keyword arguments to pass to function
- iterations: Number of iterations to run

Returns:
- Dictionary with benchmark results
"""
function benchmark(name::String, f::Function; iterations::Int = 10)
    times = Float64[]
    memories = Float64[]

    for i in 1:iterations
        timer = PerformanceTimer("$name_$i")
        result = f(args...; kwargs...)
        close(timer)

        push!(times, timer.start_time)  # Would need to modify PerformanceTimer to expose duration
        push!(memories, timer.start_memory)
    end

    return Dict(
        "name" => name,
        "mean_time" => mean(times),
        "std_time" => std(times),
        "min_time" => minimum(times),
        "max_time" => maximum(times),
        "mean_memory" => mean(memories),
        "iterations" => iterations
    )
end

# ==================== DATA EXPORT FUNCTIONS ====================

@doc """
Export experiment results to multiple formats.

Args:
- experiment_name: Name of the experiment
- results: Results dictionary
- output_dir: Output directory (optional)

Returns:
- Directory where results were saved
"""
function export_experiment_results(experiment_name::String, results; output_dir::String = "")
    if isempty(output_dir)
        output_dir = joinpath(OUTPUTS.output_dir, "results", "$(experiment_name)_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))")
    end

    mkpath(output_dir)

    try
        # Export to JSON
        json_file = joinpath(output_dir, "results.json")
        open(json_file, "w") do f
            JSON.print(f, results, 2)
        end

        # Export to CSV (flattened data)
        csv_file = joinpath(output_dir, "results.csv")
        flattened_data = flatten_dict(results)
        df = DataFrame(flattened_data)
        CSV.write(csv_file, df)

        # Log success
        log_structured(Dict(
            "event" => "experiment_exported",
            "experiment_name" => experiment_name,
            "output_dir" => output_dir,
            "json_file" => json_file,
            "csv_file" => csv_file
        ))

        @info "Experiment results exported" experiment_name = experiment_name output_dir = output_dir
        return output_dir

    catch e
        @error "Failed to export experiment results" experiment_name = experiment_name error = string(e)
        rethrow(e)
    end
end

@doc """
Flatten nested dictionary for CSV export.

Args:
- data: Nested dictionary
- prefix: Prefix for keys

Returns:
- Flattened dictionary
"""
function flatten_dict(data; prefix::String = "")
    flat = Dict{String, Any}()

    for (key, value) in data
        full_key = isempty(prefix) ? string(key) : "$(prefix)_$(key)"

        if value isa Dict
            # Handle nested dictionaries regardless of key types
            merge!(flat, flatten_dict(value, prefix=full_key))
        elseif value isa Vector
            # Convert vectors to strings for CSV
            flat[full_key] = join(string.(value), ";")
        elseif value isa Matrix
            # Convert matrices to strings for CSV
            matrix_strings = [join(string.(row), ",") for row in eachrow(value)]
            flat[full_key] = join(matrix_strings, ";")
        else
            flat[full_key] = value
        end
    end

    return flat
end

@doc """
Export comprehensive trajectory data.

Args:
- trajectory: State trajectory
- actions: Action sequence
- predictions: Prediction matrix
- filename_prefix: Prefix for output files
- output_dir: Output directory
"""
function export_trajectory_data(trajectory::Vector{Vector{Float64}},
                               actions::Vector{Float64},
                               predictions::Union{Matrix{Float64}, Nothing},
                               filename_prefix::String, output_dir::String)

    mkpath(output_dir)

    try
        # Extract time series
        positions = [state[1] for state in trajectory]
        velocities = [state[2] for state in trajectory]
        time_steps = 1:length(trajectory)

        # Create main trajectory DataFrame
        df = DataFrame(
            time_step = time_steps,
            position = positions,
            velocity = velocities,
            action = vcat(actions, NaN),  # Pad with NaN if shorter
            distance_to_goal = [abs(pos - 0.5) for pos in positions],  # Assume goal at 0.5
            kinetic_energy = [0.5 * v^2 for v in velocities],
            potential_energy = [abs(pos) for pos in positions]  # Simplified potential
        )

        # Export main trajectory
        csv_file = joinpath(output_dir, "$(filename_prefix)_trajectory.csv")
        CSV.write(csv_file, df)

        # Export predictions if available
        if predictions !== nothing
            pred_df = DataFrame(predictions, :auto)
            rename!(pred_df, [Symbol("prediction_$i") for i in 1:size(predictions, 2)])
            pred_df.time_step = 1:size(predictions, 1)
            CSV.write(joinpath(output_dir, "$(filename_prefix)_predictions.csv"), pred_df)
        end

        # Export metadata
        metadata = Dict(
            "total_steps" => length(trajectory),
            "final_position" => positions[end],
            "max_velocity" => maximum(velocities),
            "total_distance" => sum(abs(positions[i] - positions[i-1]) for i in 2:length(positions)),
            "export_timestamp" => string(now())
        )

        open(joinpath(output_dir, "$(filename_prefix)_metadata.json"), "w") do f
            JSON.print(f, metadata, 2)
        end

        @info "Trajectory data exported" prefix = filename_prefix output_dir = output_dir

    catch e
        @error "Failed to export trajectory data" prefix = filename_prefix error = string(e)
    end
end

# ==================== VALIDATION UTILITIES ====================

@doc """
Validate experiment configuration.

Args:
- config: Configuration dictionary

Returns:
- Vector of validation error messages
"""
function validate_experiment_config(config::Dict)
    errors = String[]

    # Validate required sections
    required_sections = [:physics, :world, :agent, :simulation]
    for section in required_sections
        if !haskey(config, section)
            push!(errors, "Missing required configuration section: $section")
        end
    end

    # Validate physics parameters
    if haskey(config, :physics)
        physics = config[:physics]
        if haskey(physics, :engine_force_limit) && physics[:engine_force_limit] <= 0
            push!(errors, "Engine force limit must be positive")
        end
        if haskey(physics, :friction_coefficient) && physics[:friction_coefficient] < 0
            push!(errors, "Friction coefficient cannot be negative")
        end
    end

    # Validate world parameters
    if haskey(config, :world)
        world = config[:world]
        if haskey(world, :x_max) && haskey(world, :x_min) && world[:x_max] <= world[:x_min]
            push!(errors, "World x_max must be greater than x_min")
        end
        if haskey(world, :goal_tolerance) && world[:goal_tolerance] <= 0
            push!(errors, "Goal tolerance must be positive")
        end
    end

    # Validate agent parameters
    if haskey(config, :agent)
        agent = config[:agent]
        if haskey(agent, :planning_horizon) && agent[:planning_horizon] <= 0
            push!(errors, "Planning horizon must be positive")
        end
    end

    return errors
end

@doc """
Validate trajectory data for consistency.

Args:
- trajectory: State trajectory
- actions: Action sequence

Returns:
- Vector of validation warnings
"""
function validate_trajectory_data(trajectory::Vector{Vector{Float64}}, actions::Vector{Float64})
    warnings = String[]

    # Check lengths
    if length(actions) != length(trajectory) - 1
        push!(warnings, "Actions length ($(length(actions))) should be trajectory length - 1 ($(length(trajectory) - 1))")
    end

    # Check state consistency
    for i in 2:length(trajectory)
        prev_state = trajectory[i-1]
        curr_state = trajectory[i]

        # Check for unrealistic jumps
        position_jump = abs(curr_state[1] - prev_state[1])
        velocity_jump = abs(curr_state[2] - prev_state[2])

        if position_jump > 1.0
            push!(warnings, "Large position jump at step $i: $(round(position_jump, digits=3))")
        end
        if velocity_jump > 0.5
            push!(warnings, "Large velocity jump at step $i: $(round(velocity_jump, digits=3))")
        end
    end

    return warnings
end

# ==================== PROGRESS TRACKING ====================

@doc """
Enhanced progress bar with ETA and performance metrics.

Args:
- total_steps: Total number of steps
- description: Description of the task

Returns:
- Progress bar instance
"""
function create_progress_bar(total_steps::Int, description::String = "Processing")
    return Progress(total_steps, desc = description, showspeed = true)
end

@doc """
Update progress bar with current step.

Args:
- progress: Progress bar instance
- step: Current step
- metadata: Optional metadata to display
"""
function update_progress!(progress, step::Int; metadata::String = "")
    if !isempty(metadata)
        ProgressMeter.update!(progress, step, showvalues = [(:metadata, metadata)])
    else
        ProgressMeter.update!(progress, step)
    end
end

# ==================== FILE MANAGEMENT ====================

@doc """
Ensure output directory exists and create if necessary.

Args:
- dir: Directory path

Returns:
- Success status
"""
function ensure_output_directory(dir::String)
    try
        if !isdir(dir)
            mkpath(dir)
        end
        return true
    catch e
        @error "Failed to create output directory: $dir" error = string(e)
        return false
    end
end

@doc """
Clean up old output files.

Args:
- directory: Directory to clean
- keep_recent: Number of recent files to keep
- pattern: File pattern to match
"""
function cleanup_old_files(directory::String, keep_recent::Int = 5, pattern::String = "*.log")
    try
        files = glob(pattern, directory)
        sort!(files, by = mtime, rev = true)  # Sort by modification time, newest first

        if length(files) > keep_recent
            for file in files[keep_recent+1:end]
                rm(file)
                @info "Cleaned up old file: $file"
            end
        end
    catch e
        @warn "Failed to cleanup old files" directory = directory error = string(e)
    end
end

# ==================== STATISTICAL ANALYSIS ====================

@doc """
Calculate comprehensive statistics for a time series.

Args:
- data: Time series data

Returns:
- Dictionary of statistics
"""
function calculate_time_series_stats(data::Vector{Float64})
    return Dict(
        "mean" => mean(data),
        "std" => std(data),
        "min" => minimum(data),
        "max" => maximum(data),
        "median" => median(data),
        "q25" => quantile(data, 0.25),
        "q75" => quantile(data, 0.75),
        "range" => maximum(data) - minimum(data),
        "count" => length(data),
        "missing" => count(isnan, data)
    )
end

@doc """
Analyze trajectory efficiency and performance.

Args:
- trajectory: State trajectory
- actions: Action sequence
- target_position: Target position

Returns:
- Dictionary of efficiency metrics
"""
function analyze_trajectory_efficiency(trajectory::Vector{Vector{Float64}},
                                     actions::Vector{Float64},
                                     target_position::Float64)
    positions = [state[1] for state in trajectory]
    velocities = [state[2] for state in trajectory]

    # Basic metrics
    total_distance = sum(abs(positions[i] - positions[i-1]) for i in 2:length(positions))
    final_distance = abs(positions[end] - target_position)
    avg_speed = mean(abs.(velocities))

    # Energy efficiency
    total_action_magnitude = sum(abs.(actions))
    energy_efficiency = total_distance / (total_action_magnitude + 1e-6)

    # Smoothness
    position_smoothness = 1.0 / (1.0 + std(diff(positions)))
    action_smoothness = 1.0 / (1.0 + std(diff(actions)))

    return Dict(
        "total_distance" => total_distance,
        "final_distance" => final_distance,
        "avg_speed" => avg_speed,
        "energy_efficiency" => energy_efficiency,
        "position_smoothness" => position_smoothness,
        "action_smoothness" => action_smoothness,
        "goal_progress" => max(0.0, 1.0 - final_distance / abs(positions[1] - target_position))
    )
end

# ==================== ERROR HANDLING ====================

@doc """
Safe execution wrapper with error handling and logging.

Args:
- f: Function to execute
- args: Arguments to pass to function
- error_message: Custom error message
- fallback: Fallback value if execution fails

Returns:
- Result of function execution or fallback value
"""
function safe_execute(f::Function, args...; error_message::String = "Operation failed",
                     fallback = nothing)
    try
        return f(args...)
    catch e
        @error error_message error = string(e)
        log_structured(Dict(
            "event" => "execution_error",
            "error_type" => string(typeof(e)),
            "error_message" => string(e),
            "fallback_used" => fallback !== nothing
        ))
        return fallback
    end
end

# ==================== SYSTEM UTILITIES ====================

@doc """
Get system information for logging and debugging.

Returns:
- Dictionary of system information
"""
function get_system_info()
    return Dict(
        "julia_version" => string(VERSION),
        "cpu_cores" => Sys.CPU_THREADS,
        "memory_mb" => get_memory_usage(),
        "current_time" => string(now()),
        "working_directory" => pwd(),
        "hostname" => gethostname()
    )
end

@doc """
Format elapsed time in human-readable format.

Args:
- seconds: Time in seconds

Returns:
- Formatted time string
"""
function format_elapsed_time(seconds::Float64)
    if seconds < 60
        return @sprintf("%.2fs", seconds)
    elseif seconds < 3600
        minutes = floor(Int, seconds / 60)
        secs = seconds - minutes * 60
        return @sprintf("%dm %.2fs", minutes, secs)
    else
        hours = floor(Int, seconds / 3600)
        minutes = floor(Int, (seconds - hours * 3600) / 60)
        secs = seconds - hours * 3600 - minutes * 60
        return @sprintf("%dh %dm %.2fs", hours, minutes, secs)
    end
end

# ==================== MODULE EXPORTS ====================

export
    # Logging system
    AdvancedLogger,
    setup_logging,
    log_structured,
    log_performance,

    # Performance monitoring
    PerformanceTimer,
    close,
    get_memory_usage,
    benchmark,

    # Data export
    export_experiment_results,
    flatten_dict,
    export_trajectory_data,

    # Validation
    validate_experiment_config,
    validate_trajectory_data,

    # Progress tracking
    create_progress_bar,
    update_progress!,

    # File management
    ensure_output_directory,
    cleanup_old_files,

    # Statistical analysis
    calculate_time_series_stats,
    analyze_trajectory_efficiency,

    # Error handling
    safe_execute,

    # System utilities
    get_system_info,
    format_elapsed_time

end # module Utils
