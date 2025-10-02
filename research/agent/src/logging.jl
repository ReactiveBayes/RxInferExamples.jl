# Logging Module for Active Inference Agents
# Comprehensive structured logging with multiple output formats

"""
Logging module for Active Inference agents.

Provides comprehensive logging with multiple output formats:
- Console logging (human-readable)
- File logging (persistent storage)
- Structured JSON logging (machine-readable)
- Performance CSV logging (for analysis)
"""
module LoggingUtils

using Logging
using Dates
using JSON
using Printf

# Import configuration constants
include("constants.jl")
using .AgentConstants: LOGGING, OUTPUTS

# ==================== STRUCTURED LOGGER ====================

"""
Structured JSON logger for machine-readable event logging.
"""
struct StructuredLogger <: AbstractLogger
    io::IOStream
    min_level::LogLevel
end

function Logging.handle_message(logger::StructuredLogger, level, message, _module, group, id, file, line; kwargs...)
    if level >= logger.min_level
        log_entry = Dict(
            "timestamp" => string(now()),
            "level" => string(level),
            "message" => message,
            "module" => string(_module),
            "file" => basename(string(file)),
            "line" => line
        )
        
        # Add additional keyword arguments
        for (k, v) in kwargs
            log_entry[string(k)] = v
        end
        
        println(logger.io, JSON.json(log_entry))
        flush(logger.io)
    end
end

Logging.shouldlog(::StructuredLogger, level, _module, group, id) = true
Logging.min_enabled_level(logger::StructuredLogger) = logger.min_level
Logging.catch_exceptions(::StructuredLogger) = false

# ==================== PERFORMANCE LOGGER ====================

"""
Performance CSV logger for timing and metrics.
"""
struct PerformanceLogger <: AbstractLogger
    io::IOStream
    min_level::LogLevel
    header_written::Ref{Bool}
end

function PerformanceLogger(io::IOStream, min_level::LogLevel = Logging.Info)
    logger = PerformanceLogger(io, min_level, Ref(false))
    
    # Write CSV header
    if !logger.header_written[]
        println(io, "timestamp,operation,value,unit")
        flush(io)
        logger.header_written[] = true
    end
    
    return logger
end

function Logging.handle_message(logger::PerformanceLogger, level, message, _module, group, id, file, line; kwargs...)
    if level >= logger.min_level
        # Look for performance data in kwargs
        if haskey(kwargs, :operation) && haskey(kwargs, :value)
            timestamp = now()
            operation = kwargs[:operation]
            value = kwargs[:value]
            unit = get(kwargs, :unit, "")
            
            println(logger.io, "$timestamp,$operation,$value,$unit")
            flush(logger.io)
        end
    end
end

Logging.shouldlog(::StructuredLogger, level, _module, group, id) = true
Logging.min_enabled_level(logger::PerformanceLogger) = logger.min_level
Logging.catch_exceptions(::PerformanceLogger) = false

# ==================== LOGGING SETUP ====================

"""
Setup comprehensive logging system.

Args:
- verbose: Enable verbose console logging
- structured: Enable structured JSON logging
- performance: Enable performance CSV logging
- log_file: Path to log file (optional)

Returns:
- Dictionary with logger handles
"""
function setup_logging(;
    verbose::Bool = LOGGING.enable_logging,
    structured::Bool = LOGGING.enable_structured,
    performance::Bool = LOGGING.enable_performance,
    log_file::Union{Nothing, String} = nothing
)
    loggers = Dict{Symbol, Any}()
    
    # Determine log level
    log_level = verbose ? Logging.Info : Logging.Warn
    
    # Setup console logger
    if LOGGING.log_to_console
        console_logger = ConsoleLogger(stderr, log_level)
        global_logger(console_logger)
        loggers[:console] = console_logger
    end
    
    # Setup file logger
    if LOGGING.log_to_file
        log_path = log_file !== nothing ? log_file : LOGGING.log_file
        
        # Create directory if needed
        log_dir = dirname(log_path)
        if !isdir(log_dir) && !isempty(log_dir)
            mkpath(log_dir)
        end
        
        try
            log_io = open(log_path, "w")
            file_logger = SimpleLogger(log_io, log_level)
            loggers[:file] = file_logger
            loggers[:file_io] = log_io
        catch e
            @warn "Could not create file logger" exception=e
        end
    end
    
    # Setup structured logger
    if structured
        struct_path = LOGGING.structured_file
        
        # Create directory if needed
        struct_dir = dirname(struct_path)
        if !isdir(struct_dir) && !isempty(struct_dir)
            mkpath(struct_dir)
        end
        
        try
            struct_io = open(struct_path, "w")
            struct_logger = StructuredLogger(struct_io, log_level)
            loggers[:structured] = struct_logger
            loggers[:structured_io] = struct_io
        catch e
            @warn "Could not create structured logger" exception=e
        end
    end
    
    # Setup performance logger
    if performance
        perf_path = LOGGING.performance_file
        
        # Create directory if needed
        perf_dir = dirname(perf_path)
        if !isdir(perf_dir) && !isempty(perf_dir)
            mkpath(perf_dir)
        end
        
        try
            perf_io = open(perf_path, "w")
            perf_logger = PerformanceLogger(perf_io, Logging.Info)
            loggers[:performance] = perf_logger
            loggers[:performance_io] = perf_io
        catch e
            @warn "Could not create performance logger" exception=e
        end
    end
    
    @info "Logging initialized" console=LOGGING.log_to_console file=LOGGING.log_to_file structured=structured performance=performance
    
    return loggers
end

"""
Close all loggers and flush output.

Args:
- loggers: Dictionary of loggers from setup_logging
"""
function close_logging(loggers::Dict{Symbol, Any})
    # Close file IO handles
    for (key, value) in loggers
        if key in [:file_io, :structured_io, :performance_io]
            try
                close(value)
            catch e
                @warn "Error closing logger IO" key=key exception=e
            end
        end
    end
    
    @info "Logging closed"
end

# ==================== LOGGING UTILITIES ====================

"""
Log performance metric.

Args:
- operation: Operation name
- value: Metric value
- unit: Unit of measurement (optional)
"""
function log_performance(operation::String, value::Float64; unit::String = "")
    @info "PERF" operation=operation value=value unit=unit
end

"""
Log structured event.

Args:
- event_type: Type of event
- data: Event data dictionary
"""
function log_event(event_type::String, data::Dict{String, Any})
    # Convert to named tuple for @info macro
    if isempty(data)
        @info event_type
    else
        # Create pairs for logging
        pairs = [Symbol(k) => v for (k, v) in data]
        @info event_type pairs...
    end
end

"""
Log agent step.

Args:
- step: Step number
- action: Action taken
- observation: Observation received
- free_energy: Free energy value (optional)
"""
function log_agent_step(step::Int, action::Vector{Float64}, observation::Vector{Float64};
                       free_energy::Union{Nothing, Float64} = nothing)
    log_data = Dict{String, Any}(
        "step" => step,
        "action" => action,
        "observation" => observation
    )
    
    if free_energy !== nothing
        log_data["free_energy"] = free_energy
    end
    
    log_event("agent_step", log_data)
end

"""
Log inference result.

Args:
- step: Step number
- inference_time: Time taken for inference
- free_energy: Free energy value
- converged: Whether inference converged
"""
function log_inference(step::Int, inference_time::Float64,
                      free_energy::Union{Nothing, Float64} = nothing;
                      converged::Bool = false)
    log_data = Dict{String, Any}(
        "step" => step,
        "inference_time" => inference_time,
        "converged" => converged
    )
    
    if free_energy !== nothing
        log_data["free_energy"] = free_energy
    end
    
    log_event("inference", log_data)
    log_performance("inference_time", inference_time, unit="seconds")
end

"""
Create progress bar for long operations.
"""
mutable struct ProgressBar
    total::Int
    current::Int
    width::Int
    start_time::Float64
    last_update::Float64
    update_interval::Float64
    
    function ProgressBar(total::Int; width::Int = 50, update_interval::Float64 = 0.1)
        new(total, 0, width, time(), time(), update_interval)
    end
end

"""
Update progress bar.
"""
function update!(pb::ProgressBar, current::Int = pb.current + 1)
    pb.current = current
    
    # Rate limit updates
    current_time = time()
    if current_time - pb.last_update < pb.update_interval && pb.current < pb.total
        return
    end
    pb.last_update = current_time
    
    elapsed = current_time - pb.start_time
    progress = pb.current / pb.total
    bar_length = round(Int, pb.width * progress)
    
    bar = "█" ^ bar_length * "░" ^ (pb.width - bar_length)
    percentage = round(progress * 100, digits=1)
    
    eta = if progress > 0
        remaining = elapsed / progress * (1 - progress)
        @sprintf("%.1fs", remaining)
    else
        "N/A"
    end
    
    print("\r[$bar] $(percentage)% ($(pb.current)/$(pb.total)) ETA: $eta  ")
    flush(stdout)
end

"""
Finish progress bar.
"""
function finish!(pb::ProgressBar)
    update!(pb, pb.total)
    println()  # New line
    
    elapsed = time() - pb.start_time
    @info "Operation completed" total_time=elapsed steps=pb.total
end

# Export public API
export StructuredLogger, PerformanceLogger
export setup_logging, close_logging
export log_performance, log_event, log_agent_step, log_inference
export ProgressBar, update!, finish!

end # module LoggingUtils

