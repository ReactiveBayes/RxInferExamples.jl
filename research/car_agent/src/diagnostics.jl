# Diagnostics and Memory Tracing Module
# Comprehensive diagnostic tracking for Active Inference agents

@doc """
Diagnostics module for Active Inference agents.

Provides comprehensive diagnostic tracking including:
- Memory usage monitoring
- Performance profiling
- Belief evolution tracking
- Prediction accuracy analysis
- Free energy monitoring
"""
module Diagnostics

using Statistics
using Dates
using Printf
using LinearAlgebra

# Import configuration
include("../config.jl")
using .Config: DIAGNOSTICS, LOGGING

# ==================== MEMORY TRACING ====================

@doc """
Memory tracer for tracking memory usage during agent execution.
"""
mutable struct MemoryTracer
    enabled::Bool
    trace_interval::Int
    
    # Memory measurements
    timestamps::Vector{DateTime}
    memory_mb::Vector{Float64}
    gc_time::Vector{Float64}
    
    # Statistics
    peak_memory::Float64
    avg_memory::Float64
    total_gc_time::Float64
    
    function MemoryTracer(; enabled::Bool = LOGGING.enable_memory_trace,
                           trace_interval::Int = LOGGING.memory_trace_interval)
        new(
            enabled,
            trace_interval,
            DateTime[],
            Float64[],
            Float64[],
            0.0,
            0.0,
            0.0
        )
    end
end

@doc """
Take a memory measurement.
"""
function trace_memory!(tracer::MemoryTracer)
    if !tracer.enabled
        return
    end
    
    # Get current memory usage (Julia's GC stats)
    gc_stats = Base.gc_num()
    memory_mb = gc_stats.total_allocd / 1024 / 1024
    gc_time = gc_stats.total_time / 1e9  # Convert to seconds
    
    push!(tracer.timestamps, now())
    push!(tracer.memory_mb, memory_mb)
    push!(tracer.gc_time, gc_time)
    
    # Update statistics
    tracer.peak_memory = max(tracer.peak_memory, memory_mb)
    if !isempty(tracer.memory_mb)
        tracer.avg_memory = mean(tracer.memory_mb)
    end
    tracer.total_gc_time = gc_time
end

@doc """
Get memory trace summary.
"""
function get_memory_summary(tracer::MemoryTracer)::Dict{String, Any}
    if isempty(tracer.memory_mb)
        return Dict(
            "enabled" => tracer.enabled,
            "measurements" => 0,
            "peak_memory_mb" => 0.0,
            "avg_memory_mb" => 0.0,
            "total_gc_time" => 0.0
        )
    end
    
    return Dict(
        "enabled" => tracer.enabled,
        "measurements" => length(tracer.memory_mb),
        "peak_memory_mb" => tracer.peak_memory,
        "avg_memory_mb" => tracer.avg_memory,
        "min_memory_mb" => minimum(tracer.memory_mb),
        "max_memory_mb" => maximum(tracer.memory_mb),
        "total_gc_time" => tracer.total_gc_time,
        "memory_growth" => length(tracer.memory_mb) > 1 ?
            tracer.memory_mb[end] - tracer.memory_mb[1] : 0.0
    )
end

# ==================== PERFORMANCE PROFILER ====================

@doc """
Performance profiler for tracking execution timing.
"""
mutable struct PerformanceProfiler
    enabled::Bool
    
    # Timing measurements
    operation_times::Dict{String, Vector{Float64}}
    operation_counts::Dict{String, Int}
    
    # Current timing
    active_timers::Dict{String, Float64}
    
    function PerformanceProfiler(; enabled::Bool = DIAGNOSTICS.track_inference_time)
        new(
            enabled,
            Dict{String, Vector{Float64}}(),
            Dict{String, Int}(),
            Dict{String, Float64}()
        )
    end
end

@doc """
Start timing an operation.
"""
function start_timer!(profiler::PerformanceProfiler, operation::String)
    if !profiler.enabled
        return
    end
    
    profiler.active_timers[operation] = time()
end

@doc """
Stop timing an operation and record the duration.
"""
function stop_timer!(profiler::PerformanceProfiler, operation::String)
    if !profiler.enabled
        return
    end
    
    if !haskey(profiler.active_timers, operation)
        @warn "Timer not started for operation: $operation"
        return
    end
    
    duration = time() - profiler.active_timers[operation]
    delete!(profiler.active_timers, operation)
    
    # Record timing
    if !haskey(profiler.operation_times, operation)
        profiler.operation_times[operation] = Float64[]
        profiler.operation_counts[operation] = 0
    end
    
    push!(profiler.operation_times[operation], duration)
    profiler.operation_counts[operation] += 1
end

@doc """
Get performance summary.
"""
function get_performance_summary(profiler::PerformanceProfiler)::Dict{String, Any}
    if !profiler.enabled || isempty(profiler.operation_times)
        return Dict("enabled" => profiler.enabled, "operations" => Dict())
    end
    
    operations = Dict{String, Any}()
    
    for (op, times) in profiler.operation_times
        operations[op] = Dict(
            "count" => profiler.operation_counts[op],
            "total_time" => sum(times),
            "avg_time" => mean(times),
            "min_time" => minimum(times),
            "max_time" => maximum(times),
            "std_time" => std(times)
        )
    end
    
    return Dict(
        "enabled" => profiler.enabled,
        "operations" => operations,
        "total_operations" => sum(values(profiler.operation_counts))
    )
end

# ==================== BELIEF TRACKER ====================

@doc """
Belief tracker for monitoring state belief evolution.
"""
mutable struct BeliefTracker
    enabled::Bool
    
    # Belief history
    belief_means::Vector{Vector{Float64}}
    belief_covs::Vector{Matrix{Float64}}
    timestamps::Vector{Int}
    
    # Statistics
    belief_changes::Vector{Float64}  # Norm of belief change
    uncertainty_trace::Vector{Float64}  # Trace of covariance
    
    function BeliefTracker(; enabled::Bool = DIAGNOSTICS.track_beliefs)
        new(
            enabled,
            Vector{Vector{Float64}}(),
            Vector{Matrix{Float64}}(),
            Int[],
            Float64[],
            Float64[]
        )
    end
end

@doc """
Record a belief state.
"""
function record_belief!(tracker::BeliefTracker,
                       step::Int,
                       mean::Vector{Float64},
                       cov::Matrix{Float64})
    if !tracker.enabled
        return
    end
    
    push!(tracker.timestamps, step)
    push!(tracker.belief_means, copy(mean))
    push!(tracker.belief_covs, copy(cov))
    
    # Compute belief change (if not first measurement)
    if length(tracker.belief_means) > 1
        prev_mean = tracker.belief_means[end-1]
        change = norm(mean - prev_mean)
        push!(tracker.belief_changes, change)
    end
    
    # Compute uncertainty (trace of covariance)
    push!(tracker.uncertainty_trace, tr(cov))
end

@doc """
Get belief tracking summary.
"""
function get_belief_summary(tracker::BeliefTracker)::Dict{String, Any}
    if !tracker.enabled || isempty(tracker.belief_means)
        return Dict("enabled" => tracker.enabled, "measurements" => 0)
    end
    
    return Dict(
        "enabled" => tracker.enabled,
        "measurements" => length(tracker.belief_means),
        "final_belief_mean" => tracker.belief_means[end],
        "final_uncertainty" => tracker.uncertainty_trace[end],
        "avg_belief_change" => isempty(tracker.belief_changes) ? 
            0.0 : mean(tracker.belief_changes),
        "max_belief_change" => isempty(tracker.belief_changes) ?
            0.0 : maximum(tracker.belief_changes),
        "avg_uncertainty" => mean(tracker.uncertainty_trace),
        "uncertainty_reduction" => length(tracker.uncertainty_trace) > 1 ?
            tracker.uncertainty_trace[1] - tracker.uncertainty_trace[end] : 0.0
    )
end

# ==================== PREDICTION TRACKER ====================

@doc """
Prediction accuracy tracker.
"""
mutable struct PredictionTracker
    enabled::Bool
    
    # Prediction history
    predictions::Vector{Vector{Vector{Float64}}}  # [step][horizon][state_dim]
    actual_states::Vector{Vector{Float64}}
    prediction_errors::Vector{Vector{Float64}}  # [step][horizon]
    
    function PredictionTracker(; enabled::Bool = DIAGNOSTICS.track_predictions)
        new(
            enabled,
            Vector{Vector{Vector{Float64}}}(),
            Vector{Vector{Float64}}(),
            Vector{Vector{Float64}}()
        )
    end
end

@doc """
Record predictions.
"""
function record_predictions!(tracker::PredictionTracker,
                            predictions::Vector{Vector{Float64}},
                            actual_state::Vector{Float64})
    if !tracker.enabled
        return
    end
    
    push!(tracker.predictions, predictions)
    push!(tracker.actual_states, copy(actual_state))
    
    # Compute prediction errors for each horizon
    if length(tracker.predictions) > 1
        prev_predictions = tracker.predictions[end-1]
        errors = Float64[]
        
        for (i, pred) in enumerate(prev_predictions)
            if i <= length(tracker.actual_states)
                # Error is norm of difference
                error = norm(pred - actual_state)
                push!(errors, error)
            end
        end
        
        if !isempty(errors)
            push!(tracker.prediction_errors, errors)
        end
    end
end

@doc """
Get prediction tracking summary.
"""
function get_prediction_summary(tracker::PredictionTracker)::Dict{String, Any}
    if !tracker.enabled || isempty(tracker.predictions)
        return Dict("enabled" => tracker.enabled, "measurements" => 0)
    end
    
    # Compute average error by horizon
    horizon_errors = Dict{Int, Float64}()
    
    for errors in tracker.prediction_errors
        for (h, error) in enumerate(errors)
            if !haskey(horizon_errors, h)
                horizon_errors[h] = 0.0
            end
            horizon_errors[h] += error
        end
    end
    
    # Average
    for h in keys(horizon_errors)
        horizon_errors[h] /= length(tracker.prediction_errors)
    end
    
    return Dict(
        "enabled" => tracker.enabled,
        "measurements" => length(tracker.predictions),
        "avg_error_by_horizon" => horizon_errors,
        "total_prediction_sets" => length(tracker.prediction_errors)
    )
end

# ==================== FREE ENERGY TRACKER ====================

@doc """
Free energy tracker for monitoring variational free energy.
"""
mutable struct FreeEnergyTracker
    enabled::Bool
    
    # Free energy history
    free_energies::Vector{Float64}
    timestamps::Vector{Int}
    
    # Convergence analysis
    converged_at::Union{Nothing, Int}
    convergence_tolerance::Float64
    
    function FreeEnergyTracker(; enabled::Bool = DIAGNOSTICS.track_free_energy,
                               convergence_tolerance::Float64 = 1e-6)
        new(
            enabled,
            Float64[],
            Int[],
            nothing,
            convergence_tolerance
        )
    end
end

@doc """
Record free energy value.
"""
function record_free_energy!(tracker::FreeEnergyTracker,
                            step::Int,
                            free_energy::Float64)
    if !tracker.enabled
        return
    end
    
    push!(tracker.timestamps, step)
    push!(tracker.free_energies, free_energy)
    
    # Check convergence
    if tracker.converged_at === nothing && length(tracker.free_energies) > 1
        fe_change = abs(tracker.free_energies[end] - tracker.free_energies[end-1])
        if fe_change < tracker.convergence_tolerance
            tracker.converged_at = step
        end
    end
end

@doc """
Get free energy summary.
"""
function get_free_energy_summary(tracker::FreeEnergyTracker)::Dict{String, Any}
    if !tracker.enabled || isempty(tracker.free_energies)
        return Dict("enabled" => tracker.enabled, "measurements" => 0)
    end
    
    return Dict(
        "enabled" => tracker.enabled,
        "measurements" => length(tracker.free_energies),
        "final_free_energy" => tracker.free_energies[end],
        "min_free_energy" => minimum(tracker.free_energies),
        "max_free_energy" => maximum(tracker.free_energies),
        "avg_free_energy" => mean(tracker.free_energies),
        "converged" => tracker.converged_at !== nothing,
        "converged_at_step" => tracker.converged_at,
        "fe_reduction" => length(tracker.free_energies) > 1 ?
            tracker.free_energies[1] - tracker.free_energies[end] : 0.0
    )
end

# ==================== COMPREHENSIVE DIAGNOSTICS ====================

@doc """
Comprehensive diagnostics collector.

Combines all diagnostic trackers into a single interface.
"""
mutable struct DiagnosticsCollector
    memory_tracer::MemoryTracer
    performance_profiler::PerformanceProfiler
    belief_tracker::BeliefTracker
    prediction_tracker::PredictionTracker
    free_energy_tracker::FreeEnergyTracker
    
    function DiagnosticsCollector()
        new(
            MemoryTracer(),
            PerformanceProfiler(),
            BeliefTracker(),
            PredictionTracker(),
            FreeEnergyTracker()
        )
    end
end

@doc """
Get comprehensive diagnostics summary.
"""
function get_comprehensive_summary(collector::DiagnosticsCollector)::Dict{String, Any}
    return Dict(
        "memory" => get_memory_summary(collector.memory_tracer),
        "performance" => get_performance_summary(collector.performance_profiler),
        "beliefs" => get_belief_summary(collector.belief_tracker),
        "predictions" => get_prediction_summary(collector.prediction_tracker),
        "free_energy" => get_free_energy_summary(collector.free_energy_tracker)
    )
end

@doc """
Print diagnostics report.
"""
function print_diagnostics_report(collector::DiagnosticsCollector)
    summary = get_comprehensive_summary(collector)
    
    println("\n" * "="^60)
    println("COMPREHENSIVE DIAGNOSTICS REPORT")
    println("="^60)
    
    # Memory
    println("\n[Memory Usage]")
    mem = summary["memory"]
    if mem["measurements"] > 0
        @printf("  Peak Memory: %.2f MB\n", mem["peak_memory_mb"])
        @printf("  Avg Memory: %.2f MB\n", mem["avg_memory_mb"])
        @printf("  Memory Growth: %.2f MB\n", mem["memory_growth"])
        @printf("  Total GC Time: %.3f s\n", mem["total_gc_time"])
    else
        println("  No measurements available")
    end
    
    # Performance
    println("\n[Performance]")
    perf = summary["performance"]
    if haskey(perf, "operations") && !isempty(perf["operations"])
        for (op, stats) in perf["operations"]
            println("  $op:")
            @printf("    Count: %d\n", stats["count"])
            @printf("    Total Time: %.3f s\n", stats["total_time"])
            @printf("    Avg Time: %.4f s\n", stats["avg_time"])
            @printf("    Min/Max: %.4f / %.4f s\n", stats["min_time"], stats["max_time"])
        end
    else
        println("  No measurements available")
    end
    
    # Beliefs
    println("\n[Belief Evolution]")
    beliefs = summary["beliefs"]
    if beliefs["measurements"] > 0
        @printf("  Measurements: %d\n", beliefs["measurements"])
        @printf("  Final Uncertainty: %.6f\n", beliefs["final_uncertainty"])
        @printf("  Avg Belief Change: %.6f\n", beliefs["avg_belief_change"])
        @printf("  Uncertainty Reduction: %.6f\n", beliefs["uncertainty_reduction"])
    else
        println("  No measurements available")
    end
    
    # Predictions
    println("\n[Prediction Accuracy]")
    preds = summary["predictions"]
    if preds["measurements"] > 0
        @printf("  Measurements: %d\n", preds["measurements"])
        if haskey(preds, "avg_error_by_horizon") && !isempty(preds["avg_error_by_horizon"])
            println("  Average Error by Horizon:")
            for (h, error) in sort(collect(preds["avg_error_by_horizon"]))
                @printf("    Horizon %d: %.6f\n", h, error)
            end
        end
    else
        println("  No measurements available")
    end
    
    # Free Energy
    println("\n[Free Energy]")
    fe = summary["free_energy"]
    if fe["measurements"] > 0
        @printf("  Measurements: %d\n", fe["measurements"])
        @printf("  Final FE: %.2f\n", fe["final_free_energy"])
        @printf("  Min/Max FE: %.2f / %.2f\n", fe["min_free_energy"], fe["max_free_energy"])
        @printf("  FE Reduction: %.2f\n", fe["fe_reduction"])
        @printf("  Converged: %s\n", fe["converged"] ? "Yes (step $(fe["converged_at_step"]))" : "No")
    else
        println("  No measurements available")
    end
    
    println("\n" * "="^60)
end

# Export public API
export MemoryTracer, PerformanceProfiler, BeliefTracker, PredictionTracker, FreeEnergyTracker
export DiagnosticsCollector
export trace_memory!, start_timer!, stop_timer!, record_belief!, record_predictions!, record_free_energy!
export get_memory_summary, get_performance_summary, get_belief_summary, get_prediction_summary, get_free_energy_summary
export get_comprehensive_summary, print_diagnostics_report

end # module Diagnostics

