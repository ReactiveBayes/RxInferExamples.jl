"""
Advanced RxInfer Diagnostics Module

Provides comprehensive diagnostic capabilities for RxInfer inference:
- Memory Addon for message tracing
- Inference callbacks for iteration tracking
- Logger Pipeline Stage for message passing visualization
- Benchmark callbacks for performance analysis
"""
module CoinTossDiagnostics

using RxInfer
using ReactiveMP
using Distributions
using Statistics
using DataFrames
using CSV
using JSON
using Dates
using Printf
using Logging

export DiagnosticConfig, DiagnosticResults, DiagnosticCallbacks
export run_inference_with_diagnostics, save_diagnostics, visualize_message_trace
export create_diagnostic_callbacks, extract_message_trace, extract_benchmark_stats

"""
Configuration for diagnostic features
"""
struct DiagnosticConfig
    enable_memory_addon::Bool
    enable_callbacks::Bool
    enable_pipeline_logger::Bool
    enable_benchmark::Bool
    verbose::Bool
end

function DiagnosticConfig(;
    enable_memory_addon::Bool = true,
    enable_callbacks::Bool = true,
    enable_pipeline_logger::Bool = false,  # Can be very verbose
    enable_benchmark::Bool = true,
    verbose::Bool = true
)
    return DiagnosticConfig(
        enable_memory_addon,
        enable_callbacks,
        enable_pipeline_logger,
        enable_benchmark,
        verbose
    )
end

"""
Container for diagnostic results
"""
struct DiagnosticResults
    memory_trace::Union{Nothing, Dict{String, Any}}
    callback_trace::Union{Nothing, Vector{Dict{String, Any}}}
    benchmark_stats::Union{Nothing, DataFrame}
    message_history::Union{Nothing, Vector{Dict{String, Any}}}
end

"""
Custom callbacks for tracking inference progress
"""
mutable struct DiagnosticCallbacks
    iteration_data::Vector{Dict{String, Any}}
    marginal_updates::Vector{Dict{String, Any}}
    verbose::Bool
    
    function DiagnosticCallbacks(; verbose::Bool = true)
        new(
            Vector{Dict{String, Any}}(),
            Vector{Dict{String, Any}}(),
            verbose
        )
    end
end

"""
Callback: Before iteration starts
"""
function before_iteration_callback(callbacks::DiagnosticCallbacks)
    return function(model, iteration)
        if callbacks.verbose
            @info "Starting iteration $iteration"
        end
        push!(callbacks.iteration_data, Dict(
            "iteration" => iteration,
            "event" => "start",
            "timestamp" => now()
        ))
    end
end

"""
Callback: After iteration finishes
"""
function after_iteration_callback(callbacks::DiagnosticCallbacks)
    return function(model, iteration)
        if callbacks.verbose
            @info "Completed iteration $iteration"
        end
        push!(callbacks.iteration_data, Dict(
            "iteration" => iteration,
            "event" => "end",
            "timestamp" => now()
        ))
    end
end

"""
Callback: On marginal update
"""
function on_marginal_update_callback(callbacks::DiagnosticCallbacks)
    return function(model, variable_name, posterior)
        μ = mean(posterior)
        σ = std(posterior)
        
        if callbacks.verbose
            @info "Updated $variable_name" mean=μ std=σ
        end
        
        push!(callbacks.marginal_updates, Dict(
            "variable" => string(variable_name),
            "mean" => μ,
            "std" => σ,
            "timestamp" => now(),
            "distribution" => string(typeof(posterior))
        ))
    end
end

"""
Create callback tuple for RxInfer
"""
function create_diagnostic_callbacks(callbacks::DiagnosticCallbacks)
    return (
        before_iteration = before_iteration_callback(callbacks),
        after_iteration = after_iteration_callback(callbacks),
        on_marginal_update = on_marginal_update_callback(callbacks)
    )
end

"""
Extract message trace from Memory Addon
"""
function extract_message_trace(result, variable_name::Symbol)
    if !haskey(result.posteriors, variable_name)
        @warn "Variable $variable_name not found in posteriors"
        return nothing
    end
    
    posterior_raw = result.posteriors[variable_name]
    # Handle case where posteriors is a vector (from multiple iterations)
    posterior = posterior_raw isa Vector ? posterior_raw[end] : posterior_raw
    addons = ReactiveMP.getaddons(posterior)
    
    if isempty(addons)
        @warn "No addons found for $variable_name"
        return nothing
    end
    
    memory_addon = addons[1]
    
    # Extract message information
    message_info = Dict{String, Any}()
    message_info["variable"] = string(variable_name)
    message_info["posterior_type"] = string(typeof(posterior))
    message_info["posterior_params"] = params(posterior)
    message_info["messages"] = []
    
    # Try to extract memory trace
    if isdefined(memory_addon, :memory) && memory_addon.memory !== nothing
        product_memory = memory_addon.memory
        message_info["trace"] = string(product_memory)
        
        # Parse individual messages if available
        trace_str = string(product_memory)
        message_info["full_trace"] = trace_str
    end
    
    return message_info
end

"""
Extract benchmark statistics
"""
function extract_benchmark_stats(benchmark_callbacks)
    try
        stats_matrix = RxInfer.get_benchmark_stats(benchmark_callbacks)
        
        # Convert to DataFrame for easier handling
        df = DataFrame(
            Operation = stats_matrix[:, 1],
            Min_ns = stats_matrix[:, 2],
            Max_ns = stats_matrix[:, 3],
            Mean_ns = stats_matrix[:, 4],
            Median_ns = stats_matrix[:, 5],
            Std_ns = stats_matrix[:, 6]
        )
        
        # Convert to more readable units (microseconds)
        for col in [:Min_ns, :Max_ns, :Mean_ns, :Median_ns, :Std_ns]
            df[!, Symbol(replace(string(col), "_ns" => "_μs"))] = df[!, col] ./ 1000.0
        end
        
        return df
    catch e
        @warn "Failed to extract benchmark stats" exception=e
        return nothing
    end
end

"""
Run inference with full diagnostics
"""
function run_inference_with_diagnostics(
    model,
    data::Vector{Float64},
    prior_a::Float64,
    prior_b::Float64;
    config::DiagnosticConfig = DiagnosticConfig(),
    iterations::Int = 10,
    n_benchmark_runs::Int = 1
)
    @info "Running inference with diagnostics" config
    
    # Prepare addons
    addons_list = []
    if config.enable_memory_addon
        push!(addons_list, AddonMemory())
        @info "Memory Addon enabled - will trace message computations"
    end
    
    # Prepare callbacks
    diagnostic_callbacks = DiagnosticCallbacks(verbose = config.verbose)
    callback_tuple = create_diagnostic_callbacks(diagnostic_callbacks)
    
    # Prepare benchmark separately (RxInferBenchmarkCallbacks handles its own registration)
    benchmark_callbacks = config.enable_benchmark ? RxInferBenchmarkCallbacks() : nothing
    
    # Run inference (potentially multiple times for benchmarking)
    result = nothing
    for run in 1:n_benchmark_runs
        if n_benchmark_runs > 1 && config.verbose
            @info "Benchmark run $run/$n_benchmark_runs"
        end
        
        # Use benchmark callbacks if enabled, otherwise use diagnostic callbacks
        final_callbacks = if benchmark_callbacks !== nothing && run == 1
            # Use diagnostic callbacks for first run only
            callback_tuple
        elseif benchmark_callbacks !== nothing
            # Use benchmark callbacks for subsequent runs
            benchmark_callbacks
        else
            callback_tuple
        end
        
        result = infer(
            model = model,
            data = (y = data,),
            iterations = iterations,
            free_energy = true,  # Enable free energy tracking
            addons = length(addons_list) > 0 ? tuple(addons_list...) : (),
            callbacks = final_callbacks,
            showprogress = config.verbose && run == 1  # Only show progress on first run
        )
    end
    
    # Extract diagnostics
    memory_trace = nothing
    if config.enable_memory_addon && result !== nothing
        @info "Extracting memory trace..."
        memory_trace = extract_message_trace(result, :θ)
    end
    
    callback_trace = if config.enable_callbacks
        vcat(diagnostic_callbacks.iteration_data, diagnostic_callbacks.marginal_updates)
    else
        nothing
    end
    
    benchmark_stats = if config.enable_benchmark && benchmark_callbacks !== nothing
        @info "Extracting benchmark statistics..."
        extract_benchmark_stats(benchmark_callbacks)
    else
        nothing
    end
    
    diagnostics = DiagnosticResults(
        memory_trace,
        callback_trace,
        benchmark_stats,
        nothing  # message_history - for future use
    )
    
    return result, diagnostics
end

"""
Save diagnostic results to files
"""
function save_diagnostics(diagnostics::DiagnosticResults, output_dir::String)
    mkpath(output_dir)
    
    @info "Saving diagnostic results to $output_dir"
    
    # Save memory trace
    if diagnostics.memory_trace !== nothing
        filepath = joinpath(output_dir, "memory_trace.json")
        open(filepath, "w") do io
            JSON.print(io, diagnostics.memory_trace, 2)
        end
        @info "Saved memory trace" filepath
    end
    
    # Save callback trace
    if diagnostics.callback_trace !== nothing
        filepath = joinpath(output_dir, "callback_trace.json")
        open(filepath, "w") do io
            # Convert timestamps to strings for JSON
            trace_copy = map(diagnostics.callback_trace) do entry
                d = copy(entry)
                if haskey(d, "timestamp")
                    d["timestamp"] = string(d["timestamp"])
                end
                d
            end
            JSON.print(io, trace_copy, 2)
        end
        @info "Saved callback trace" filepath
        
        # Save separate CSVs for iteration events and marginal updates
        if !isempty(diagnostics.callback_trace)
            # Iteration events
            iteration_events = filter(x -> haskey(x, "event"), diagnostics.callback_trace)
            if !isempty(iteration_events)
                df_iter = DataFrame(iteration_events)
                csv_path = joinpath(output_dir, "iteration_events.csv")
                CSV.write(csv_path, df_iter)
                @info "Saved iteration events" filepath=csv_path
            end
            
            # Marginal updates
            marginal_updates = filter(x -> haskey(x, "variable"), diagnostics.callback_trace)
            if !isempty(marginal_updates)
                df_marg = DataFrame(marginal_updates)
                csv_path = joinpath(output_dir, "marginal_updates.csv")
                CSV.write(csv_path, df_marg)
                @info "Saved marginal updates" filepath=csv_path
            end
        end
    end
    
    # Save benchmark stats
    if diagnostics.benchmark_stats !== nothing
        filepath = joinpath(output_dir, "benchmark_stats.csv")
        CSV.write(filepath, diagnostics.benchmark_stats)
        @info "Saved benchmark statistics" filepath
        
        # Also create a summary
        summary = Dict(
            "model_creation" => Dict(
                "mean_μs" => diagnostics.benchmark_stats[1, :Mean_μs],
                "std_μs" => diagnostics.benchmark_stats[1, :Std_μs]
            ),
            "inference" => Dict(
                "mean_μs" => diagnostics.benchmark_stats[2, :Mean_μs],
                "std_μs" => diagnostics.benchmark_stats[2, :Std_μs]
            ),
            "iteration" => Dict(
                "mean_μs" => diagnostics.benchmark_stats[3, :Mean_μs],
                "std_μs" => diagnostics.benchmark_stats[3, :Std_μs]
            )
        )
        
        filepath = joinpath(output_dir, "benchmark_summary.json")
        open(filepath, "w") do io
            JSON.print(io, summary, 2)
        end
        @info "Saved benchmark summary" filepath
    end
    
    return output_dir
end

"""
Create diagnostic visualizations
"""
function visualize_message_trace(diagnostics::DiagnosticResults, output_dir::String)
    mkpath(output_dir)
    
    if diagnostics.memory_trace !== nothing
        # Save a detailed text report
        filepath = joinpath(output_dir, "message_trace_report.txt")
        open(filepath, "w") do io
            println(io, "=" ^ 80)
            println(io, "MESSAGE TRACE REPORT")
            println(io, "=" ^ 80)
            println(io)
            
            trace = diagnostics.memory_trace
            println(io, "Variable: ", trace["variable"])
            println(io, "Posterior Type: ", trace["posterior_type"])
            println(io, "Posterior Parameters: ", trace["posterior_params"])
            println(io)
            
            if haskey(trace, "full_trace")
                println(io, "Full Message Trace:")
                println(io, "-" ^ 80)
                println(io, trace["full_trace"])
            end
        end
        @info "Saved message trace report" filepath
    end
    
    if diagnostics.callback_trace !== nothing
        # Create iteration timing plot
        filepath = joinpath(output_dir, "iteration_trace_report.txt")
        open(filepath, "w") do io
            println(io, "=" ^ 80)
            println(io, "ITERATION CALLBACK TRACE")
            println(io, "=" ^ 80)
            println(io)
            
            for entry in diagnostics.callback_trace
                if haskey(entry, "event")
                    println(io, "Iteration $(entry["iteration"]) - $(entry["event"]): $(entry["timestamp"])")
                elseif haskey(entry, "variable")
                    println(io, "  Updated $(entry["variable"]): μ=$(entry["mean"]), σ=$(entry["std"])")
                end
            end
        end
        @info "Saved iteration trace report" filepath
    end
    
    return output_dir
end

end # module

