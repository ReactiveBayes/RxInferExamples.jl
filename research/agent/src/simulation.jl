# Generic Simulation Runner
# Runs simulations with any compatible agent-environment pair

using Logging
using Printf
using CSV
using DataFrames
using JSON
using Dates

include("types.jl")
include("agents/abstract_agent.jl")
include("environments/abstract_environment.jl")
include("diagnostics.jl")
include("logging.jl")
include("visualization.jl")

using .Main: StateVector, ActionVector, ObservationVector
using .Diagnostics
using .LoggingUtils: setup_logging, close_logging, ProgressBar, update!, finish!
using .Visualization

"""
SimulationConfig

Configuration for running a simulation.
"""
struct SimulationConfig
    max_steps::Int
    enable_diagnostics::Bool
    enable_logging::Bool
    verbose::Bool
    log_interval::Int
    
    function SimulationConfig(;
        max_steps::Int = 100,
        enable_diagnostics::Bool = true,
        enable_logging::Bool = true,
        verbose::Bool = false,
        log_interval::Int = 10
    )
        new(max_steps, enable_diagnostics, enable_logging, verbose, log_interval)
    end
end

"""
SimulationResult

Results from a simulation run.
"""
struct SimulationResult{S,A,O}
    states::Vector{StateVector{S}}
    actions::Vector{ActionVector{A}}
    observations::Vector{ObservationVector{O}}
    predictions::Vector{Vector{StateVector{S}}}
    diagnostics::Union{Nothing, DiagnosticsCollector}
    steps_taken::Int
    total_time::Float64
end

"""
run_simulation(agent, env, config)

Run a generic Active Inference simulation with any compatible agent-environment pair.

The type system ensures agent and environment have matching dimensions at compile time.

Args:
- agent: AbstractActiveInferenceAgent{S,A,O}
- env: AbstractEnvironment{S,A,O}
- config: SimulationConfig

Returns:
- SimulationResult with states, actions, observations, and diagnostics
"""
function run_simulation(
    agent::AbstractActiveInferenceAgent{S,A,O},
    env::AbstractEnvironment{S,A,O},
    config::SimulationConfig
) where {S,A,O}
    
    # Setup logging
    loggers = nothing
    if config.enable_logging
        loggers = setup_logging(
            verbose = config.verbose,
            structured = false,
            performance = false
        )
    end
    
    # Setup diagnostics
    diagnostics = config.enable_diagnostics ? DiagnosticsCollector() : nothing
    
    # History storage
    states = StateVector{S}[]
    actions = ActionVector{A}[]
    observations = ObservationVector{O}[]
    predictions = Vector{StateVector{S}}[]
    
    # Reset environment and agent
    initial_obs = reset!(env)
    reset!(agent)
    
    # Record initial state
    push!(states, get_state(env))
    push!(observations, initial_obs)
    
    @info "Starting simulation" max_steps=config.max_steps state_dim=S action_dim=A obs_dim=O
    
    # Progress bar
    progress = config.verbose ? ProgressBar(config.max_steps) : nothing
    
    start_time = time()
    
    # Main simulation loop
    for t in 1:config.max_steps
        # Get action from agent
        action = get_action(agent)
        push!(actions, action)
        
        # Execute in environment
        observation = step!(env, action)
        push!(observations, observation)
        
        # Get current state
        current_state = get_state(env)
        push!(states, current_state)
        
        # Get predictions
        preds = get_predictions(agent)
        push!(predictions, preds)
        
        # Perform inference
        if diagnostics !== nothing
            start_timer!(diagnostics.performance_profiler, "inference")
        end
        
        step!(agent, observation, action)
        
        if diagnostics !== nothing
            stop_timer!(diagnostics.performance_profiler, "inference")
            
            # Record diagnostics
            if t % 10 == 0
                trace_memory!(diagnostics.memory_tracer)
            end
        end
        
        # Slide planning window
        slide!(agent)
        
        # Log periodically
        if config.verbose && t % config.log_interval == 0
            @info "step" t=t action=Vector(action) observation=Vector(observation) state=Vector(current_state)
        end
        
        # Update progress
        if progress !== nothing
            update!(progress, t)
        end
    end
    
    if progress !== nothing
        finish!(progress)
    end
    
    total_time = time() - start_time
    
    @info "Simulation complete" steps=config.max_steps time=round(total_time, digits=3)
    
    # Print diagnostics if enabled
    if diagnostics !== nothing && config.verbose
        print_diagnostics_report(diagnostics)
    end
    
    # Close logging
    if loggers !== nothing
        close_logging(loggers)
    end
    
    return SimulationResult(
        states,
        actions,
        observations,
        predictions,
        diagnostics,
        config.max_steps,
        total_time
    )
end

"""
save_simulation_outputs(result, output_dir, state_dim, action_dim, goal_state; 
                       generate_visualizations=true, generate_animations=true)

Save comprehensive simulation outputs to a specified directory.

Saves:
- Trajectory data (CSV)
- Summary statistics (CSV)
- Diagnostics (JSON)
- Full result (JSON)
- Visualizations (PNG)
- Animations (GIF)
"""
function save_simulation_outputs(
    result::SimulationResult{S,A,O},
    output_dir::String,
    goal_state=nothing;
    generate_visualizations::Bool=true,
    generate_animations::Bool=true
) where {S,A,O}
    
    println("\n" * "="^70)
    println("SAVING SIMULATION OUTPUTS")
    println("="^70)
    println("Output directory: $output_dir")
    
    # Create output subdirectories
    data_dir = joinpath(output_dir, "data")
    results_dir = joinpath(output_dir, "results")
    diagnostics_dir = joinpath(output_dir, "diagnostics")
    plots_dir = joinpath(output_dir, "plots")
    animations_dir = joinpath(output_dir, "animations")
    
    mkpath(data_dir)
    mkpath(results_dir)
    mkpath(diagnostics_dir)
    mkpath(plots_dir)
    mkpath(animations_dir)
    
    # 1. Save trajectory data
    println("\n[Trajectory Data]")
    
    # Build trajectory DataFrame based on dimensionality
    if S == 1
        trajectory_df = DataFrame(
            step = 1:length(result.states),
            position = [s[1] for s in result.states],
            action = vcat([0.0], [a[1] for a in result.actions])
        )
    elseif S == 2
        trajectory_df = DataFrame(
            step = 1:length(result.states),
            position = [s[1] for s in result.states],
            velocity = [s[2] for s in result.states],
            action = vcat([0.0], [a[1] for a in result.actions])
        )
    else
        # Generic multi-dimensional case
        trajectory_df = DataFrame(step = 1:length(result.states))
        for i in 1:S
            trajectory_df[!, Symbol("state_$i")] = [s[i] for s in result.states]
        end
        for i in 1:A
            trajectory_df[!, Symbol("action_$i")] = vcat([0.0], [a[i] for a in result.actions])
        end
    end
    
    CSV.write(joinpath(data_dir, "trajectory.csv"), trajectory_df)
    println("  ✓ Saved: data/trajectory.csv")
    
    # 2. Save observations
    if S == 1
        obs_df = DataFrame(
            step = 1:length(result.observations),
            obs_position = [o[1] for o in result.observations]
        )
    elseif S == 2
        obs_df = DataFrame(
            step = 1:length(result.observations),
            obs_position = [o[1] for o in result.observations],
            obs_velocity = [o[2] for o in result.observations]
        )
    else
        obs_df = DataFrame(step = 1:length(result.observations))
        for i in 1:O
            obs_df[!, Symbol("obs_$i")] = [o[i] for o in result.observations]
        end
    end
    
    CSV.write(joinpath(data_dir, "observations.csv"), obs_df)
    println("  ✓ Saved: data/observations.csv")
    
    # 3. Save summary statistics
    println("\n[Summary Statistics]")
    
    metrics = ["steps_taken", "total_time", "avg_time_per_step"]
    values = [
        result.steps_taken,
        result.total_time,
        result.total_time / result.steps_taken
    ]
    
    # Add state-specific metrics
    if S == 1
        final_pos = result.states[end][1]
        push!(metrics, "final_position")
        push!(values, final_pos)
        
        if goal_state !== nothing
            goal_pos = goal_state[1]
            dist = abs(final_pos - goal_pos)
            push!(metrics, "goal_position")
            push!(values, goal_pos)
            push!(metrics, "distance_to_goal")
            push!(values, dist)
            push!(metrics, "goal_reached")
            push!(values, dist < 0.1 ? 1.0 : 0.0)
        end
    elseif S == 2
        final_pos = result.states[end][1]
        final_vel = result.states[end][2]
        push!(metrics, "final_position")
        push!(values, final_pos)
        push!(metrics, "final_velocity")
        push!(values, final_vel)
        
        if goal_state !== nothing
            goal_pos = goal_state[1]
            goal_vel = goal_state[2]
            dist_pos = abs(final_pos - goal_pos)
            dist_vel = abs(final_vel - goal_vel)
            push!(metrics, "goal_position")
            push!(values, goal_pos)
            push!(metrics, "goal_velocity")
            push!(values, goal_vel)
            push!(metrics, "distance_to_goal_position")
            push!(values, dist_pos)
            push!(metrics, "distance_to_goal_velocity")
            push!(values, dist_vel)
            push!(metrics, "goal_reached")
            push!(values, (dist_pos < 0.1 && dist_vel < 0.05) ? 1.0 : 0.0)
        end
    end
    
    summary_df = DataFrame(metric = metrics, value = values)
    CSV.write(joinpath(results_dir, "summary.csv"), summary_df)
    println("  ✓ Saved: results/summary.csv")
    
    # 4. Save diagnostics
    if result.diagnostics !== nothing
        println("\n[Diagnostics]")
        diag_summary = get_comprehensive_summary(result.diagnostics)
        
        open(joinpath(diagnostics_dir, "diagnostics.json"), "w") do io
            JSON.print(io, diag_summary, 2)
        end
        println("  ✓ Saved: diagnostics/diagnostics.json")
        
        # Save detailed diagnostics
        if haskey(diag_summary["performance"], "operations")
            perf_data = diag_summary["performance"]["operations"]
            open(joinpath(diagnostics_dir, "performance.json"), "w") do io
                JSON.print(io, perf_data, 2)
            end
            println("  ✓ Saved: diagnostics/performance.json")
        end
    end
    
    # 5. Save full result metadata
    println("\n[Metadata]")
    metadata = Dict(
        "timestamp" => string(now()),
        "state_dim" => S,
        "action_dim" => A,
        "observation_dim" => O,
        "steps_taken" => result.steps_taken,
        "total_time" => result.total_time,
        "avg_time_per_step" => result.total_time / result.steps_taken,
        "diagnostics_enabled" => result.diagnostics !== nothing
    )
    
    if goal_state !== nothing
        metadata["goal_state"] = Vector(goal_state)
    end
    
    open(joinpath(output_dir, "metadata.json"), "w") do io
        JSON.print(io, metadata, 2)
    end
    println("  ✓ Saved: metadata.json")
    
    # 6. Generate visualizations
    if generate_visualizations
        println("\n[Visualizations]")
        try
            plots_created = generate_all_visualizations(result, plots_dir, S)
            println("  ✓ Created: $(length(plots_created)) static plots")
            
            # Create animations in animations_dir
            if generate_animations
                println("\n[Animations]")
                if S == 1
                    animate_trajectory_1d(result, animations_dir)
                    println("  ✓ Created: animations/trajectory_1d.gif")
                elseif S == 2
                    animate_trajectory_2d(result, animations_dir)
                    println("  ✓ Created: animations/trajectory_2d.gif")
                end
            end
        catch e
            @warn "Visualization generation failed" exception=e
            println("  ⚠ Visualization generation failed: $e")
        end
    end
    
    # 7. Create summary report
    println("\n[Summary Report]")
    report_path = joinpath(output_dir, "REPORT.md")
    
    open(report_path, "w") do io
        write(io, "# Simulation Report\n\n")
        write(io, "**Generated:** $(now())\n\n")
        write(io, "---\n\n")
        
        write(io, "## Configuration\n\n")
        write(io, "- **State Dimension:** $S\n")
        write(io, "- **Action Dimension:** $A\n")
        write(io, "- **Observation Dimension:** $O\n")
        write(io, "- **Steps:** $(result.steps_taken)\n")
        write(io, "- **Total Time:** $(round(result.total_time, digits=3))s\n")
        write(io, "- **Avg Time/Step:** $(round(result.total_time/result.steps_taken, digits=4))s\n\n")
        
        if goal_state !== nothing
            write(io, "- **Goal State:** $(Vector(goal_state))\n")
            write(io, "- **Final State:** $(Vector(result.states[end]))\n\n")
        end
        
        write(io, "---\n\n")
        write(io, "## Results\n\n")
        
        # Read summary and display
        summary_df = CSV.read(joinpath(results_dir, "summary.csv"), DataFrame)
        write(io, "| Metric | Value |\n")
        write(io, "|--------|-------|\n")
        for row in eachrow(summary_df)
            write(io, "| $(row.metric) | $(round(row.value, digits=6)) |\n")
        end
        write(io, "\n")
        
        if result.diagnostics !== nothing
            write(io, "---\n\n")
            write(io, "## Diagnostics\n\n")
            
            diag_summary = get_comprehensive_summary(result.diagnostics)
            
            # Memory
            if diag_summary["memory"]["measurements"] > 0
                mem = diag_summary["memory"]
                write(io, "### Memory Usage\n\n")
                write(io, "- Peak: $(round(mem["peak_memory_mb"], digits=2)) MB\n")
                write(io, "- Average: $(round(mem["avg_memory_mb"], digits=2)) MB\n")
                write(io, "- Growth: $(round(mem["memory_growth"], digits=2)) MB\n")
                write(io, "- GC Time: $(round(mem["total_gc_time"], digits=3))s\n\n")
            end
            
            # Performance
            if haskey(diag_summary["performance"], "operations")
                write(io, "### Performance\n\n")
                for (op, stats) in diag_summary["performance"]["operations"]
                    write(io, "**$op:**\n")
                    write(io, "- Count: $(stats["count"])\n")
                    write(io, "- Total Time: $(round(stats["total_time"], digits=3))s\n")
                    write(io, "- Avg Time: $(round(stats["avg_time"], digits=4))s\n")
                    write(io, "- Min/Max: $(round(stats["min_time"], digits=4))s / $(round(stats["max_time"], digits=4))s\n\n")
                end
            end
        end
        
        write(io, "---\n\n")
        write(io, "## Outputs\n\n")
        write(io, "### Data Files\n")
        write(io, "- `data/trajectory.csv` - Full state trajectory\n")
        write(io, "- `data/observations.csv` - Observation sequence\n\n")
        
        write(io, "### Results\n")
        write(io, "- `results/summary.csv` - Summary statistics\n\n")
        
        if result.diagnostics !== nothing
            write(io, "### Diagnostics\n")
            write(io, "- `diagnostics/diagnostics.json` - Comprehensive diagnostics\n")
            write(io, "- `diagnostics/performance.json` - Performance metrics\n\n")
        end
        
        if generate_visualizations
            write(io, "### Visualizations\n")
            write(io, "- `plots/trajectory_$(S)d.png` - Trajectory plot\n")
            if S == 2
                write(io, "- `plots/mountain_car_landscape.png` - Landscape visualization\n")
            end
            write(io, "- `plots/diagnostics.png` - Diagnostics plots\n\n")
            
            if generate_animations
                write(io, "### Animations\n")
                write(io, "- `animations/trajectory_$(S)d.gif` - Animated trajectory\n\n")
            end
        end
        
        write(io, "---\n\n")
        write(io, "**Framework:** Generic Agent-Environment Framework v0.1.1\n")
    end
    
    println("  ✓ Saved: REPORT.md")
    
    println("\n" * "="^70)
    println("✅ ALL OUTPUTS SAVED SUCCESSFULLY")
    println("="^70)
    println("\nOutput location: $output_dir")
    println()
    
    return output_dir
end

# Export
export SimulationConfig, SimulationResult, run_simulation, save_simulation_outputs

