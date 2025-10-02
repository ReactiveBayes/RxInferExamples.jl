#!/usr/bin/env julia
# Example: Mountain Car with Generic Active Inference Agent
# Demonstrates how to use the car_agent framework for a specific problem
# 
# This example produces COMPREHENSIVE outputs in all directories:
# - logs/ : Application logs, performance metrics, memory traces
# - data/ : State trajectories, actions, observations (CSV & JSON)
# - plots/ : Multiple visualizations of results
# - animations/ : GIF animation of agent behavior
# - diagnostics/ : Detailed diagnostic data and reports
# - results/ : Summary statistics and final report

using LinearAlgebra
using HypergeometricFunctions: _₂F₁
using Plots
using CSV
using DataFrames
using JSON
using Dates

# Include the generic agent framework (check if already loaded from run.jl)
if !isdefined(Main, :Config)
    include("../config.jl")
end
include("../src/diagnostics.jl")
include("../src/logging.jl")

using .Config
using .Diagnostics
using .LoggingUtils
using RxInfer
import RxInfer.ReactiveMP: getrecent, messageout

@doc """
Mountain Car example using the generic Active Inference agent framework.

This example shows how to:
1. Define problem-specific physics (transition and control functions)
2. Create a real Active Inference agent with RxInfer
3. Run a simulation loop
4. Collect diagnostics and visualize results
"""

# ==================== RXINFER MODEL (TOP-LEVEL) ====================

# RxInfer generative model for mountain car
# This @model must be defined at top level (RxInfer v4+ requirement)
@model function mountain_car_model(m_u, V_u, m_x, V_x, m_s_t_min, V_s_t_min, T, Fg, Fa, Ff, engine_force_limit)
    
    # Transition function modeling transition due to gravity and friction
    g = (s_t_min::AbstractVector) -> begin 
        s_t = similar(s_t_min) # Next state
        s_t[2] = s_t_min[2] + Fg(s_t_min[1]) + Ff(s_t_min[2]) # Update velocity
        s_t[1] = s_t_min[1] + s_t[2] # Update position
        return s_t
    end
    
    # Function for modeling engine control
    h = (u::AbstractVector) -> [0.0, Fa(u[1])] 
    
    # Inverse engine force, from change in state to corresponding engine force
    h_inv = (delta_s_dot::AbstractVector) -> [atanh(clamp(delta_s_dot[2], -engine_force_limit+1e-3, engine_force_limit-1e-3)/engine_force_limit)] 
    
    # Internal model parameters
    Gamma = 1e4*diageye(2) # Transition precision
    Theta = 1e-4*diageye(2) # Observation variance

    s_t_min ~ MvNormal(mean = m_s_t_min, cov = V_s_t_min)
    s_k_min = s_t_min

    local s
    
    for k in 1:T
        u[k] ~ MvNormal(mean = m_u[k], cov = V_u[k])
        u_h_k[k] ~ h(u[k]) where { meta = DeltaMeta(method = Linearization(), inverse = h_inv) }
        s_g_k[k] ~ g(s_k_min) where { meta = DeltaMeta(method = Linearization()) }
        u_s_sum[k] ~ s_g_k[k] + u_h_k[k]
        s[k] ~ MvNormal(mean = u_s_sum[k], precision = Gamma)
        x[k] ~ MvNormal(mean = s[k], cov = Theta)
        x[k] ~ MvNormal(mean = m_x[k], cov = V_x[k]) # goal
        s_k_min = s[k]
    end
    
    return (s, )
end

# ==================== PROBLEM-SPECIFIC PHYSICS ====================

@doc """
Create physics functions for mountain car environment.
"""
function create_mountain_car_physics(; engine_force_limit::Float64 = 0.04,
                                     friction_coefficient::Float64 = 0.1)
    
    # Engine force function
    Fa = (a::Real) -> engine_force_limit * tanh(a)
    
    # Friction force function
    Ff = (y_dot::Real) -> -friction_coefficient * y_dot
    
    # Gravitational force function
    Fg = (y::Real) -> begin
        if y < 0
            return -0.05*(2*y + 1)
        else
            return -0.05*((1 + 5*y^2)^(-0.5) + (y^2)*(1 + 5*y^2)^(-3/2) + (y^4)/16)
        end
    end
    
    # Height function for visualization
    height = (x::Float64) -> begin
        if x < 0
            h = x^2 + x
        else
            h = x * _₂F₁(0.5, 0.5, 1.5, -5*x^2) +
                x^3 * _₂F₁(1.5, 1.5, 2.5, -5*x^2) / 3 +
                x^5 / 80
        end
        return 0.05*h
    end
    
    return (Fa, Ff, Fg, height)
end

# ==================== AGENT SETUP ====================

@doc """
Create mountain car Active Inference agent with real RxInfer inference.
"""
function create_mountain_car_agent(
    Fa::Function, Ff::Function, Fg::Function;
    initial_position::Float64 = -0.5,
    initial_velocity::Float64 = 0.0,
    goal_position::Float64 = 0.5,
    goal_velocity::Float64 = 0.0,
    planning_horizon::Int = 20,
    engine_force_limit::Float64 = 0.04
)
    T = planning_horizon
    huge = 1e6
    tiny = 1e-6
    
    # Control priors
    Epsilon = fill(huge, 1, 1)  # Control prior variance
    m_u = Vector{Float64}[[0.0] for k=1:T]
    V_u = Matrix{Float64}[Epsilon for k=1:T]
    
    # Goal priors
    Sigma = 1e-4*diageye(2)  # Goal prior variance
    x_target = [goal_position, goal_velocity]
    m_x = [zeros(2) for k=1:T]
    V_x = [huge*diageye(2) for k=1:T]
    V_x[end] = Sigma  # Set prior to reach goal at t=T
    
    # Initial state belief (mutable for closures)
    state_belief = Ref((m = [initial_position, initial_velocity], 
                       V = tiny * diageye(2)))
    
    # Inference result storage (use Ref for mutable closure capture)
    result_ref = Ref{Union{Nothing, Any}}(nothing)
    
    # The compute function performs inference
    compute = (upsilon_t::Float64, y_hat_t::Vector{Float64}) -> begin
        m_u[1] = [upsilon_t]  # Register action
        V_u[1] = fill(tiny, 1, 1)  # Clamp control prior
        
        m_x[1] = y_hat_t  # Register observation
        V_x[1] = tiny*diageye(2)  # Clamp goal prior
        
        data = Dict(:m_u => m_u, 
                    :V_u => V_u, 
                    :m_x => m_x, 
                    :V_x => V_x,
                    :m_s_t_min => state_belief[].m,
                    :V_s_t_min => state_belief[].V)
        
        model = mountain_car_model(T = T, Fg = Fg, Fa = Fa, Ff = Ff, 
                                   engine_force_limit = engine_force_limit)
        result_ref[] = infer(model = model, data = data)
        
        return result_ref[]
    end
    
    # The act function returns the inferred action
    act = () -> begin
        if result_ref[] !== nothing
            return mode(result_ref[].posteriors[:u][2])[1]
        else
            return 0.0
        end
    end
    
    # The future function returns predicted states
    future = () -> begin
        if result_ref[] !== nothing
            return getindex.(mode.(result_ref[].posteriors[:s]), 1)
        else
            return zeros(T)
        end
    end
    
    # The slide function shifts the planning window
    slide = () -> begin
        if result_ref[] !== nothing
            model_obj = RxInfer.getmodel(result_ref[].model)
            (s,) = RxInfer.getreturnval(model_obj)
            varref = RxInfer.getvarref(model_obj, s)
            var = RxInfer.getvariable(varref)
            
            slide_msg_idx = 3
            (m_new, V_new) = mean_cov(getrecent(messageout(var[2], slide_msg_idx)))
            state_belief[] = (m = m_new, V = V_new)
        end
        
        m_u = circshift(m_u, -1)
        m_u[end] = [0.0]
        V_u = circshift(V_u, -1)
        V_u[end] = Epsilon
        
        m_x = circshift(m_x, -1)
        m_x[end] = x_target
        V_x = circshift(V_x, -1)
        V_x[end] = Sigma
    end
    
    # Return agent interface
    return (compute = compute, act = act, slide = slide, future = future,
            result = () -> result_ref[], m_s = () -> state_belief[].m)
end

# ==================== ENVIRONMENT SIMULATION ====================

@doc """
Simple environment simulator for mountain car.
"""
mutable struct MountainCarEnvironment
    state::Vector{Float64}
    Fa::Function
    Ff::Function
    Fg::Function
    
    function MountainCarEnvironment(initial_state::Vector{Float64},
                                    Fa::Function, Ff::Function, Fg::Function)
        new(copy(initial_state), Fa, Ff, Fg)
    end
end

@doc """
Execute action in environment.
"""
function execute_action!(env::MountainCarEnvironment, action::Vector{Float64})
    # Compute next state
    v_new = env.state[2] + env.Fg(env.state[1]) + env.Ff(env.state[2]) + env.Fa(action[1])
    x_new = env.state[1] + v_new
    
    env.state = [x_new, v_new]
    return copy(env.state)
end

# ==================== MAIN SIMULATION ====================

@doc """
Run mountain car simulation with Active Inference agent.
"""
function run_mountain_car_simulation(;
    max_steps::Int = 100,
    verbose::Bool = true,
    enable_diagnostics::Bool = true
)
    println("="^60)
    println("Mountain Car Active Inference Example")
    println("="^60)
    
    # Setup logging
    loggers = setup_logging(verbose=verbose, structured=true, performance=true)
    
    # Create physics
    Fa, Ff, Fg, height = create_mountain_car_physics()
    
    # Create agent
    initial_position = -0.5
    initial_velocity = 0.0
    agent = create_mountain_car_agent(
        Fa, Ff, Fg;
        initial_position = initial_position,
        initial_velocity = initial_velocity,
        goal_position = 0.5,
        goal_velocity = 0.0,
        planning_horizon = 15  # Reduced for faster inference
    )
    
    # Create environment
    env = MountainCarEnvironment([initial_position, initial_velocity], Fa, Ff, Fg)
    
    # Setup diagnostics
    diagnostics = enable_diagnostics ? DiagnosticsCollector() : nothing
    
    # Simulation history
    states = Vector{Vector{Float64}}()
    actions = Vector{Float64}()
    predictions = Vector{Vector{Float64}}()
    
    # Initial state
    push!(states, copy(env.state))
    
    # Simulation loop
    @info "Starting simulation" max_steps=max_steps
    progress = ProgressBar(max_steps)
    
    total_inference_time = 0.0
    
    for t in 1:max_steps
        # Get action from agent
        action_val = agent.act()
        push!(actions, action_val)
        
        # Execute in environment
        observation = execute_action!(env, [action_val])
        push!(states, copy(observation))
        
        # Get predictions
        preds = agent.future()
        push!(predictions, preds)
        
        # Perform inference
        inference_time = @elapsed begin
            agent.compute(action_val, observation)
        end
        total_inference_time += inference_time
        
        if diagnostics !== nothing
            # Trace memory periodically
            if t % LOGGING.memory_trace_interval == 0
                trace_memory!(diagnostics.memory_tracer)
            end
            
            # Record diagnostics
            m_s = agent.m_s()
            record_belief!(diagnostics.belief_tracker, t, m_s, zeros(2,2))
            # Note: Skipping prediction tracking due to dimension mismatch (positions vs full state)
        end
        
        # Slide planning window
        agent.slide()
        
        # Log step
        if verbose && t % 10 == 0
            @info "agent_step" step=t action=action_val observation=observation
        end
        
        # Update progress
        LoggingUtils.update!(progress, t)
        
        # Check goal achievement
        goal_distance = abs(env.state[1] - 0.5)
        if goal_distance < 0.05
            @info "Goal reached!" step=t position=env.state[1]
            break
        end
    end
    
    LoggingUtils.finish!(progress)
    
    # Print results
    println("\n" * "="^60)
    println("Simulation Complete")
    println("="^60)
    println("Steps: $(length(actions))")
    println("Total inference time: $(round(total_inference_time, digits=4))s")
    println("Avg inference time: $(round(total_inference_time/length(actions), digits=6))s")
    println("Final position: $(states[end][1])")
    println("Final velocity: $(states[end][2])")
    
    if diagnostics !== nothing
        print_diagnostics_report(diagnostics)
    end
    
    # Close logging
    close_logging(loggers)
    
    return (agent, env, states, actions, predictions, diagnostics)
end

# ==================== DATA EXPORT ====================

@doc """
Export simulation data to outputs/data/ directory.
"""
function export_simulation_data(states, actions, diagnostics)
    println("\n📊 Exporting Simulation Data...")
    
    # Create DataFrame for trajectory
    positions = [s[1] for s in states]
    velocities = [s[2] for s in states]
    # Align actions with states (actions are one less)
    action_values = [0.0; actions]  # Pad with zero for initial state
    
    df_trajectory = DataFrame(
        step = 1:length(states),
        position = positions,
        velocity = velocities,
        action = action_values[1:length(states)]
    )
    
    # Save trajectory as CSV
    csv_path = joinpath(Config.OUTPUTS.data_dir, "trajectory.csv")
    CSV.write(csv_path, df_trajectory)
    file_size = filesize(csv_path) / 1024  # KB
    println("   ✓ trajectory.csv ($(round(file_size, digits=2)) KB)")
    
    # Export as JSON for structured access
    trajectory_json = Dict(
        "states" => [[s[1], s[2]] for s in states],
        "actions" => actions,
        "metadata" => Dict(
            "num_steps" => length(states),
            "initial_position" => states[1][1],
            "final_position" => states[end][1],
            "goal_position" => 0.5
        )
    )
    
    json_path = joinpath(Config.OUTPUTS.data_dir, "trajectory.json")
    open(json_path, "w") do io
        JSON.print(io, trajectory_json, 2)
    end
    file_size = filesize(json_path) / 1024
    println("   ✓ trajectory.json ($(round(file_size, digits=2)) KB)")
    
    # Export diagnostics data if available
    if diagnostics !== nothing && !isempty(diagnostics.belief_tracker.belief_means)
        beliefs_data = []
        for i in 1:length(diagnostics.belief_tracker.timestamps)
            push!(beliefs_data, Dict(
                "step" => diagnostics.belief_tracker.timestamps[i],
                "mean" => diagnostics.belief_tracker.belief_means[i],
                "cov_trace" => diagnostics.belief_tracker.uncertainty_trace[i],
                "belief_change" => i > 1 ? diagnostics.belief_tracker.belief_changes[i-1] : 0.0
            ))
        end
        
        beliefs_path = joinpath(Config.OUTPUTS.data_dir, "beliefs.json")
        open(beliefs_path, "w") do io
            JSON.print(io, beliefs_data, 2)
        end
        file_size = filesize(beliefs_path) / 1024
        println("   ✓ beliefs.json ($(round(file_size, digits=2)) KB)")
    end
    println("   → Saved to outputs/data/ ($(length(states)) states, $(length(actions)) actions)")
end

# ==================== VISUALIZATION ====================

@doc """
Create comprehensive visualizations and save to outputs/plots/.
"""
function create_comprehensive_plots(states, actions, diagnostics, height)
    println("\n📈 Creating Visualizations...")
    
    # Extract data
    positions = [s[1] for s in states]
    velocities = [s[2] for s in states]
    action_values = actions  # Already Float64 vector
    
    # Plot 1: Mountain Car Trajectory
    x_range = range(-2.0, 2.0, length=400)
    y_range = [height(x) for x in x_range]
    
    p1 = plot(x_range, y_range, 
             title="Mountain Car Trajectory",
             xlabel="Position", ylabel="Height",
             label="Landscape", color=:black, linewidth=2,
             legend=:topright)
    
    trajectory_heights = [height(p) for p in positions]
    plot!(p1, positions, trajectory_heights,
          label="Agent Path", color=:blue, linewidth=2, alpha=0.7)
    
    scatter!(p1, [positions[1]], [height(positions[1])],
            label="Start", color=:green, markersize=8)
    scatter!(p1, [0.5], [height(0.5)],
            label="Goal", color=:red, markersize=10, markershape=:star)
    
    plot_path1 = joinpath(Config.OUTPUTS.plots_dir, "trajectory.png")
    savefig(p1, plot_path1)
    file_size = filesize(plot_path1) / 1024
    println("   ✓ trajectory.png ($(round(file_size, digits=2)) KB)")
    
    # Plot 2: State Evolution
    p2 = plot(1:length(positions), positions,
             title="Position Over Time",
             xlabel="Step", ylabel="Position",
             label="Position", color=:blue, linewidth=2,
             legend=:right)
    hline!(p2, [0.5], label="Goal", color=:red, linestyle=:dash, linewidth=2)
    
    plot_path2 = joinpath(Config.OUTPUTS.plots_dir, "position_evolution.png")
    savefig(p2, plot_path2)
    file_size = filesize(plot_path2) / 1024
    println("   ✓ position_evolution.png ($(round(file_size, digits=2)) KB)")
    
    # Plot 3: Velocity
    p3 = plot(1:length(velocities), velocities,
             title="Velocity Over Time",
             xlabel="Step", ylabel="Velocity",
             label="Velocity", color=:purple, linewidth=2)
    
    plot_path3 = joinpath(Config.OUTPUTS.plots_dir, "velocity_evolution.png")
    savefig(p3, plot_path3)
    file_size = filesize(plot_path3) / 1024
    println("   ✓ velocity_evolution.png ($(round(file_size, digits=2)) KB)")
    
    # Plot 4: Actions
    p4 = plot(1:length(action_values), action_values,
             title="Control Actions Over Time",
             xlabel="Step", ylabel="Action (Force)",
             label="Action", color=:orange, linewidth=2)
    
    plot_path4 = joinpath(Config.OUTPUTS.plots_dir, "actions.png")
    savefig(p4, plot_path4)
    file_size = filesize(plot_path4) / 1024
    println("   ✓ actions.png ($(round(file_size, digits=2)) KB)")
    
    # Plot 5: Phase Space
    p5 = plot(positions, velocities,
             title="Phase Space Trajectory",
             xlabel="Position", ylabel="Velocity",
             label="Trajectory", color=:blue, linewidth=2,
             arrow=true)
    scatter!(p5, [positions[1]], [velocities[1]],
            label="Start", color=:green, markersize=8)
    scatter!(p5, [positions[end]], [velocities[end]],
            label="End", color=:red, markersize=8)
    
    plot_path5 = joinpath(Config.OUTPUTS.plots_dir, "phase_space.png")
    savefig(p5, plot_path5)
    file_size = filesize(plot_path5) / 1024
    println("   ✓ phase_space.png ($(round(file_size, digits=2)) KB)")
    
    # Plot 6: Combined Summary
    p_combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900),
                     plot_title="Mountain Car Active Inference - Complete Results")
    
    plot_path6 = joinpath(Config.OUTPUTS.plots_dir, "complete_summary.png")
    savefig(p_combined, plot_path6)
    file_size = filesize(plot_path6) / 1024
    println("   ✓ complete_summary.png ($(round(file_size, digits=2)) KB)")
    println("   → Saved 6 plots to outputs/plots/")
    
    return p_combined
end

@doc """
Create animation of agent behavior and save to outputs/animations/.
"""
function create_animation(states, height)
    println("\n🎬 Creating Animation...")
    
    positions = [s[1] for s in states]
    
    # Create landscape
    x_range = range(-2.0, 2.0, length=400)
    y_range = [height(x) for x in x_range]
    
    print("   Rendering frames: ")
    anim = @animate for i in 1:length(positions)
        # Plot landscape
        plot(x_range, y_range,
             title="Mountain Car - Step $i/$(length(positions))",
             xlabel="Position", ylabel="Height",
             label="Landscape", color=:black, linewidth=2,
             ylim=(-0.5, 0.5), legend=:topright)
        
        # Plot trajectory so far
        if i > 1
            traj_heights = [height(p) for p in positions[1:i]]
            plot!(positions[1:i], traj_heights,
                  label="Path", color=:blue, linewidth=1, alpha=0.5)
        end
        
        # Plot car position
        car_height = height(positions[i])
        scatter!([positions[i]], [car_height],
                label="Car", color=:green, markersize=10, markershape=:circle)
        
        # Plot goal
        scatter!([0.5], [height(0.5)],
                label="Goal", color=:red, markersize=10, markershape=:star)
        
        # Progress indicator
        if i % 10 == 0 || i == length(positions)
            print("$(i) ")
        end
    end
    println("✓")
    
    print("   Encoding GIF...")
    gif_path = joinpath(Config.OUTPUTS.animations_dir, "mountain_car.gif")
    gif(anim, gif_path, fps=Config.VISUALIZATION.animation_fps)
    file_size = filesize(gif_path) / 1024
    println(" ✓")
    println("   ✓ mountain_car.gif ($(round(file_size, digits=2)) KB)")
    println("   → Saved $(length(positions)) frames at $(Config.VISUALIZATION.animation_fps) fps to outputs/animations/")
end

# ==================== RESULTS EXPORT ====================

@doc """
Export comprehensive results summary to outputs/results/.
"""
function export_results_summary(agent, states, actions, diagnostics)
    println("\n📋 Generating Results Summary...")
    
    positions = [s[1] for s in states]
    velocities = [s[2] for s in states]
    
    # Compute statistics
    final_position = states[end][1]
    goal_reached = abs(final_position - 0.5) < 0.05
    num_steps = length(states)
    
    total_distance_traveled = sum(abs(positions[i] - positions[i-1]) for i in 2:length(positions))
    max_velocity = maximum(abs.(velocities))
    total_energy = sum(abs.(actions))
    
    # Create summary
    summary = Dict(
        "simulation" => Dict(
            "timestamp" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
            "num_steps" => num_steps,
            "goal_reached" => goal_reached,
            "final_position" => final_position,
            "goal_position" => 0.5,
            "goal_distance" => abs(final_position - 0.5)
        ),
        "performance" => Dict(
            "total_distance_traveled" => total_distance_traveled,
            "max_velocity" => max_velocity,
            "total_control_effort" => total_energy,
            "avg_control_effort" => total_energy / num_steps
        ),
        "agent" => Dict(
            "planning_horizon" => 20,
            "state_dim" => 2,
            "action_dim" => 1,
            "total_steps" => num_steps
        )
    )
    
    # Add diagnostics if available
    if diagnostics !== nothing
        memory_summary = get_memory_summary(diagnostics.memory_tracer)
        perf_summary = get_performance_summary(diagnostics.performance_profiler)
        
        summary["diagnostics"] = Dict(
            "memory" => memory_summary,
            "performance" => perf_summary
        )
    end
    
    # Save as JSON
    json_path = joinpath(Config.OUTPUTS.results_dir, "simulation_summary.json")
    open(json_path, "w") do io
        JSON.print(io, summary, 2)
    end
    file_size = filesize(json_path) / 1024
    println("   ✓ simulation_summary.json ($(round(file_size, digits=2)) KB)")
    
    # Create human-readable report
    report = """
    ============================================================
    MOUNTAIN CAR SIMULATION - FINAL REPORT
    ============================================================
    
    Timestamp: $(summary["simulation"]["timestamp"])
    
    SIMULATION RESULTS
    ------------------
    Steps Completed:     $(summary["simulation"]["num_steps"])
    Goal Reached:        $(summary["simulation"]["goal_reached"] ? "YES ✓" : "NO ✗")
    Final Position:      $(round(summary["simulation"]["final_position"], digits=4))
    Goal Position:       $(summary["simulation"]["goal_position"])
    Distance to Goal:    $(round(summary["simulation"]["goal_distance"], digits=4))
    
    PERFORMANCE METRICS
    -------------------
    Total Distance:      $(round(summary["performance"]["total_distance_traveled"], digits=4))
    Max Velocity:        $(round(summary["performance"]["max_velocity"], digits=4))
    Total Control:       $(round(summary["performance"]["total_control_effort"], digits=4))
    Avg Control:         $(round(summary["performance"]["avg_control_effort"], digits=4))
    
    AGENT CONFIGURATION
    -------------------
    Planning Horizon:    $(summary["agent"]["planning_horizon"])
    State Dimension:     $(summary["agent"]["state_dim"])
    Action Dimension:    $(summary["agent"]["action_dim"])
    Total Steps:         $(summary["agent"]["total_steps"])
    
    ============================================================
    """
    
    report_path = joinpath(Config.OUTPUTS.results_dir, "simulation_report.txt")
    open(report_path, "w") do io
        write(io, report)
    end
    file_size = filesize(report_path) / 1024
    println("   ✓ simulation_report.txt ($(round(file_size, digits=2)) KB) → outputs/results/")
    
    # Print to console
    println(report)
end

# ==================== RUN EXAMPLE ====================

# Run if executed directly
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    println("\n" * "="^70)
    println("MOUNTAIN CAR ACTIVE INFERENCE - COMPREHENSIVE SIMULATION")
    println("="^70)
    println("\nThis example will generate outputs in ALL directories:")
    println("  📁 outputs/logs/         - Application logs and performance data")
    println("  📁 outputs/data/         - Trajectory data (CSV & JSON)")
    println("  📁 outputs/plots/        - Multiple visualization plots")
    println("  📁 outputs/animations/   - Animated GIF of agent behavior")
    println("  📁 outputs/diagnostics/  - Detailed diagnostic reports")
    println("  📁 outputs/results/      - Summary statistics and reports")
    println("\n" * "="^70 * "\n")
    
    # Ensure all output directories exist
    dirs_created = Config.ensure_output_directories()
    println("✓ Initialized $(length(dirs_created)) output directories\n")
    
    # Run simulation with comprehensive logging
    println("🚗 Running Active Inference Simulation...")
    println("   Configuration: Horizon=15, Max Steps=50")
    println("")
    agent, env, states, actions, predictions, diagnostics = run_mountain_car_simulation(
        max_steps = 50,  # Reduced for faster demonstration
        verbose = true,
        enable_diagnostics = true
    )
    println("\n✓ Simulation complete: $(length(states)) steps")
    
    # Get physics for visualization
    Fa, Ff, Fg, height = create_mountain_car_physics()
    
    # Export simulation data
    export_simulation_data(states, actions, diagnostics)
    
    # Create all visualizations
    create_comprehensive_plots(states, actions, diagnostics, height)
    
    # Create animation
    create_animation(states, height)
    
    # Export results summary
    export_results_summary(agent, states, actions, diagnostics)
    
    # Final validation - compute total output size
    println("\n" * "="^70)
    println("📦 OUTPUT SUMMARY")
    println("="^70)
    
    output_dirs = [
        ("logs", Config.OUTPUTS.logs_dir),
        ("data", Config.OUTPUTS.data_dir),
        ("plots", Config.OUTPUTS.plots_dir),
        ("animations", Config.OUTPUTS.animations_dir),
        ("diagnostics", Config.OUTPUTS.diagnostics_dir),
        ("results", Config.OUTPUTS.results_dir)
    ]
    
    local total_size = 0.0
    local total_files = 0
    
    for (name, dir_path) in output_dirs
        files = readdir(dir_path, join=false)
        dir_size = sum(filesize(joinpath(dir_path, f)) for f in files if isfile(joinpath(dir_path, f))) / 1024  # KB
        total_size = total_size + dir_size
        total_files = total_files + length(files)
        
        if !isempty(files)
            println("✓ outputs/$name/")
            println("  $(length(files)) files, $(round(dir_size, digits=2)) KB")
            for file in files
                file_size = filesize(joinpath(dir_path, file)) / 1024
                println("    → $file ($(round(file_size, digits=2)) KB)")
            end
        else
            println("○ outputs/$name/ (empty)")
        end
    end
    
    println("\n" * "─"^70)
    println("Total: $total_files files, $(round(total_size, digits=2)) KB")
    println("=" ^70)
    
    println("\n" * "="^70)
    println("✅ EXAMPLE COMPLETED SUCCESSFULLY!")
    println("="^70)
    println("\nAll outputs have been saved to the outputs/ directory.")
    println("Check outputs/results/simulation_report.txt for a detailed summary.\n")
end

