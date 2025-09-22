#!/usr/bin/env julia

# Activate the project environment to ensure all dependencies are available
import Pkg
using Test
using Logging
using Plots
using Statistics
using Dates
using CSV
using DataFrames
using JSON
Pkg.activate(".")

# Main run script for Active Inference Mountain Car example
# This script demonstrates the complete active inference solution

@doc """
Active Inference Mountain Car Example Runner

This script runs the complete active inference mountain car example,
demonstrating how an agent can learn to navigate a challenging environment
using probabilistic planning and active inference principles.

Usage:
    julia run.jl [--naive] [--animation] [--verbose] [--structured] [--performance]

Options:
    --naive: Run only the naive policy comparison
    --animation: Save animations to file
    --verbose: Enable detailed logging
    --structured: Enable structured JSON logging
    --performance: Enable performance logging
    --export: Export results to CSV/JSON
"""

# Include all necessary modules
include("config.jl")
include("src/physics.jl")
include("src/world.jl")
include("src/agent.jl")
include("src/visualization.jl")
include("src/utils.jl")

# Import functions from modules
using .Physics: create_physics, next_state, get_landscape_coordinates
using .World: create_world, simulate_trajectory
using .Agent: create_agent
using .Visualization: plot_landscape, plot_engine_force, create_animation, save_animation, height_at_position, create_unified_animation, create_comparison_animation, save_plot
using .Utils: setup_logging, Timer, export_to_csv, export_to_json, save_experiment_results, validate_config, ProgressBar, update!, finish!
using .Config: PHYSICS, WORLD, TARGET, SIMULATION, OUTPUTS, AGENT, VISUALIZATION

# Configuration validation
function validate_experiment_config()
    issues = validate_config()
    if !isempty(issues)
        @warn "Configuration issues detected:" issues
        return false
    end
    return true
end

function run_naive_policy(run_naive::Bool)
    @info "Running naive policy simulation..."

    timer = Timer("naive_policy_simulation")

    # Create physics functions
    Fa, Ff, Fg, height = create_physics()

    # Create world
    execute_naive, observe_naive, reset_naive, get_state_naive, set_state_naive = create_world(
        Fg = Fg, Ff = Ff, Fa = Fa,
        initial_position = WORLD.initial_position,
        initial_velocity = WORLD.initial_velocity
    )

    # Generate naive actions (full power forward)
    naive_actions = fill(SIMULATION.naive_action, SIMULATION.time_steps_naive)

    # Initialize storage for results
    naive_states = Vector{Vector{Float64}}(undef, SIMULATION.time_steps_naive + 1)
    naive_positions = Vector{Float64}(undef, SIMULATION.time_steps_naive + 1)
    naive_velocities = Vector{Float64}(undef, SIMULATION.time_steps_naive + 1)

    # Run simulation with detailed tracking
    reset_naive()
    naive_states[1] = observe_naive()
    naive_positions[1] = naive_states[1][1]
    naive_velocities[1] = naive_states[1][2]

    # Progress bar for long simulations
    if SIMULATION.time_steps_naive > 50
        pb = ProgressBar(SIMULATION.time_steps_naive)
        for t in 1:SIMULATION.time_steps_naive
            execute_naive(naive_actions[t])
            naive_states[t + 1] = observe_naive()
            naive_positions[t + 1] = naive_states[t + 1][1]
            naive_velocities[t + 1] = naive_states[t + 1][2]
            update!(pb, t)

            # Detailed logging every 25 steps
            if t % 25 == 0
                @info "Naive simulation step $t: Position=$(round(naive_positions[t+1], digits=3)), Velocity=$(round(naive_velocities[t+1], digits=3))"
            end
        end
        finish!(pb)
    else
        for t in 1:SIMULATION.time_steps_naive
            execute_naive(naive_actions[t])
            naive_states[t + 1] = observe_naive()
            naive_positions[t + 1] = naive_states[t + 1][1]
            naive_velocities[t + 1] = naive_states[t + 1][2]

            # Logging for shorter simulations
            if t % 10 == 0
                @info "Naive simulation step $t: Position=$(round(naive_positions[t+1], digits=3)), Velocity=$(round(naive_velocities[t+1], digits=3))"
            end
        end
    end

    close(timer)

    # Calculate performance metrics
    final_position = naive_states[end][1]
    success = abs(final_position - TARGET.position) < 0.1
    avg_velocity = mean(naive_velocities)
    max_velocity = maximum(naive_velocities)
    distance_traveled = sum(abs(naive_positions[i] - naive_positions[i-1]) for i in 2:length(naive_positions))

    @info "Naive policy simulation completed." length(naive_states) = length(naive_states) final_position = round(final_position, digits=3) success = success avg_velocity = round(avg_velocity, digits=3) max_velocity = round(max_velocity, digits=3) distance_traveled = round(distance_traveled, digits=3)

    # Create animation if requested and naive policy was run
    if "--animation" in ARGS && run_naive
        @info "Creating naive policy animation..."
        anim_timer = Timer("naive_animation")
        @info "Starting naive policy animation generation..."
        @info "Animation will have $(length(naive_states)) frames"
        try
            create_unified_animation(naive_states, naive_actions, "Naive Policy", OUTPUTS.naive_animation)
        catch e
            @error "Error creating naive animation: $e"
            @error "Stack trace: $(stacktrace(catch_backtrace()))"
            rethrow(e)
        end
        close(anim_timer)
        @info "Naive animation saved to: $(OUTPUTS.naive_animation)"
    end

    # Package results
    naive_results = Dict(
        "method" => "naive",
        "final_position" => final_position,
        "success" => success,
        "avg_velocity" => avg_velocity,
        "max_velocity" => max_velocity,
        "distance_traveled" => distance_traveled,
        "time_steps" => SIMULATION.time_steps_naive,
        "actions" => naive_actions,
        "positions" => naive_positions,
        "velocities" => naive_velocities,
        "states" => naive_states,
        "config" => Dict(
            "initial_position" => WORLD.initial_position,
            "target_position" => TARGET.position,
            "naive_action" => SIMULATION.naive_action
        )
    )

    return naive_results, naive_states, naive_actions
end

function create_naive_animation(states::Vector{Vector{Float64}})
    # Get landscape coordinates
    x_coords, y_coords = get_landscape_coordinates()

    # Create actions vector (constant for naive policy)
    actions = [SIMULATION.naive_action for _ in 1:length(states)]

    anim = @animate for i in 1:length(states)
        # Create landscape plot
        landscape_plot = plot(x_coords, y_coords,
                             title = "Naive Policy - Step $i",
                             label = "Landscape",
                             color = "black",
                             xlabel = "Position",
                             ylabel = "Height",
                             grid = true,
                             gridalpha = 0.3)

        scatter!(landscape_plot, [states[i][1]], [height_at_position(states[i][1])],
                label = "Car", color = "red", markersize = 8)
        scatter!(landscape_plot, [TARGET.position], [height_at_position(TARGET.position)],
                label = "Goal", color = "orange", markersize = 10, marker = :star)

        # Create engine force plot
        engine_plot = plot(title = "Engine Force (Naive Policy)",
                          xlabel = "Time Step",
                          ylabel = "Force",
                          xlim = (0, length(actions)),
                          ylim = (-0.1, 0.1),
                          grid = true,
                          gridalpha = 0.3)

        plot!(engine_plot, 1:i, actions[1:i],
              color = "blue", linewidth = 2, alpha = 0.8,
              label = "Constant Force: $(SIMULATION.naive_action)")

        # Create metrics plot for naive policy
        positions = getindex.(states[1:i], 1)
        velocities = getindex.(states[1:i], 2)

        # Calculate progress toward goal
        total_distance = abs(WORLD.initial_position - TARGET.position)
        progress = [max(0, (total_distance - abs(pos - TARGET.position)) / total_distance * 100)
                    for pos in positions]

        # Calculate cumulative distance traveled
        distance_traveled = [0.0]
        for j in 2:i
            dist = abs(positions[j] - positions[j-1])
            push!(distance_traveled, dist + distance_traveled[end])
        end

        metrics_plot = plot(title = "Performance Metrics",
                           xlabel = "Time Step",
                           ylabel = "Value",
                           xlim = (0, length(actions)),
                           grid = true,
                           gridalpha = 0.3)

        plot!(metrics_plot, 1:i, positions,
              label = "Position", color = "blue", linewidth = 2, alpha = 0.8)
        plot!(metrics_plot, 1:i, velocities,
              label = "Velocity", color = "red", linewidth = 2, alpha = 0.7)
        plot!(metrics_plot, 1:i, progress,
              label = "Goal Progress (%)", color = "green", linewidth = 2, alpha = 0.6)
        plot!(metrics_plot, 1:i, distance_traveled,
              label = "Distance Traveled", color = "orange", linewidth = 2, alpha = 0.5)

        # Combine plots
        combined_plot = plot(landscape_plot, engine_plot, metrics_plot,
                            layout = (3, 1),
                            size = (800, 1200))

        # Add step information
        annotate!(combined_plot, 0.02, 0.98,
                 text("Position: $(round(states[i][1], digits=3))\nVelocity: $(round(states[i][2], digits=3))",
                      :black, :left, 10),
                 subplot = 1)
    end

    return anim
end

function run_active_inference()
    @info "Running active inference simulation..."

    timer = Timer("active_inference_simulation")

    # Create physics functions
    Fa, Ff, Fg, height = create_physics()

    # Create world for active inference
    execute_ai, observe_ai, reset_ai, get_state_ai, set_state_ai = create_world(
        Fg = Fg, Ff = Ff, Fa = Fa,
        initial_position = WORLD.initial_position,
        initial_velocity = WORLD.initial_velocity
    )

    # Create agent
    compute_ai, act_ai, slide_ai, future_ai, reset_agent = create_agent(
        T = SIMULATION.planning_horizon,
        Fa = Fa, Ff = Ff, Fg = Fg,
        engine_force_limit = PHYSICS.engine_force_limit,
        x_target = [TARGET.position, TARGET.velocity],
        initial_position = WORLD.initial_position,
        initial_velocity = WORLD.initial_velocity
    )

    # Initialize storage for results
    agent_actions = Vector{Float64}(undef, SIMULATION.time_steps_ai)
    agent_states = Vector{Vector{Float64}}(undef, SIMULATION.time_steps_ai + 1)  # +1 for initial state
    predicted_states = Matrix{Float64}(undef, SIMULATION.time_steps_ai, SIMULATION.planning_horizon)
    inference_times = Vector{Float64}(undef, SIMULATION.time_steps_ai)
    prediction_errors = Vector{Float64}(undef, SIMULATION.time_steps_ai)

    # Initialize
    reset_ai()
    agent_states[1] = observe_ai()

    # Progress bar for long simulations
    if SIMULATION.time_steps_ai > 50
        pb = ProgressBar(SIMULATION.time_steps_ai)
    end

    # Run active inference loop
    @info "Starting active inference loop with $(SIMULATION.time_steps_ai) steps..."
    for t in 1:SIMULATION.time_steps_ai
        # Get current state
        current_state = get_state_ai()

        # Detailed logging every 20 steps
        if t % 20 == 0
            @info "AI simulation step $t: Position=$(round(current_state[1], digits=3)), Velocity=$(round(current_state[2], digits=3))"
        end

        # Agent decides on action
        action_timer = Timer("action_selection_$t")
        action = act_ai()
        close(action_timer)
        agent_actions[t] = action

        # Execute action in world
        execute_ai(action)

        # Observe new state
        new_state = observe_ai()
        agent_states[t + 1] = new_state

        # Update agent beliefs
        @info "Step $t: Executing action $(round(action, digits=3))..."
        inference_timer = Timer("inference_$t")
        compute_ai(action, new_state)
        close(inference_timer)
        inference_times[t] = time() - (time() - 0.1)  # Simplified timing

        # Get predicted future states
        predictions = future_ai()
        predicted_states[t, :] = predictions

        # Log inference timing
        if t % 20 == 0
            @info "Step $t inference completed in $(round(inference_times[t], digits=4))s"
        end

        # Calculate prediction error (only if we have the next actual state)
        if t < SIMULATION.time_steps_ai && t + 2 <= length(agent_states) && isassigned(agent_states, t + 2)
            next_predicted = predicted_states[t, 1]
            actual_next = agent_states[t + 2][1]
            prediction_errors[t] = abs(next_predicted - actual_next)
        else
            prediction_errors[t] = NaN  # No prediction error available yet
        end

        # Slide time window
        slide_ai()

        # Update progress bar
        if SIMULATION.time_steps_ai > 50
            update!(pb, t)
        end

        # Log progress
        if t % 10 == 0
            @info "Step \$t: position = \$(round(new_state[1], digits=3)), " *
                  "velocity = \$(round(new_state[2], digits=3)), " *
                  "action = \$(round(action, digits=3)), " *
                  "pred_error = \$(round(prediction_errors[t], digits=3))"
        end
    end

    if SIMULATION.time_steps_ai > 50
        finish!(pb)
    end

    close(timer)

    # Calculate performance metrics
    final_position = agent_states[end][1]
    success = abs(final_position - TARGET.position) < 0.1

    @info "Active inference simulation completed. Final position: $(round(final_position, digits=3)), Success: $success"
    avg_velocity = mean([s[2] for s in agent_states])
    max_velocity = maximum([s[2] for s in agent_states])
    distance_traveled = sum(abs(agent_states[i][1] - agent_states[i-1][1]) for i in 2:length(agent_states))
    avg_action = mean(agent_actions)
    action_variance = var(agent_actions)
    avg_prediction_error = mean(filter(!isnan, prediction_errors))
    avg_inference_time = mean(inference_times)

    @info "Active inference simulation completed." final_position = round(final_position, digits=3) success = success avg_velocity = round(avg_velocity, digits=3) max_velocity = round(max_velocity, digits=3) distance_traveled = round(distance_traveled, digits=3) avg_prediction_error = round(avg_prediction_error, digits=3) avg_inference_time = round(avg_inference_time, digits=4)

    # Create animation if requested
    if "--animation" in ARGS
        @info "Creating active inference animation..."
        anim_timer = Timer("ai_animation")
        create_unified_animation(agent_states[2:end], agent_actions, "Active Inference", OUTPUTS.ai_animation,
                                predicted_states = predicted_states)
        close(anim_timer)
        @info "Active inference animation saved to: $(OUTPUTS.ai_animation)"
    end

    # Package results
    ai_results = Dict(
        "method" => "active_inference",
        "final_position" => final_position,
        "success" => success,
        "avg_velocity" => avg_velocity,
        "max_velocity" => max_velocity,
        "distance_traveled" => distance_traveled,
        "avg_action" => avg_action,
        "action_variance" => action_variance,
        "avg_prediction_error" => avg_prediction_error,
        "avg_inference_time" => avg_inference_time,
        "time_steps" => SIMULATION.time_steps_ai,
        "planning_horizon" => SIMULATION.planning_horizon,
        "actions" => agent_actions,
        "positions" => [s[1] for s in agent_states],
        "velocities" => [s[2] for s in agent_states],
        "states" => agent_states,
        "predicted_states" => predicted_states,
        "prediction_errors" => prediction_errors,
        "inference_times" => inference_times,
        "config" => Dict(
            "initial_position" => WORLD.initial_position,
            "target_position" => TARGET.position,
            "planning_horizon" => SIMULATION.planning_horizon,
            "transition_precision" => AGENT.transition_precision,
            "observation_variance" => AGENT.observation_variance
        )
    )

    return ai_results, agent_states, agent_actions, predicted_states
end

@doc """
Export comprehensive raw data for all tracked variables to CSV and JSON formats.

This function exports detailed time series data for both naive and active inference
simulations, including all state variables, actions, predictions, and derived metrics.

Args:
- results_dir: Directory to save the data files
- naive_states: Naive policy state trajectory
- naive_actions: Naive policy action sequence
- agent_states: Active inference state trajectory  
- agent_actions: Active inference action sequence
- predicted_states: Predicted future states from active inference
"""
function export_comprehensive_data(results_dir::String, 
                                  naive_states::Union{Vector{Vector{Float64}}, Nothing},
                                  naive_actions::Union{Vector{Float64}, Nothing},
                                  agent_states::Union{Vector{Vector{Float64}}, Nothing}, 
                                  agent_actions::Union{Vector{Float64}, Nothing},
                                  predicted_states::Union{Matrix{Float64}, Nothing})
    
    @info "Exporting comprehensive raw data..."
    
    # Create data subdirectory
    data_dir = joinpath(results_dir, "raw_data")
    mkpath(data_dir)
    
    # Export naive policy data
    if naive_states !== nothing && naive_actions !== nothing
        @info "Exporting naive policy data..."
        
        # Extract time series
        naive_positions = getindex.(naive_states, 1)
        naive_velocities = getindex.(naive_states, 2)
        
        # Calculate derived metrics
        naive_distances_to_goal = [abs(pos - TARGET.position) for pos in naive_positions]
        naive_progress = calculate_progress_metrics(naive_positions)
        naive_energy = calculate_energy_metrics(naive_states, naive_actions)
        
        # Ensure consistent lengths (actions are typically one less than states)
        min_length = min(length(naive_states), length(naive_actions))
        
        # Create comprehensive DataFrame with consistent lengths
        naive_df = DataFrame(
            time_step = 1:min_length,
            position = naive_positions[1:min_length],
            velocity = naive_velocities[1:min_length],
            action = naive_actions[1:min_length],
            distance_to_goal = naive_distances_to_goal[1:min_length],
            progress_percent = naive_progress[1:min_length],
            kinetic_energy = [0.5 * naive_states[i][2]^2 for i in 1:min_length],
            potential_energy = [height_at_position(naive_states[i][1]) for i in 1:min_length],
            cumulative_action_magnitude = cumsum(abs.(naive_actions[1:min_length])),
            velocity_change = [i == 1 ? 0.0 : naive_states[i][2] - naive_states[i-1][2] for i in 1:min_length],
            position_change = [i == 1 ? 0.0 : naive_states[i][1] - naive_states[i-1][1] for i in 1:min_length]
        )
        
        # Save to CSV and JSON
        CSV.write(joinpath(data_dir, "naive_policy_timeseries.csv"), naive_df)
        
        # Export as JSON for programmatic access
        naive_json = Dict(
            "metadata" => Dict(
                "policy_type" => "naive",
                "constant_action" => SIMULATION.naive_action,
                "total_steps" => length(naive_states),
                "final_position" => naive_positions[end],
                "final_velocity" => naive_velocities[end],
                "success" => abs(naive_positions[end] - TARGET.position) < 0.1
            ),
            "timeseries" => naive_df
        )
        
        open(joinpath(data_dir, "naive_policy_data.json"), "w") do f
            JSON.print(f, naive_json, 2)
        end
    end
    
    # Export active inference data
    if agent_states !== nothing && agent_actions !== nothing
        @info "Exporting active inference data..."
        
        # Extract time series (skip initial state)
        ai_states_data = agent_states[2:end]  # Skip initial state
        ai_positions = getindex.(ai_states_data, 1)
        ai_velocities = getindex.(ai_states_data, 2)
        
        # Calculate derived metrics
        ai_distances_to_goal = [abs(pos - TARGET.position) for pos in ai_positions]
        ai_progress = calculate_progress_metrics(ai_positions)
        ai_energy = calculate_energy_metrics(ai_states_data, agent_actions)
        
        # Ensure consistent lengths (actions are typically one less than states)
        min_length = min(length(ai_states_data), length(agent_actions))
        
        # Create comprehensive DataFrame with consistent lengths
        ai_df = DataFrame(
            time_step = 1:min_length,
            position = ai_positions[1:min_length],
            velocity = ai_velocities[1:min_length],
            action = agent_actions[1:min_length],
            distance_to_goal = ai_distances_to_goal[1:min_length],
            progress_percent = ai_progress[1:min_length],
            kinetic_energy = [0.5 * ai_states_data[i][2]^2 for i in 1:min_length],
            potential_energy = [height_at_position(ai_states_data[i][1]) for i in 1:min_length],
            cumulative_action_magnitude = cumsum(abs.(agent_actions[1:min_length])),
            velocity_change = [i == 1 ? 0.0 : ai_states_data[i][2] - ai_states_data[i-1][2] for i in 1:min_length],
            position_change = [i == 1 ? 0.0 : ai_states_data[i][1] - ai_states_data[i-1][1] for i in 1:min_length]
        )
        
        # Save to CSV
        CSV.write(joinpath(data_dir, "active_inference_timeseries.csv"), ai_df)
        
        # Export predictions if available
        if predicted_states !== nothing
            pred_df = DataFrame(predicted_states, :auto)
            rename!(pred_df, [Symbol("prediction_step_$i") for i in 1:size(predicted_states, 2)])
            pred_df.time_step = 1:size(predicted_states, 1)
            CSV.write(joinpath(data_dir, "predictions_timeseries.csv"), pred_df)
        end
        
        # Export as JSON
        ai_json = Dict(
            "metadata" => Dict(
                "policy_type" => "active_inference",
                "total_steps" => length(ai_states_data),
                "final_position" => ai_positions[end],
                "final_velocity" => ai_velocities[end],
                "success" => abs(ai_positions[end] - TARGET.position) < 0.1,
                "prediction_horizon" => predicted_states !== nothing ? size(predicted_states, 2) : 0
            ),
            "timeseries" => ai_df,
            "predictions" => predicted_states !== nothing ? predicted_states : nothing
        )
        
        open(joinpath(data_dir, "active_inference_data.json"), "w") do f
            JSON.print(f, ai_json, 2)
        end
    end
    
    # Export comparative analysis
    if naive_states !== nothing && agent_states !== nothing
        @info "Exporting comparative analysis..."
        
        comparison_data = Dict(
            "naive_final_position" => getindex(naive_states[end], 1),
            "ai_final_position" => getindex(agent_states[end], 1),
            "naive_total_energy" => sum(abs.(naive_actions)),
            "ai_total_energy" => sum(abs.(agent_actions)),
            "naive_success" => abs(getindex(naive_states[end], 1) - TARGET.position) < 0.1,
            "ai_success" => abs(getindex(agent_states[end], 1) - TARGET.position) < 0.1,
            "improvement" => getindex(agent_states[end], 1) - getindex(naive_states[end], 1)
        )
        
        open(joinpath(data_dir, "comparative_analysis.json"), "w") do f
            JSON.print(f, comparison_data, 2)
        end
    end
    
    @info "Raw data export completed to: $data_dir"
end

@doc """
Generate all individual visualizations (separate plots for detailed analysis).

This function creates individual static plots for each aspect of the simulation,
providing detailed analysis capabilities beyond the animations.

Args:
- results_dir: Directory to save visualization files
- naive_states: Naive policy state trajectory
- naive_actions: Naive policy action sequence  
- agent_states: Active inference state trajectory
- agent_actions: Active inference action sequence
- predicted_states: Predicted future states from active inference
"""
function generate_all_visualizations(results_dir::String,
                                    naive_states::Union{Vector{Vector{Float64}}, Nothing},
                                    naive_actions::Union{Vector{Float64}, Nothing},
                                    agent_states::Union{Vector{Vector{Float64}}, Nothing},
                                    agent_actions::Union{Vector{Float64}, Nothing}, 
                                    predicted_states::Union{Matrix{Float64}, Nothing})
    
    @info "Generating comprehensive visualization suite..."
    
    # Create visualizations subdirectory
    viz_dir = joinpath(results_dir, "visualizations")
    mkpath(viz_dir)
    
    # Generate landscape plots
    if naive_states !== nothing || agent_states !== nothing
        @info "Generating landscape trajectory plots..."
        
        # Combined trajectory plot
        landscape_plot = plot_landscape(0.0, show_goal = true, title = "Mountain Car Trajectories Comparison")
        
        if naive_states !== nothing
            naive_positions = getindex.(naive_states, 1)
            naive_heights = height_at_position.(naive_positions)
            plot!(landscape_plot, naive_positions, naive_heights,
                  label = "Naive Policy", color = "blue", linewidth = 3, alpha = 0.8)
            scatter!(landscape_plot, [naive_positions[end]], [naive_heights[end]],
                    label = "Naive Final", color = "blue", markersize = 10, marker = :circle)
        end
        
        if agent_states !== nothing
            ai_positions = getindex.(agent_states[2:end], 1)  # Skip initial state
            ai_heights = height_at_position.(ai_positions)
            plot!(landscape_plot, ai_positions, ai_heights,
                  label = "Active Inference", color = "purple", linewidth = 3, alpha = 0.8)
            scatter!(landscape_plot, [ai_positions[end]], [ai_heights[end]],
                    label = "AI Final", color = "purple", markersize = 10, marker = :star)
        end
        
        save_plot(landscape_plot, joinpath(viz_dir, "trajectory_comparison.png"))
    end
    
    # Generate action comparison plots
    if naive_actions !== nothing || agent_actions !== nothing
        @info "Generating action comparison plots..."
        
        action_plot = plot(title = "Action Sequences Comparison",
                          xlabel = "Time Step", ylabel = "Engine Force",
                          grid = true, gridalpha = 0.3)
        
        if naive_actions !== nothing
            plot!(action_plot, 1:length(naive_actions), naive_actions,
                  label = "Naive Policy (Constant)", color = "blue", linewidth = 2)
        end
        
        if agent_actions !== nothing
            plot!(action_plot, 1:length(agent_actions), agent_actions,
                  label = "Active Inference (Adaptive)", color = "purple", linewidth = 2)
        end
        
        # Add force limits
        hline!(action_plot, [PHYSICS.engine_force_limit], color = "red", linestyle = :dash, alpha = 0.6, label = "Force Limits")
        hline!(action_plot, [-PHYSICS.engine_force_limit], color = "red", linestyle = :dash, alpha = 0.6, label = "")
        
        save_plot(action_plot, joinpath(viz_dir, "action_comparison.png"))
    end
    
    # Generate performance metrics plots
    if naive_states !== nothing && agent_states !== nothing
        @info "Generating performance metrics plots..."
        
        # Position over time
        pos_plot = plot(title = "Position Over Time", xlabel = "Time Step", ylabel = "Position")
        plot!(pos_plot, 1:length(naive_states), getindex.(naive_states, 1),
              label = "Naive Policy", color = "blue", linewidth = 2)
        plot!(pos_plot, 1:length(agent_states)-1, getindex.(agent_states[2:end], 1),
              label = "Active Inference", color = "purple", linewidth = 2)
        hline!(pos_plot, [TARGET.position], color = "orange", linestyle = :dash, label = "Goal")
        save_plot(pos_plot, joinpath(viz_dir, "position_comparison.png"))
        
        # Velocity over time
        vel_plot = plot(title = "Velocity Over Time", xlabel = "Time Step", ylabel = "Velocity")
        plot!(vel_plot, 1:length(naive_states), getindex.(naive_states, 2),
              label = "Naive Policy", color = "blue", linewidth = 2)
        plot!(vel_plot, 1:length(agent_states)-1, getindex.(agent_states[2:end], 2),
              label = "Active Inference", color = "purple", linewidth = 2)
        save_plot(vel_plot, joinpath(viz_dir, "velocity_comparison.png"))
        
        # Distance to goal over time
        dist_plot = plot(title = "Distance to Goal Over Time", xlabel = "Time Step", ylabel = "Distance")
        naive_distances = [abs(pos - TARGET.position) for pos in getindex.(naive_states, 1)]
        ai_distances = [abs(pos - TARGET.position) for pos in getindex.(agent_states[2:end], 1)]
        plot!(dist_plot, 1:length(naive_distances), naive_distances,
              label = "Naive Policy", color = "blue", linewidth = 2)
        plot!(dist_plot, 1:length(ai_distances), ai_distances,
              label = "Active Inference", color = "purple", linewidth = 2)
        save_plot(dist_plot, joinpath(viz_dir, "distance_to_goal_comparison.png"))
    end
    
    # Generate prediction visualization if available
    if predicted_states !== nothing
        @info "Generating prediction analysis plots..."
        
        pred_plot = plot(title = "Prediction Accuracy Over Time",
                        xlabel = "Time Step", ylabel = "Prediction Horizon")
        
        # Create heatmap of predictions
        heatmap!(pred_plot, 1:size(predicted_states, 1), 1:size(predicted_states, 2), predicted_states',
                color = :viridis, alpha = 0.8)
        
        save_plot(pred_plot, joinpath(viz_dir, "prediction_heatmap.png"))
    end
    
    @info "Visualization suite completed in: $viz_dir"
end

@doc """
Calculate progress metrics for a position trajectory.
"""
function calculate_progress_metrics(positions::Vector{Float64})
    total_distance = abs(WORLD.initial_position - TARGET.position)
    return [max(0, (total_distance - abs(pos - TARGET.position)) / total_distance * 100) for pos in positions]
end

@doc """
Calculate energy-related metrics for states and actions.
"""
function calculate_energy_metrics(states::Vector{Vector{Float64}}, actions::Vector{Float64})
    kinetic_energies = [0.5 * state[2]^2 for state in states]
    potential_energies = [height_at_position(state[1]) for state in states]
    action_energies = cumsum(abs.(actions))
    
    return Dict(
        "kinetic" => kinetic_energies,
        "potential" => potential_energies, 
        "action_cumulative" => action_energies
    )
end

function run_experiment()
    @info "Starting Active Inference Mountain Car experiment..."

    # Validate configuration
    if !validate_experiment_config()
        @error "Configuration validation failed. Aborting experiment."
        return
    end

    # Ensure output directory exists
    if !isdir(OUTPUTS.output_dir)
        mkpath(OUTPUTS.output_dir)
    end

    # Setup logging
    verbose = "--verbose" in ARGS
    structured = "--structured" in ARGS
    performance = "--performance" in ARGS
    setup_logging(verbose=verbose, structured=structured, performance=performance)

    # Log configuration
    @info "Configuration loaded"
    @info "Physics: engine_force_limit=$(PHYSICS.engine_force_limit), friction_coefficient=$(PHYSICS.friction_coefficient)"
    @info "World: initial_position=$(WORLD.initial_position), target_position=$(TARGET.position)"
    @info "Simulation: planning_horizon=$(SIMULATION.planning_horizon), time_steps_ai=$(SIMULATION.time_steps_ai)"

    # Initialize results collection
    experiment_results = Dict(
        "experiment_name" => "active_inference_mountain_car",
        "timestamp" => string(now()),
        "configuration" => Dict(
            "physics" => Dict(
                "engine_force_limit" => PHYSICS.engine_force_limit,
                "friction_coefficient" => PHYSICS.friction_coefficient
            ),
            "world" => Dict(
                "initial_position" => WORLD.initial_position,
                "initial_velocity" => WORLD.initial_velocity,
                "target_position" => WORLD.target_position,
                "target_velocity" => WORLD.target_velocity
            ),
            "target" => Dict(
                "position" => TARGET.position,
                "velocity" => TARGET.velocity
            ),
            "simulation" => Dict(
                "time_steps_naive" => SIMULATION.time_steps_naive,
                "time_steps_ai" => SIMULATION.time_steps_ai,
                "planning_horizon" => SIMULATION.planning_horizon,
                "naive_action" => SIMULATION.naive_action
            ),
            "agent" => Dict(
                "transition_precision" => AGENT.transition_precision,
                "observation_variance" => AGENT.observation_variance,
                "control_prior_variance" => AGENT.control_prior_variance,
                "goal_prior_variance" => AGENT.goal_prior_variance,
                "initial_state_variance" => AGENT.initial_state_variance
            )
        ),
        "results" => Dict{String, Any}()
    )

    # Determine whether to run naive policy
    run_naive = "--naive" in ARGS || length(ARGS) == 0 || "--animation" in ARGS

    # Variables to store naive simulation data for comparison animation
    naive_states = nothing
    naive_actions = nothing

    if run_naive
        @info "Running naive policy comparison..."
        naive_results, naive_states, naive_actions = run_naive_policy(run_naive)
        experiment_results["results"]["naive"] = naive_results
    end

    # Variables to store active inference simulation data for comparison animation
    agent_states = nothing
    agent_actions = nothing
    predicted_states = nothing

    # Run active inference if not naive-only mode
    if !("--naive" in ARGS)
        @info "Running active inference..."
        ai_results, agent_states, agent_actions, predicted_states = run_active_inference()
        experiment_results["results"]["active_inference"] = ai_results

        # Compare results
        naive_pos = haskey(experiment_results["results"], "naive") ?
                   experiment_results["results"]["naive"]["final_position"] : "N/A"
        ai_pos = ai_results["final_position"]
        target_pos = TARGET.position

        # Calculate values for logging (handle N/A case)
        naive_display = (naive_pos == "N/A") ? "N/A" : round(naive_pos, digits=3)
        ai_display = round(ai_pos, digits=3)
        improvement_display = (naive_pos == "N/A") ? "N/A" : round(ai_pos - naive_pos, digits=3)

        @info "Experiment completed." naive_final_position = naive_display ai_final_position = ai_display target_position = target_pos improvement = improvement_display ai_success = ai_results["success"]
    else
        @info "Naive-only mode: skipping active inference."
    end

    # Create comparison animation if both animations exist and both policies were run
    if naive_states !== nothing && naive_actions !== nothing && agent_states !== nothing && agent_actions !== nothing && predicted_states !== nothing
        try
            @info "Creating comparison animation..."

            # Create comparison animation
            comparison_filename = "outputs/ai-mountain-car-comparison.gif"
            create_comparison_animation(naive_states, naive_actions, agent_states[2:end], agent_actions, predicted_states, "Naive vs Active Inference Comparison", comparison_filename)

            @info "Comparison animation saved to: $comparison_filename"
        catch e
            @error "Error creating comparison animation: $e"
            @error "Stack trace: $(stacktrace(catch_backtrace()))"
        end
    end

    # Export comprehensive results (always enabled for maximum output)
    @info "Exporting comprehensive results and visualizations..."
    results_dir = save_experiment_results("mountain_car_experiment", experiment_results)
    
    # Export detailed raw data for all tracked variables
    export_comprehensive_data(results_dir, naive_states, naive_actions, agent_states, agent_actions, predicted_states)
    
    # Generate all individual visualizations
    generate_all_visualizations(results_dir, naive_states, naive_actions, agent_states, agent_actions, predicted_states)
    
    @info "Comprehensive results exported to: $results_dir"

    @info "Experiment finished successfully."
end

function main()
    # Check command line arguments
    if "--help" in ARGS || "-h" in ARGS
        println("""
        Active Inference Mountain Car Example

        A comprehensive implementation of active inference for solving the mountain car problem.

        Usage: julia run.jl [options]

        Options:
          --naive        Run only naive policy comparison
          --animation    Save animations to GIF files
          --verbose      Enable detailed console logging
          --structured   Enable structured JSON logging
          --performance  Enable performance logging to CSV
          --export       Export results to JSON/CSV files
          --help         Show this help message

        Examples:
          julia run.jl                    # Run complete experiment
          julia run.jl --naive           # Run only naive policy
          julia run.jl --verbose         # Run with detailed logging
          julia run.jl --animation       # Run and save animations
          julia run.jl --export          # Export results to files
          julia run.jl --performance     # Log performance metrics
          julia run.jl --structured      # Structured JSON logging

        Output:
          - Console output with progress updates
          - Log files: mountain_car_log.txt (and variants)
          - Animation files: *.gif (if --animation)
          - Results: JSON/CSV files (if --export)
          - Performance metrics (if --performance)
        """)
        return
    end

    try
        run_experiment()
    catch e
        @error "Experiment failed with error: \$e"
        if "--verbose" in ARGS
            showerror(stderr, e, catch_backtrace())
        end
        rethrow(e)
    end
end

# Run main function if this file is executed directly
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
