# Visualization module for the Active Inference Mountain Car example
# Handles plotting and animation creation

@doc """
Visualization module for mountain car environment.

This module provides functions for creating plots and animations of the
mountain car simulation, including landscape visualization, trajectory
plotting, and engine force visualization.
"""
module Visualization

using Plots
using Printf
using Statistics
using HypergeometricFunctions: _₂F₁
import ..Config: VISUALIZATION, OUTPUTS, SIMULATION, PHYSICS, TARGET, AGENT, WORLD
import ..Physics: get_landscape_coordinates, create_physics

@doc """
Create the mountain landscape plot with car position and goal.

Args:
- car_position: Current car position
- car_states: Historical car states (optional)
- predicted_positions: Predicted future positions (optional)
- show_goal: Whether to show goal position
- show_height_contours: Whether to show height contours
- show_velocity_field: Whether to show velocity field arrows
- title: Plot title
- theme: Color theme (:default, :dark, :colorblind_friendly)

Returns:
- Plot object
"""
function plot_landscape(car_position::Float64;
                       car_states::Union{Vector{Vector{Float64}}, Nothing} = nothing,
                       predicted_positions::Union{Vector{Float64}, Nothing} = nothing,
                       show_goal::Bool = true,
                       show_height_contours::Bool = false,
                       show_velocity_field::Bool = false,
                       title::String = "Mountain Car Environment",
                       theme::Symbol = :default)

    # Get landscape coordinates
    x_coords, y_coords = get_landscape_coordinates()

    # Set color scheme based on theme
    colors = get_color_scheme(theme)

    # Create base plot with enhanced styling
    plt = plot(x_coords, y_coords,
               title = title,
               xlabel = "Position",
               ylabel = "Height",
               label = "Landscape",
               color = colors.landscape,
               linewidth = 3,
               size = VISUALIZATION.plot_size,
               grid = true,
               gridalpha = 0.3,
               gridstyle = :dash,
               fontfamily = "Arial",
               titlefontsize = 14,
               guidefontsize = 12,
               tickfontsize = 10,
               legendfontsize = 10)

    # Add height contours if requested
    if show_height_contours
        contour!(plt, x_coords, y_coords,
                levels = 5,
                color = colors.contour,
                alpha = 0.4,
                label = "Height Contours")
    end

    # Add velocity field if requested
    if show_velocity_field
        add_velocity_field!(plt, x_coords, colors.velocity_field)
    end

    # Add current car position with enhanced styling
    car_height = height_at_position(car_position)
    scatter!(plt, [car_position], [car_height],
             label = "Current Position",
             color = colors.car,
             markersize = 12,
             marker = :circle,
             markerstrokewidth = 2,
             markerstrokecolor = :white)

    # Add historical trajectory if provided with gradient effect
    if car_states !== nothing && !isempty(car_states)
        positions = getindex.(car_states, 1)
        heights = height_at_position.(positions)

        # Plot trajectory with simple color for now
        plot!(plt, positions, heights,
              label = "Trajectory",
              color = colors.trajectory,
              alpha = 0.8,
              linewidth = 3)
    end

    # Add predicted future positions if provided with uncertainty visualization
    if predicted_positions !== nothing && !isempty(predicted_positions)
        pred_heights = height_at_position.(predicted_positions)

        # Add prediction points
        scatter!(plt, predicted_positions, pred_heights,
                label = "Predictions",
                color = colors.predictions,
                alpha = 0.7,
                markersize = 8,
                marker = :diamond)

        # Add prediction uncertainty (simplified)
        for (i, pos) in enumerate(predicted_positions)
            h = pred_heights[i]
            # Add small error bars to show uncertainty
            plot!(plt, [pos, pos], [h-0.01, h+0.01],
                  color = colors.uncertainty,
                  alpha = 0.3,
                  linewidth = 2,
                  label = i == 1 ? "Uncertainty" : "")
        end
    end

    # Add goal if requested with enhanced styling
    if show_goal
        goal_x = TARGET.position
        goal_y = height_at_position(goal_x)
        scatter!(plt, [goal_x], [goal_y],
                label = "Goal",
                color = colors.goal,
                markersize = 16,
                marker = :star,
                markerstrokewidth = 3,
                markerstrokecolor = :white)

        # Add goal region highlight
        plot!(plt, [goal_x-0.05, goal_x+0.05], [goal_y, goal_y],
              fillrange = [0, goal_y],
              fillalpha = 0.1,
              color = colors.goal_region,
              label = "")
    end

    # Add current position annotation
    annotate!(plt, car_position, car_height + 0.1,
              text("Pos: $(round(car_position, digits=2))", colors.car, :center, 10))

    return plt
end

@doc """
Create engine force plot showing action history.

Args:
- actions: Vector of actions taken over time
- current_step: Current time step

Returns:
- Plot object
"""
function plot_engine_force(actions::Vector{Float64}, current_step::Int)
    plt = plot(actions[1:current_step],
               title = "Engine Force (Agent Actions)",
               xlabel = "Time Step",
               ylabel = "Force",
               label = "Action",
               color = "purple",
               linewidth = 2,
               xlim = (0, length(actions)),
               ylim = VISUALIZATION.engine_force_limits,
               size = VISUALIZATION.plot_size)

    # Add horizontal lines for force limits
    hline!(plt, [PHYSICS.engine_force_limit], color = "red", linestyle = :dash,
           label = "Force Limit", alpha = 0.5)
    hline!(plt, [-PHYSICS.engine_force_limit], color = "red", linestyle = :dash,
           label = "", alpha = 0.5)

    return plt
end

@doc """
Create combined plot with landscape and engine force.

Args:
- car_position: Current car position
- actions: Vector of actions taken
- current_step: Current time step
- car_states: Historical car states (optional)
- predicted_positions: Predicted future positions (optional)

Returns:
- Combined plot object
"""
function plot_combined(car_position::Float64, actions::Vector{Float64}, current_step::Int;
                      car_states::Union{Vector{Vector{Float64}}, Nothing} = nothing,
                      predicted_positions::Union{Vector{Float64}, Nothing} = nothing,
                      show_metrics::Bool = true)
    # Create landscape plot
    landscape_plot = plot_landscape(car_position;
                                   car_states = car_states,
                                   predicted_positions = predicted_positions)

    # Create engine force plot with adaptive scaling
    engine_plot = plot_engine_force(actions, current_step;
                                   adaptive_scaling = true)

    # Create metrics plot if requested
    metrics_plot = if show_metrics && car_states !== nothing && current_step > 1
        create_metrics_plot(car_states, actions, current_step)
    else
        plot(title = "Performance Metrics",
             xlabel = "Time Step",
             ylabel = "Value",
             xlim = (0, length(actions)),
             grid = true,
             gridalpha = 0.3)
    end

    # Combine plots
    combined_plot = plot(landscape_plot, engine_plot, metrics_plot,
                        layout = (3, 1),
                        size = (VISUALIZATION.plot_size[1], VISUALIZATION.plot_size[2] * 3))

    return combined_plot
end

@doc """
Create comprehensive metrics plot showing multiple time series.

Args:
- car_states: Historical car states
- actions: Action history
- current_step: Current time step

Returns:
- Plot object with multiple metrics
"""
function create_metrics_plot(car_states::Vector{Vector{Float64}}, actions::Vector{Float64}, current_step::Int)
    # Extract time series data
    positions = getindex.(car_states[1:current_step], 1)
    velocities = getindex.(car_states[1:current_step], 2)

    # Calculate progress toward goal
    total_distance = abs(WORLD.initial_position - TARGET.position)
    progress = [max(0, (total_distance - abs(pos - TARGET.position)) / total_distance * 100)
                for pos in positions]

    # Calculate cumulative distance traveled
    distance_traveled = [0.0]
    for i in 2:current_step
        dist = abs(positions[i] - positions[i-1])
        push!(distance_traveled, dist + distance_traveled[end])
    end

    plt = plot(title = "Performance Metrics",
               xlabel = "Time Step",
               ylabel = "Value",
               xlim = (0, length(actions)),
               grid = true,
               gridalpha = 0.3)

    # Plot position progress
    plot!(plt, 1:current_step, positions,
          label = "Position", color = "blue", linewidth = 2, alpha = 0.8)

    # Plot velocity
    plot!(plt, 1:current_step, velocities,
          label = "Velocity", color = "red", linewidth = 2, alpha = 0.7)

    # Plot progress toward goal
    plot!(plt, 1:current_step, progress,
          label = "Goal Progress (%)", color = "green", linewidth = 2, alpha = 0.6)

    # Plot cumulative distance
    plot!(plt, 1:current_step, distance_traveled,
          label = "Distance Traveled", color = "orange", linewidth = 2, alpha = 0.5)

    return plt
end

@doc """
Create a unified animation with consistent format for any simulation.

This function creates animations with:
- Consistent 3-panel layout (landscape, forces, metrics)
- Properly updating step numbers
- Adaptive axis scaling
- Comprehensive time series metrics
- Professional formatting

Args:
- states: Vector of car states over time
- actions: Vector of actions taken
- title_base: Base title for the animation (e.g., "Naive Policy", "Active Inference")
- filename: Output filename
- predicted_states: Optional predicted future states (for AI only)
- adaptive_scaling: Whether to use adaptive y-axis scaling

Returns:
- Nothing (saves animation to file)
"""
function create_unified_animation(states::Vector{Vector{Float64}},
                                 actions::Vector{Float64},
                                 title_base::String,
                                 filename::String;
                                 predicted_states::Union{Matrix{Float64}, Nothing} = nothing,
                                 adaptive_scaling::Bool = true,
                                 fps::Int = 24)
    @info "Creating unified animation: $title_base"

    # Determine if this is an AI or naive simulation
    is_ai = predicted_states !== nothing

    # Use the shorter of states or actions length to avoid index errors
    max_frames = min(length(states), length(actions))
    anim = @animate for i in 1:max_frames
        @info "Generating animation frame $i of $max_frames for $title_base"

        # Get current state and predictions
        current_pos = states[i][1]

        if is_ai && i <= size(predicted_states, 1)
            current_predictions = predicted_states[i, :]
        elseif is_ai
            current_predictions = predicted_states[end, :]
        else
            current_predictions = nothing
        end

        # Create combined plot
        plt = plot_combined(current_pos, actions, i;
                           car_states = states[1:i],
                           predicted_positions = current_predictions,
                           show_metrics = true)

        # Update title with current step
        plot!(plt, title = "$title_base - Step $i")

        # Progress update every 10 frames
        if i % 10 == 0
            @info "Animation progress: $(round(i/max_frames*100, digits=1))% complete ($i/$max_frames frames)"
        end

        # Add comprehensive step information
        current_state = states[i]
        if i > 1
            prev_state = states[i-1]
            velocity_change = current_state[2] - prev_state[2]
            position_change = current_state[1] - prev_state[1]
        else
            velocity_change = 0.0
            position_change = 0.0
        end

        # Calculate progress toward goal
        distance_to_goal = abs(current_pos - TARGET.position)
        total_distance = abs(WORLD.initial_position - TARGET.position)
        progress_pct = max(0, (total_distance - distance_to_goal) / total_distance * 100)

        # Get action for this step (handle length mismatch)
        action_text = if i <= length(actions)
            "Action: $(round(actions[i], digits=3))"
        else
            "Action: N/A (initial state)"
        end

        info_text = """
        Position: $(round(current_pos, digits=3))
        Velocity: $(round(current_state[2], digits=3))
        ΔPos: $(round(position_change, digits=3))
        ΔVel: $(round(velocity_change, digits=3))
        Goal Distance: $(round(distance_to_goal, digits=3))
        Progress: $(round(progress_pct, digits=1))%
        $action_text
        """

        annotate!(plt, 0.02, 0.95, text(info_text, :black, :left, 8), subplot = 1)
    end

    # Save animation
    @info "Saving animation to $filename with $fps fps..."
    save_animation(anim, filename, fps = fps)
    @info "Animation saved successfully: $filename"
end

@doc """
Create animation of the simulation.

Args:
- states: Vector of car states over time
- actions: Vector of actions taken
- predicted_states: Matrix of predicted future states at each time step

Returns:
- Animation object
"""
function create_animation(states::Vector{Vector{Float64}},
                         actions::Vector{Float64},
                         predicted_states::Matrix{Float64})
    T = size(predicted_states, 2)  # Planning horizon

    anim = @animate for i in 1:length(states)
        # Create landscape plot with current state
        current_pos = states[i][1]

        # Get predicted positions for current time step
        if i <= size(predicted_states, 1)
            current_predictions = predicted_states[i, :]
        else
            current_predictions = predicted_states[end, :]
        end

        # Create combined plot
        plt = plot_combined(current_pos, actions, i;
                           car_states = states[1:i],
                           predicted_positions = current_predictions,
                           show_metrics = true)

        # Update title with current step
        plot!(plt, title = "Active Inference - Step $i")

        # Add comprehensive step information
        current_state = states[i]
        if i > 1
            prev_state = states[i-1]
            velocity_change = current_state[2] - prev_state[2]
            position_change = current_state[1] - prev_state[1]
        else
            velocity_change = 0.0
            position_change = 0.0
        end

        # Calculate progress toward goal
        distance_to_goal = abs(current_pos - TARGET.position)
        total_distance = abs(WORLD.initial_position - TARGET.position)
        progress_pct = max(0, (total_distance - distance_to_goal) / total_distance * 100)

        info_text = """
        Position: $(round(current_pos, digits=3))
        Velocity: $(round(current_state[2], digits=3))
        ΔPos: $(round(position_change, digits=3))
        ΔVel: $(round(velocity_change, digits=3))
        Goal Distance: $(round(distance_to_goal, digits=3))
        Progress: $(round(progress_pct, digits=1))%
        Action: $(round(actions[i], digits=3))
        """

        # Add comprehensive step information to landscape plot
        annotate!(plt, 0.02, 0.95, text(info_text, :black, :left, 8), subplot = 1)
    end

    return anim
end

@doc """
Save animation to file.

Args:
- animation: Animation object
- filename: Output filename (without extension)
- fps: Frames per second
"""
function save_animation(animation, filename::String = OUTPUTS.ai_animation;
                       fps::Int = VISUALIZATION.animation_fps)
    @info "Starting animation save process..."
    @info "Saving to: $filename"

    # Ensure output directory exists
    output_dir = dirname(filename)
    if !isdir(output_dir)
        @info "Creating output directory: $output_dir"
        mkpath(output_dir)
    end

    @info "Calling gif() function..."
    gif(animation, filename, fps = fps, show_msg = true)  # Enable show_msg for progress
    @info "Animation save completed successfully"
end

@doc """
Save landscape plot to file.

Args:
- plot: Plot object
- filename: Output filename
"""
function save_plot(plot, filename::String)
    savefig(plot, filename)
end

@doc """
Helper function to get height at a specific position.
"""
function height_at_position(x::Float64)
    _, _, _, height = create_physics()
    return height(x)
end


@doc """
Get color scheme based on theme selection.

Args:
- theme: Theme symbol (:default, :dark, :colorblind_friendly)

Returns:
- Named tuple with color definitions
"""
function get_color_scheme(theme::Symbol)
    if theme == :dark
        return (
            landscape = "#4a90e2",
            car = "#ff6b6b",
            goal = "#51cf66",
            trajectory = "#74c0fc",
            predictions = "#ffd43b",
            uncertainty = "#868e96",
            goal_region = "#51cf66",
            trajectory_start = "#339af0",
            trajectory_end = "#74c0fc",
            contour = "#495057",
            velocity_field = "#ced4da"
        )
    elseif theme == :colorblind_friendly
        return (
            landscape = "#1f77b4",
            car = "#ff7f0e",
            goal = "#2ca02c",
            trajectory = "#9467bd",
            predictions = "#d62728",
            uncertainty = "#8c564b",
            goal_region = "#2ca02c",
            trajectory_start = "#1f77b4",
            trajectory_end = "#9467bd",
            contour = "#7f7f7f",
            velocity_field = "#bcbd22"
        )
    else  # default
        return (
            landscape = "#2c3e50",
            car = "#e74c3c",
            goal = "#f39c12",
            trajectory = "#3498db",
            predictions = "#2ecc71",
            uncertainty = "#95a5a6",
            goal_region = "#f39c12",
            trajectory_start = "#9b59b6",
            trajectory_end = "#3498db",
            contour = "#bdc3c7",
            velocity_field = "#34495e"
        )
    end
end

@doc """
Add velocity field arrows to the plot.

Args:
- plt: Plot object to modify
- x_coords: X coordinates for field calculation
- color: Color for velocity arrows
"""
function add_velocity_field!(plt, x_coords, color)
    # Calculate velocity field (simplified)
    for x in x_coords[1:10:end]  # Sample every 10th point
        # Velocity magnitude based on slope (simplified)
        slope = (height_at_position(x + 0.01) - height_at_position(x - 0.01)) / 0.02
        velocity = -slope * 0.1  # Negative slope means positive velocity

        if abs(velocity) > 0.001
            # Add arrow indicating velocity direction
            arrow_x = x
            arrow_y = height_at_position(x)
            arrow_dx = velocity * 0.1
            arrow_dy = slope * 0.1

            # Use annotation with arrow
            annotate!(plt, arrow_x, arrow_y,
                     text("", color, :center, 12, rotation = atan(arrow_dy/arrow_dx)*180/π))
        end
    end
end

@doc """
Create enhanced engine force plot with multiple features.

Args:
- actions: Vector of actions taken over time
- current_step: Current time step
- predictions: Optional predicted actions
- show_limits: Whether to show force limits
- show_statistics: Whether to show action statistics
- theme: Color theme

Returns:
- Plot object
"""
function plot_engine_force(actions::Vector{Float64}, current_step::Int;
                          predictions::Union{Vector{Float64}, Nothing} = nothing,
                          show_limits::Bool = true,
                          show_statistics::Bool = true,
                          theme::Symbol = :default,
                          adaptive_scaling::Bool = false)

    colors = get_color_scheme(theme)

    # Calculate y-limits based on scaling mode
    y_limits = if adaptive_scaling && current_step > 0 && current_step <= length(actions)
        current_actions = actions[1:min(current_step, length(actions))]
        if !isempty(current_actions)
            data_range = maximum(current_actions) - minimum(current_actions)
            if data_range > 0
                # Add 20% padding to data range
                center = mean(current_actions)
                half_range = data_range / 2 * 1.2
                (center - half_range, center + half_range)
            else
                VISUALIZATION.engine_force_limits  # fallback to fixed limits
            end
        else
            VISUALIZATION.engine_force_limits
        end
    else
        VISUALIZATION.engine_force_limits
    end

    plt = plot(title = "Engine Force (Agent Actions)",
               xlabel = "Time Step",
               ylabel = "Force",
               label = "Actual Actions",
               color = colors.car,
               linewidth = 2,
               xlim = (0, max(current_step, length(actions))),
               ylim = y_limits,
               size = VISUALIZATION.plot_size,
               grid = true,
               gridalpha = 0.3)

    # Plot actual actions
    plot!(plt, 1:min(current_step, length(actions)), actions[1:min(current_step, length(actions))],
          color = colors.car, linewidth = 2, alpha = 0.8)

    # Add predicted actions if provided
    if predictions !== nothing && !isempty(predictions)
        plot!(plt, (current_step+1):length(actions), predictions,
              color = colors.predictions,
              linewidth = 2,
              linestyle = :dash,
              alpha = 0.7,
              label = "Predicted Actions")
    end

    # Add force limits if requested
    if show_limits
        hline!(plt, [PHYSICS.engine_force_limit],
               color = colors.goal,
               linestyle = :dash,
               alpha = 0.6,
               label = "Force Limits")
        hline!(plt, [-PHYSICS.engine_force_limit],
               color = colors.goal,
               linestyle = :dash,
               alpha = 0.6,
               label = "")
    end

    # Add statistics if requested
    if show_statistics && current_step > 0 && current_step <= length(actions)
        current_actions = actions[1:current_step]
        if !isempty(current_actions)
            stats_text = @sprintf("μ=%.3f, σ=%.3f, max=%.3f",
                                 mean(current_actions),
                                 std(current_actions),
                                 maximum(abs, current_actions))

            annotate!(plt, length(actions)*0.02, y_limits[2]*0.8,
                     text(stats_text, :black, :left, 10))
        end
    end

    return plt
end

@doc """
Create comprehensive dashboard with multiple subplots.

Args:
- car_position: Current car position
- actions: Vector of actions taken
- current_step: Current time step
- car_states: Historical car states
- predicted_positions: Predicted future positions
- predicted_actions: Predicted future actions
- performance_metrics: Dictionary of performance metrics

Returns:
- Combined plot object
"""
function create_dashboard(car_position::Float64, actions::Vector{Float64}, current_step::Int;
                         car_states::Union{Vector{Vector{Float64}}, Nothing} = nothing,
                         predicted_positions::Union{Vector{Float64}, Nothing} = nothing,
                         predicted_actions::Union{Vector{Float64}, Nothing} = nothing,
                         performance_metrics::Union{Dict{String, Any}, Nothing} = nothing,
                         theme::Symbol = :default)

    colors = get_color_scheme(theme)

    # Create layout
    layout = @layout [a; b; c]

    # Landscape plot (top)
    landscape_plot = plot_landscape(car_position;
                                   car_states = car_states,
                                   predicted_positions = predicted_positions,
                                   show_height_contours = true,
                                   theme = theme)

    # Engine force plot (middle)
    engine_plot = plot_engine_force(actions, current_step;
                                   predictions = predicted_actions,
                                   show_statistics = true,
                                   theme = theme)

    # Performance metrics plot (bottom)
    metrics_plot = if performance_metrics !== nothing
        create_metrics_plot(performance_metrics, colors)
    else
        plot(title = "Performance Metrics",
             xlabel = "Time Step",
             ylabel = "Value",
             legend = :topright)
    end

    # Combine all plots
    dashboard = plot(landscape_plot, engine_plot, metrics_plot,
                    layout = layout,
                    size = (VISUALIZATION.plot_size[1], VISUALIZATION.plot_size[2] * 3))

    return dashboard
end

@doc """
Create performance metrics visualization.

Args:
- metrics: Dictionary of performance metrics
- colors: Color scheme

Returns:
- Plot object
"""
function create_metrics_plot(metrics::Dict{String, Any}, colors)
    plt = plot(title = "Performance Metrics",
               xlabel = "Time Step",
               ylabel = "Value",
               legend = :topright)

    # Plot metrics over time if available
    if haskey(metrics, "prediction_errors")
        plot!(plt, metrics["prediction_errors"],
              label = "Prediction Error",
              color = colors.predictions,
              alpha = 0.7)
    end

    if haskey(metrics, "inference_times")
        plot!(plt, metrics["inference_times"],
              label = "Inference Time",
              color = colors.trajectory,
              alpha = 0.7)
    end

    # Add summary statistics as text
    if haskey(metrics, "avg_prediction_error")
        error_text = @sprintf("Avg Error: %.4f", metrics["avg_prediction_error"])
        time_text = @sprintf("Avg Inf Time: %.4f", metrics["avg_inference_time"])

        annotate!(plt, 0.02, 0.98, text(error_text, :black, :left, 10))
        annotate!(plt, 0.02, 0.90, text(time_text, :black, :left, 10))
    end

    return plt
end

@doc """
Create real-time plotting function for live monitoring.

Args:
- data_source: Function that returns current state
- update_interval: Update interval in seconds
- max_points: Maximum number of points to display

Returns:
- Function to start/stop real-time plotting
"""
function create_realtime_plotter(data_source::Function;
                                update_interval::Float64 = 0.1,
                                max_points::Int = 1000)

    plt = nothing
    data_buffer = []
    running = false

    function update_plot()
        if !running
            return
        end

        # Get current data
        current_data = data_source()

        # Add to buffer
        push!(data_buffer, current_data)
        if length(data_buffer) > max_points
            popfirst!(data_buffer)
        end

        # Update plot
        if !isnothing(plt)
            # This would need display() to actually show in real-time
            # For now, just store the plot
        end

        # Schedule next update
        Timer(update_interval) do timer
            update_plot()
        end
    end

    function start()
        running = true
        update_plot()
    end

    function stop()
        running = false
    end

    return start, stop, () -> data_buffer
end

using HypergeometricFunctions: _₂F₁

@doc """
Create a horizontally concatenated animation from two animations for side-by-side comparison.

This function takes two animations (naive and AI) and creates a new animation
that shows them side by side for direct comparison. The animations are synchronized
by frame count and padded if necessary.

Args:
- naive_anim: Animation object for naive policy
- ai_anim: Animation object for active inference policy
- title: Title for the comparison animation
- filename: Output filename
- fps: Frames per second for the output animation

Returns:
- Nothing (saves animation to file)
"""
function create_comparison_animation(naive_states, naive_actions, ai_states, ai_actions, ai_predicted_states, title::String, filename::String;
                                   naive_engine_forces::Union{Vector{Float64}, Nothing} = nothing, fps::Int = 24)
    @info "Creating comprehensive comparison animation: $title"

    # Use the shorter of states or actions length to avoid index errors
    naive_max_frames = min(length(naive_states), length(naive_actions))
    ai_max_frames = min(length(ai_states), length(ai_actions))
    max_frames = max(naive_max_frames, ai_max_frames)

    # Get landscape coordinates for both animations
    x_coords, y_coords = get_landscape_coordinates()

    # Create comparison animation with all panels
    comparison_anim = @animate for i in 1:max_frames
        @info "Generating comprehensive comparison frame $i of $max_frames"

        # === NAIVE POLICY PANELS (LEFT COLUMN) ===
        if i <= length(naive_states) && i <= length(naive_actions)
            # Naive landscape plot
            naive_landscape = plot(x_coords, y_coords,
                                 title = "Naive Policy - Landscape",
                                 label = "Landscape",
                                 color = "black",
                                 xlabel = "Position",
                                 ylabel = "Height",
                                 grid = true,
                                 gridalpha = 0.3,
                                 size = (400, 300))

            # Add car position and trajectory
            scatter!(naive_landscape, [naive_states[i][1]], [height_at_position(naive_states[i][1])],
                    label = "Car", color = "red", markersize = 8)
            scatter!(naive_landscape, [TARGET.position], [height_at_position(TARGET.position)],
                    label = "Goal", color = "orange", markersize = 10, marker = :star)
            
            # Add trajectory up to current point
            if i > 1
                positions = getindex.(naive_states[1:i], 1)
                heights = height_at_position.(positions)
                plot!(naive_landscape, positions, heights,
                      label = "Trajectory", color = "blue", alpha = 0.6, linewidth = 2)
            end

            # Naive engine force plot with adaptive scaling
            # Use actual engine forces for proper scaling
            actual_naive_forces = if naive_engine_forces !== nothing && i <= length(naive_engine_forces)
                naive_engine_forces[1:i]
            else
                naive_actions[1:i]  # Fallback for backward compatibility
            end
            naive_engine = plot_engine_force(actual_naive_forces, i,
                                           show_limits = true,
                                           show_statistics = false,
                                           adaptive_scaling = true,
                                           theme = :default)
            plot!(naive_engine, title = "Naive Policy - Engine Force", size = (400, 300))
            
            # Add force limits
            hline!(naive_engine, [PHYSICS.engine_force_limit], color = "red", linestyle = :dash, alpha = 0.5, label = "")
            hline!(naive_engine, [-PHYSICS.engine_force_limit], color = "red", linestyle = :dash, alpha = 0.5, label = "")

            # Naive metrics plot
            naive_metrics = create_metrics_plot(naive_states[1:i], naive_actions[1:i], i)
            plot!(naive_metrics, title = "Naive Policy - Metrics", size = (400, 300))
        else
            # Padding plots for completed naive simulation
            naive_landscape = plot(title = "Naive Policy - Completed", size = (400, 300))
            naive_engine = plot(title = "Naive Engine - Completed", size = (400, 300))
            naive_metrics = plot(title = "Naive Metrics - Completed", size = (400, 300))
        end

        # === ACTIVE INFERENCE PANELS (RIGHT COLUMN) ===
        if i <= length(ai_states) && i <= length(ai_actions)
            ai_current_pos = ai_states[i][1]

            # Get predictions for current time step
            current_predictions = if i <= size(ai_predicted_states, 1)
                ai_predicted_states[i, :]
            else
                ai_predicted_states[end, :]
            end

            # AI landscape plot
            ai_landscape = plot(x_coords, y_coords,
                              title = "Active Inference - Landscape",
                              label = "Landscape",
                              color = "black",
                              xlabel = "Position",
                              ylabel = "Height",
                              grid = true,
                              gridalpha = 0.3,
                              size = (400, 300))

            # Add car position and trajectory
            scatter!(ai_landscape, [ai_current_pos], [height_at_position(ai_current_pos)],
                    label = "Car", color = "red", markersize = 8)
            scatter!(ai_landscape, [TARGET.position], [height_at_position(TARGET.position)],
                    label = "Goal", color = "orange", markersize = 10, marker = :star)

            # Add trajectory up to current point
            if i > 1
                positions = getindex.(ai_states[1:i], 1)
                heights = height_at_position.(positions)
                plot!(ai_landscape, positions, heights,
                      label = "Trajectory", color = "purple", alpha = 0.6, linewidth = 2)
            end

            # Add predicted future positions
            if current_predictions !== nothing && !isempty(current_predictions)
                scatter!(ai_landscape, current_predictions, height_at_position.(current_predictions),
                        label = "Predictions", color = "green", alpha = 0.7, markersize = 6, marker = :diamond)
            end

            # AI engine force plot with adaptive scaling
            ai_engine = plot_engine_force(ai_actions[1:i], i,
                                        show_limits = true,
                                        show_statistics = false,
                                        adaptive_scaling = true,
                                        theme = :default)
            plot!(ai_engine, title = "Active Inference - Engine Force", size = (400, 300))
            
            # Add force limits
            hline!(ai_engine, [PHYSICS.engine_force_limit], color = "red", linestyle = :dash, alpha = 0.5, label = "")
            hline!(ai_engine, [-PHYSICS.engine_force_limit], color = "red", linestyle = :dash, alpha = 0.5, label = "")

            # AI metrics plot
            ai_metrics = create_metrics_plot(ai_states[1:i], ai_actions[1:i], i)
            plot!(ai_metrics, title = "Active Inference - Metrics", size = (400, 300))
        else
            # Padding plots for completed AI simulation
            ai_landscape = plot(title = "Active Inference - Completed", size = (400, 300))
            ai_engine = plot(title = "AI Engine - Completed", size = (400, 300))
            ai_metrics = plot(title = "AI Metrics - Completed", size = (400, 300))
        end

        # Create comprehensive side-by-side layout (3x2 grid)
        comparison_plot = plot(naive_landscape, ai_landscape,
                              naive_engine, ai_engine,
                              naive_metrics, ai_metrics,
                              layout = (3, 2),
                              size = (1200, 900),  # Larger size for 6 panels
                              plot_title = "$title - Step $i")

        # Progress update every 10 frames
        if i % 10 == 0
            @info "Comprehensive comparison animation progress: $(round(i/max_frames*100, digits=1))% complete"
        end
    end

    # Save the comparison animation
    @info "Saving comprehensive comparison animation to $filename..."
    save_animation(comparison_anim, filename, fps = fps)
    @info "Comprehensive comparison animation saved successfully: $filename"
end

# Export public functions
export plot_landscape, plot_engine_force, plot_combined, create_animation,
       save_animation, save_plot, create_dashboard, create_realtime_plotter,
       get_color_scheme, create_comparison_animation

end # module Visualization
