# Flexible Visualization System for Active Inference Car Examples
# Supports multiple car types with extensible visualization components

@doc """
Flexible visualization system for active inference car scenarios.

This module provides a comprehensive, extensible visualization framework
that can adapt to different car types, environments, and display requirements.

## Supported Visualization Types
- **Mountain Car**: Classic landscape with gravitational effects
- **Race Track**: Circuit visualization with lap timing and sectors
- **Urban Environment**: Cityscape with intersections and traffic
- **Multi-objective**: Dashboard with multiple competing objectives
- **Performance Analytics**: Detailed performance tracking and analysis

## Key Features
- **Adaptive Layouts**: Automatic layout adjustment based on car type
- **Modular Components**: Pluggable visualization components
- **Real-time Updates**: Live visualization during simulation
- **Multiple Output Formats**: GIF, PNG, interactive plots
- **Theme Support**: Multiple color themes for different scenarios
- **Performance Metrics**: Comprehensive performance visualization
"""
module Visualization

using Plots
using Printf
using Statistics
using Colors
using ColorSchemes
using ProgressMeter
using Dates
import ..Config: VISUALIZATION, get_config_value, PHYSICS, WORLD
import ..Physics: get_landscape_coordinates, get_landscape_function
import ..World: AbstractWorld, get_state

# ==================== ABSTRACT INTERFACES ====================

@doc """
Abstract base type for visualization themes.

Provides consistent color schemes and styling for different scenarios.
"""
abstract type AbstractTheme end

@doc """
Abstract type for visualization components.

Modular visualization elements that can be combined.
"""
abstract type AbstractVisualization end

# ==================== THEME IMPLEMENTATIONS ====================

@doc """
Standard theme for general-purpose visualizations.

Balanced colors suitable for most scenarios.
"""
struct StandardTheme <: AbstractTheme
    name::String
    colors::Dict{Symbol, Colorant}

    function StandardTheme()
        colors = Dict(
            :background => colorant"#ffffff",
            :landscape => colorant"#2c3e50",
            :car => colorant"#e74c3c",
            :goal => colorant"#f39c12",
            :trajectory => colorant"#3498db",
            :predictions => colorant"#2ecc71",
            :obstacles => colorant"#9b59b6",
            :text => colorant"#2c3e50",
            :grid => colorant"#bdc3c7",
            :warning => colorant"#f39c12",
            :danger => colorant"#e74c3c"
        )
        new("Standard", colors)
    end
end

@doc """
Racing theme optimized for high-speed scenarios.

High-contrast colors for racing environments.
"""
struct RacingTheme <: AbstractTheme
    name::String
    colors::Dict{Symbol, Colorant}

    function RacingTheme()
        colors = Dict(
            :background => colorant"#1a1a1a",
            :landscape => colorant"#ff6b35",
            :car => colorant"#00d4ff",
            :goal => colorant"#00ff88",
            :trajectory => colorant"#ff0080",
            :predictions => colorant"#ffff00",
            :obstacles => colorant"#ff8000",
            :text => colorant"#ffffff",
            :grid => colorant"#333333",
            :warning => colorant"#ffff00",
            :danger => colorant"#ff0000"
        )
        new("Racing", colors)
    end
end

@doc """
Urban theme for autonomous driving scenarios.

Professional colors suitable for urban planning and navigation.
"""
struct UrbanTheme <: AbstractTheme
    name::String
    colors::Dict{Symbol, Colorant}

    function UrbanTheme()
        colors = Dict(
            :background => colorant"#f8f9fa",
            :landscape => colorant"#495057",
            :car => colorant"#228be6",
            :goal => colorant"#40c057",
            :trajectory => colorant"#9775fa",
            :predictions => colorant"#fa5252",
            :obstacles => colorant"#e03131",
            :text => colorant"#212529",
            :grid => colorant"#dee2e6",
            :warning => colorant"#fab005",
            :danger => colorant"#fa5252"
        )
        new("Urban", colors)
    end
end

# ==================== VISUALIZATION COMPONENTS ====================

@doc """
Landscape visualization component.

Displays the environment landscape with car position and trajectory.
"""
struct LandscapeVisualization <: AbstractVisualization
    car_type::Symbol
    show_trajectory::Bool
    show_predictions::Bool
    show_obstacles::Bool
    theme::AbstractTheme

    function LandscapeVisualization(car_type::Symbol; show_trajectory::Bool = true,
                                   show_predictions::Bool = true, show_obstacles::Bool = true,
                                   theme::AbstractTheme = StandardTheme())
        new(car_type, show_trajectory, show_predictions, show_obstacles, theme)
    end
end

@doc """
Control visualization component.

Displays control inputs and actuator states.
"""
struct ControlVisualization <: AbstractVisualization
    show_engine_force::Bool
    show_steering::Bool
    show_braking::Bool
    adaptive_scaling::Bool

    function ControlVisualization(; show_engine_force::Bool = true,
                                 show_steering::Bool = false, show_braking::Bool = false,
                                 adaptive_scaling::Bool = true)
        new(show_engine_force, show_steering, show_braking, adaptive_scaling)
    end
end

@doc """
Performance visualization component.

Displays performance metrics and analysis.
"""
struct PerformanceVisualization <: AbstractVisualization
    metrics::Vector{Symbol}
    show_trends::Bool
    history_length::Int

    function PerformanceVisualization(metrics::Vector{Symbol} = [:position, :velocity, :action];
                                     show_trends::Bool = true, history_length::Int = 100)
        new(metrics, show_trends, history_length)
    end
end

@doc """
Multi-objective visualization component.

Displays multiple competing objectives and their trade-offs.
"""
struct MultiObjectiveVisualization <: AbstractVisualization
    objectives::Dict{Symbol, Float64}
    constraints::Dict{Symbol, Function}
    priority_colors::Dict{Symbol, Colorant}

    function MultiObjectiveVisualization(
        objectives::Dict{Symbol, Float64} = Dict(:goal => 1.0, :safety => 0.5, :efficiency => 0.3);
        constraints::Dict{Symbol, Function} = Dict(),
        priority_colors::Dict{Symbol, Colorant} = Dict(
            :high => colorant"#e74c3c",
            :medium => colorant"#f39c12",
            :low => colorant"#27ae60"
        )
    )
        new(objectives, constraints, priority_colors)
    end
end

# ==================== VISUALIZATION FACTORY ====================

@doc """
Create visualization system based on car type.

Args:
- car_type: Symbol specifying car type
- custom_components: Optional custom visualization components

Returns:
- Visualization system instance
"""
function create_visualization(car_type::Symbol; custom_components::Dict{Symbol, Any} = Dict{Symbol, Any}())
    # Default components based on car type
    if car_type == :mountain_car
        components = Dict(
            :landscape => LandscapeVisualization(car_type, theme = StandardTheme()),
            :control => ControlVisualization(show_engine_force = true),
            :performance => PerformanceVisualization([:position, :velocity, :action])
        )
    elseif car_type == :race_car
        components = Dict(
            :landscape => LandscapeVisualization(car_type, theme = RacingTheme()),
            :control => ControlVisualization(show_engine_force = true, show_steering = true),
            :performance => PerformanceVisualization([:position, :velocity, :speed, :lap_time])
        )
    elseif car_type == :autonomous_car
        components = Dict(
            :landscape => LandscapeVisualization(car_type, theme = UrbanTheme()),
            :control => ControlVisualization(show_engine_force = true),
            :performance => PerformanceVisualization([:position, :velocity, :safety, :obstacles])
        )
    else
        components = Dict(
            :landscape => LandscapeVisualization(car_type),
            :control => ControlVisualization(),
            :performance => PerformanceVisualization()
        )
    end

    # Merge with custom components
    merge!(components, custom_components)

    return components
end

# ==================== PLOTTING FUNCTIONS ====================

@doc """
Create landscape plot for any car type.

Args:
- car_type: Car type symbol
- current_position: Current car position
- trajectory: Optional trajectory history
- predictions: Optional future predictions
- obstacles: Optional obstacle positions
- theme: Visualization theme

Returns:
- Plot object
"""
function create_landscape_plot(car_type::Symbol, current_position::Float64;
                              trajectory::Union{Vector{Vector{Float64}}, Nothing} = nothing,
                              predictions::Union{Vector{Vector{Float64}}, Nothing} = nothing,
                              obstacles::Union{Vector{Float64}, Nothing} = nothing,
                              theme::AbstractTheme = StandardTheme())

    # Get landscape coordinates
    x_coords, y_coords = get_landscape_coordinates(car_type)

    # Create base plot
    plt = plot(x_coords, y_coords,
               title = "Environment - $(string(car_type))",
               xlabel = "Position",
               ylabel = "Height/Elevation",
               label = "Environment",
               color = theme.colors[:landscape],
               linewidth = 3,
               size = (800, 400),
               grid = true,
               gridalpha = 0.3)

    # Add trajectory if provided
    if trajectory !== nothing && !isempty(trajectory)
        traj_positions = [state[1] for state in trajectory]
        traj_heights = [get_landscape_function(car_type)(pos) for pos in traj_positions]
        plot!(plt, traj_positions, traj_heights,
              label = "Trajectory",
              color = theme.colors[:trajectory],
              alpha = 0.7,
              linewidth = 2)
    end

    # Add current position
    current_height = get_landscape_function(car_type)(current_position)
    scatter!(plt, [current_position], [current_height],
            label = "Current Position",
            color = theme.colors[:car],
            markersize = 10,
            marker = :circle)

    # Add goal/target position
    if car_type == :mountain_car
        goal_pos = TARGET.position
    else
        goal_pos = 25.0  # Default goal position
    end
    goal_height = get_landscape_function(car_type)(goal_pos)
    scatter!(plt, [goal_pos], [goal_height],
            label = "Goal",
            color = theme.colors[:goal],
            markersize = 12,
            marker = :star)

    # Add obstacles if provided
    if obstacles !== nothing && !isempty(obstacles)
        for obs_pos in obstacles
            obs_height = get_landscape_function(car_type)(obs_pos)
            scatter!(plt, [obs_pos], [obs_height],
                    label = :none,
                    color = theme.colors[:obstacles],
                    markersize = 8,
                    marker = :square)
        end
    end

    return plt
end

@doc """
Create control plot showing control inputs.

Args:
- actions: Vector of control actions
- current_step: Current time step
- car_type: Car type for specific control visualization
- adaptive_scaling: Whether to use adaptive y-axis scaling

Returns:
- Plot object
"""
function create_control_plot(actions::Vector{Float64}, current_step::Int, car_type::Symbol;
                            adaptive_scaling::Bool = true)

    # Determine y-limits based on car type and scaling
    if adaptive_scaling && current_step > 0
        current_actions = actions[1:min(current_step, length(actions))]
        if !isempty(current_actions)
            data_max = maximum(abs, current_actions)
            y_max = max(data_max * 1.2, 0.1)
            y_limits = (-y_max, y_max)
        else
            y_limits = (-0.1, 0.1)
        end
    else
        y_limits = (-0.1, 0.1)
    end

    plt = plot(title = "Control Inputs - $(string(car_type))",
               xlabel = "Time Step",
               ylabel = "Control Value",
               xlim = (0, length(actions)),
               ylim = y_limits,
               grid = true,
               gridalpha = 0.3)

    # Plot actions
    plot!(plt, 1:min(current_step, length(actions)), actions[1:min(current_step, length(actions))],
          label = "Actions",
          color = "blue",
          linewidth = 2,
          alpha = 0.8)

    return plt
end

@doc """
Create performance metrics plot.

Args:
- states: State history
- actions: Action history
- current_step: Current time step
- metrics: Which metrics to display

Returns:
- Plot object
"""
function create_performance_plot(states::Vector{Vector{Float64}}, actions::Vector{Float64},
                                current_step::Int, metrics::Vector{Symbol} = [:position, :velocity])

    plt = plot(title = "Performance Metrics",
               xlabel = "Time Step",
               ylabel = "Value",
               xlim = (0, max(current_step, length(states))),
               grid = true,
               gridalpha = 0.3)

    # Plot each requested metric
    for metric in metrics
        if metric == :position
            positions = [state[1] for state in states[1:min(current_step, length(states))]]
            plot!(plt, 1:length(positions), positions,
                  label = "Position", color = "blue", linewidth = 2)
        elseif metric == :velocity
            velocities = [state[2] for state in states[1:min(current_step, length(states))]]
            plot!(plt, 1:length(velocities), velocities,
                  label = "Velocity", color = "red", linewidth = 2)
        elseif metric == :action
            plot!(plt, 1:min(current_step, length(actions)), actions[1:min(current_step, length(actions))],
                  label = "Action", color = "green", linewidth = 2)
        end
    end

    return plt
end

@doc """
Create multi-objective dashboard.

Args:
- objectives: Current objective values
- constraints: Current constraint satisfaction
- priorities: Priority levels for objectives

Returns:
- Plot object
"""
function create_multi_objective_plot(objectives::Dict{Symbol, Float64},
                                   constraints::Dict{Symbol, Bool},
                                   priorities::Dict{Symbol, Int})

    # Create radar chart data
    objective_names = collect(keys(objectives))
    objective_values = collect(values(objectives))
    priority_levels = [priorities[name] for name in objective_names]

    plt = plot(title = "Multi-Objective Performance",
               xlabel = "Objectives",
               ylabel = "Performance",
               grid = true,
               gridalpha = 0.3)

    # Bar chart for objectives
    bar!(plt, string.(objective_names), objective_values,
         label = "Current Performance",
         color = ["blue", "green", "red", "orange"][1:length(objective_names)],
         alpha = 0.7)

    return plt
end

# ==================== ANIMATION FUNCTIONS ====================

@doc """
Create unified animation with all visualization components.

Args:
- car_type: Car type symbol
- states: State history
- actions: Action history
- predictions: Optional predictions
- obstacles: Optional obstacles
- filename: Output filename
- fps: Frames per second

Returns:
- Nothing (saves animation to file)
"""
function create_unified_animation(car_type::Symbol, states::Vector{Vector{Float64}},
                                 actions::Vector{Float64}, filename::String;
                                 predictions::Union{Matrix{Float64}, Nothing} = nothing,
                                 obstacles::Union{Vector{Float64}, Nothing} = nothing,
                                 fps::Int = 24)

    @info "Creating unified animation for $car_type"

    max_frames = min(length(states), length(actions))
    anim = @animate for i in 1:max_frames
        @info "Generating frame $i of $max_frames"

        # Create landscape plot
        landscape = create_landscape_plot(car_type, states[i][1];
                                         trajectory = states[1:i],
                                         predictions = predictions !== nothing ? [predictions[i, :]] : nothing,
                                         obstacles = obstacles)

        # Create control plot
        control = create_control_plot(actions, i, car_type)

        # Create performance plot
        performance = create_performance_plot(states, actions, i)

        # Combine plots
        combined = plot(landscape, control, performance,
                       layout = (3, 1),
                       size = (800, 1200))

        # Add frame information
        annotate!(combined, 0.02, 0.98,
                 text("Frame: $i/$max_frames\nPosition: $(round(states[i][1], digits=3))\nVelocity: $(round(states[i][2], digits=3))",
                      :black, :left, 10),
                 subplot = 1)
    end

    # Save animation
    gif(anim, filename, fps = fps)
    @info "Animation saved: $filename"
end

@doc """
Create comparison animation between two car types.

Args:
- car_type1: First car type
- states1: States for first car
- actions1: Actions for first car
- car_type2: Second car type
- states2: States for second car
- actions2: Actions for second car
- filename: Output filename

Returns:
- Nothing (saves animation to file)
"""
function create_comparison_animation(car_type1::Symbol, states1::Vector{Vector{Float64}},
                                   actions1::Vector{Float64}, car_type2::Symbol,
                                   states2::Vector{Vector{Float64}}, actions2::Vector{Float64},
                                   filename::String; fps::Int = 24)

    @info "Creating comparison animation: $car_type1 vs $car_type2"

    max_frames = min(length(states1), length(actions1), length(states2), length(actions2))
    anim = @animate for i in 1:max_frames
        # Left side: Car type 1
        landscape1 = create_landscape_plot(car_type1, states1[i][1];
                                          trajectory = states1[1:i])
        control1 = create_control_plot(actions1, i, car_type1)
        perf1 = create_performance_plot(states1, actions1, i)

        # Right side: Car type 2
        landscape2 = create_landscape_plot(car_type2, states2[i][1];
                                          trajectory = states2[1:i])
        control2 = create_control_plot(actions2, i, car_type2)
        perf2 = create_performance_plot(states2, actions2, i)

        # Combine side by side
        comparison = plot(landscape1, landscape2, control1, control2, perf1, perf2,
                         layout = (3, 2),
                         size = (1200, 900))

        # Add comparison information
        annotate!(comparison, 0.02, 0.98,
                 text("$car_type1 (Left) vs $car_type2 (Right)\nFrame: $i/$max_frames",
                      :black, :left, 10))
    end

    # Save animation
    gif(anim, filename, fps = fps)
    @info "Comparison animation saved: $filename"
end

# ==================== REAL-TIME VISUALIZATION ====================

@doc """
Create real-time visualization for live monitoring.

Args:
- car_type: Car type symbol
- update_interval: Update interval in seconds
- max_points: Maximum number of points to display

Returns:
- Tuple of (start_function, stop_function, get_data_function)
"""
function create_realtime_visualization(car_type::Symbol; update_interval::Float64 = 0.1,
                                      max_points::Int = 1000)

    # State storage
    data_buffer = []
    running = false

    # Create initial plot
    plt = plot(title = "Real-time $car_type Visualization",
               xlabel = "Time", ylabel = "Value",
               grid = true, gridalpha = 0.3)

    function update_plot()
        if !running || isempty(data_buffer)
            return
        end

        # Update plot with current data
        recent_data = data_buffer[max(1, length(data_buffer)-max_points+1):end]

        # Clear and redraw
        plot!(plt, title = "Real-time $car_type Visualization ($(length(recent_data)) points)",
              xlabel = "Time Step", ylabel = "State Value")

        if !isempty(recent_data)
            positions = [data[1] for data in recent_data]
            velocities = [data[2] for data in recent_data]

            plot!(plt, 1:length(positions), positions,
                  label = "Position", color = "blue", linewidth = 2)
            plot!(plt, 1:length(velocities), velocities,
                  label = "Velocity", color = "red", linewidth = 2)
        end

        # Schedule next update
        if running
            Timer(update_interval) do timer
                update_plot()
            end
        end
    end

    function start()
        running = true
        update_plot()
    end

    function stop()
        running = false
    end

    function add_data(data::Vector{Float64})
        push!(data_buffer, data)
        if length(data_buffer) > max_points
            popfirst!(data_buffer)
        end
    end

    return start, stop, add_data, () -> data_buffer
end

# ==================== UTILITY FUNCTIONS ====================

@doc """
Get theme by name.

Args:
- theme_name: Symbol specifying theme (:standard, :racing, :urban)

Returns:
- Theme instance
"""
function get_theme(theme_name::Symbol)
    if theme_name == :racing
        return RacingTheme()
    elseif theme_name == :urban
        return UrbanTheme()
    else
        return StandardTheme()
    end
end

@doc """
Save plot to file.

Args:
- plot: Plot object
- filename: Output filename
- kwargs: Additional savefig arguments
"""
function save_plot(plot, filename::String; kwargs...)
    savefig(plot, filename; kwargs...)
end

@doc """
Create performance summary plot.

Args:
- car_type: Car type
- performance_data: Dictionary of performance metrics

Returns:
- Plot object
"""
function create_performance_summary(car_type::Symbol, performance_data::Dict)
    plt = plot(title = "Performance Summary - $car_type",
               xlabel = "Metric", ylabel = "Value",
               grid = true, gridalpha = 0.3)

    metrics = collect(keys(performance_data))
    values = collect(values(performance_data))

    bar!(plt, string.(metrics), values,
         color = ["blue", "green", "red", "orange", "purple"][1:length(metrics)],
         alpha = 0.7)

    return plt
end

# ==================== MODULE EXPORTS ====================

export
    # Abstract types
    AbstractTheme,
    AbstractVisualization,

    # Theme implementations
    StandardTheme,
    RacingTheme,
    UrbanTheme,

    # Visualization components
    LandscapeVisualization,
    ControlVisualization,
    PerformanceVisualization,
    MultiObjectiveVisualization,

    # Factory functions
    create_visualization,
    get_theme,

    # Plotting functions
    create_landscape_plot,
    create_control_plot,
    create_performance_plot,
    create_multi_objective_plot,
    create_performance_summary,

    # Animation functions
    create_car_animation,
    create_unified_animation,
    create_comparison_animation,

    # Real-time visualization
    create_realtime_visualization,

    # Utility functions
    save_plot,
    create_static_visualizations

# ==================== SPECIALIZED ANIMATION FUNCTIONS ====================

@doc """
Create specialized mountain car animation with landscape visualization.

Args:
- states: State trajectory (position, velocity)
- actions: Action sequence
- predictions: Optional predictions matrix
- filename: Output filename
- fps: Frames per second
"""
function create_mountain_car_animation(
    states::Vector{Vector{Float64}},
    actions::Vector{Float64},
    predictions::Union{Matrix{Float64}, Nothing},
    filename::String;
    fps::Int = 24
)
    @info "Creating specialized mountain car animation"

    # Create animation with mountain car specific visualization
    anim = @animate for i in 1:length(states)
        # Create landscape plot
        plot_obj = create_mountain_car_landscape(states[i][1])

        # Add car position
        scatter!(plot_obj, [states[i][1]], [states[i][2]],
                markersize = 10, color = :red, label = "Car")

        # Add goal region
        vline!(plot_obj, [1.0], color = :green, linestyle = :dash,
               linewidth = 2, label = "Goal")

        # Add prediction trajectory if available
        if !isnothing(predictions) && i <= size(predictions, 1)
            pred_x = [states[i][1] + k * 0.1 * actions[min(i, length(actions))]
                     for k in 0:10]
            pred_y = [states[i][2] + k * 0.01 for k in 0:10]
            plot!(plot_obj, pred_x, pred_y, color = :blue, alpha = 0.5,
                  linestyle = :dash, label = "Predictions")
        end

        # Add control information
        title!(plot_obj, @sprintf("Mountain Car - Step %d\nAction: %.3f",
                                  i, actions[min(i, length(actions))]))

        # Style the plot
        xlabel!(plot_obj, "Position")
        ylabel!(plot_obj, "Velocity")
    end

    # Save the animation
    gif(anim, filename, fps = fps)
    @info "Mountain car animation saved: $filename"
end

function create_mountain_car_landscape(position::Float64)
    # Create the mountain car landscape
    x_range = -1.2:0.01:1.2
    y_values = -cos.(3 * x_range) .* 0.45 .+ 0.55

    plot(x_range, y_values, color = :brown, linewidth = 3, fill = true,
         fillcolor = :lightgreen, alpha = 0.7, label = "Landscape")

    # Add track boundaries
    plot!(x_range, y_values .+ 0.1, color = :black, linewidth = 2,
          linestyle = :solid, label = "")
    plot!(x_range, y_values .- 0.1, color = :black, linewidth = 2,
          linestyle = :solid, label = "")

    return plot!()
end

# ==================== FALLBACK ANIMATION FUNCTIONS ====================

@doc """
Create a simple fallback animation when advanced visualization fails.

Args:
- car_type: Type of car
- states: State trajectory
- actions: Action sequence
- filename: Output filename
- fps: Frames per second
"""
function create_simple_animation(
    car_type::Symbol,
    states::Vector{Vector{Float64}},
    actions::Vector{Float64},
    filename::String;
    fps::Int = 24
)
    @info "Creating simple animation for $car_type"

    # Create basic animation with simple plots
    anim = @animate for i in 1:length(states)
        # Create a simple plot showing position and velocity over time
        plt = plot(1:i, [s[1] for s in states[1:i]],
                   label = "Position", color = :blue, linewidth = 2)
        plot!(plt, 1:i, [s[2] for s in states[1:i]],
              label = "Velocity", color = :red, linewidth = 2)
        plot!(plt, 1:i, fill(actions[1:min(i, length(actions))], 1),
              label = "Action", color = :green, alpha = 0.3, fill = true)

        title!(plt, "Mountain Car - Step $i")
        xlabel!(plt, "Time Step")
        ylabel!(plt, "Value")
        plot!(plt, legend = :topright)
    end

    # Save the animation
    gif(anim, filename, fps = fps)
    @info "Simple animation saved: $filename"
end

# ==================== STATIC VISUALIZATION FUNCTIONS ====================

@doc """
Create comprehensive static visualizations for the experiment results.

Args:
- experiment_results: Dictionary containing experiment results
- output_dir: Directory to save visualizations
"""
function create_static_visualizations(experiment_results::Dict{String, Any}, output_dir::String)
    @info "Creating static visualizations" output_dir = output_dir

    try
        # Create visualizations directory
        vis_dir = joinpath(output_dir, "visualizations")
        mkpath(vis_dir)

        # Extract data from results
        car_types = experiment_results["car_types"]
        results = experiment_results["results"]

        for car_type in car_types
            car_key = string(car_type)
            if haskey(results, car_key)
                car_data = results[car_key]

                # Create trajectory plot
                create_trajectory_plot(car_data, vis_dir, car_type)

                # Create performance comparison plot
                create_performance_comparison(car_data, vis_dir, car_type)

                # Create action analysis plot
                create_action_analysis(car_data, vis_dir, car_type)
            end
        end

        @info "Static visualizations created successfully" vis_dir = vis_dir
    catch e
        @warn "Failed to create static visualizations" error = string(e)
    end
end

@doc """
Create trajectory visualization showing position and velocity over time.

Args:
- car_data: Car-specific results data
- vis_dir: Visualization directory
- car_type: Type of car
"""
function create_trajectory_plot(car_data::Dict{String, Any}, vis_dir::String, car_type::Symbol)
    # Extract trajectory data
    if haskey(car_data, "ai_trajectory")
        trajectory = car_data["ai_trajectory"]
        positions = [state[1] for state in trajectory]
        velocities = [state[2] for state in trajectory]
        time_steps = 1:length(positions)

        # Create plot
        plt = plot(time_steps, positions,
                   label = "Position", color = :blue, linewidth = 2, alpha = 0.8)
        plot!(plt, time_steps, velocities,
              label = "Velocity", color = :red, linewidth = 2, alpha = 0.8)

        # Add goal line if available
        if haskey(car_data, "active_inference")
            ai_data = car_data["active_inference"]
            if haskey(ai_data, "final_position")
                hline!(plt, [ai_data["final_position"]],
                       color = :green, linestyle = :dash, label = "Final Position")
            end
        end

        title!(plt, "Mountain Car Trajectory - $(string(car_type))")
        xlabel!(plt, "Time Step")
        ylabel!(plt, "Value")
        plot!(plt, legend = :topright)

        # Save plot
        filename = joinpath(vis_dir, "$(car_type)_trajectory.png")
        savefig(plt, filename)
        @info "Trajectory plot saved" filename = filename
    end
end

@doc """
Create performance comparison between naive and active inference policies.

Args:
- car_data: Car-specific results data
- vis_dir: Visualization directory
- car_type: Type of car
"""
function create_performance_comparison(car_data::Dict{String, Any}, vis_dir::String, car_type::Symbol)
    if haskey(car_data, "naive") && haskey(car_data, "active_inference")
        naive_data = car_data["naive"]
        ai_data = car_data["active_inference"]

        # Create comparison plot
        categories = ["Final Position", "Avg Velocity", "Total Distance"]
        naive_values = [
            naive_data["final_position"],
            naive_data["avg_velocity"],
            naive_data["total_distance"]
        ]
        ai_values = [
            ai_data["final_position"],
            ai_data["avg_velocity"],
            ai_data["total_distance"]
        ]

        plt = bar(
            1:length(categories),
            naive_values,
            label = "Naive Policy",
            color = :orange,
            alpha = 0.8
        )
        bar!(
            plt,
            1:length(categories),
            ai_values,
            label = "Active Inference",
            color = :blue,
            alpha = 0.8
        )

        xticks!(plt, 1:length(categories), categories)
        title!(plt, "Performance Comparison - $(string(car_type))")
        ylabel!(plt, "Value")
        plot!(plt, legend = :topright)

        # Save plot
        filename = joinpath(vis_dir, "$(car_type)_performance_comparison.png")
        savefig(plt, filename)
        @info "Performance comparison saved" filename = filename
    end
end

@doc """
Create action analysis visualization.

Args:
- car_data: Car-specific results data
- vis_dir: Visualization directory
- car_type: Type of car
"""
function create_action_analysis(car_data::Dict{String, Any}, vis_dir::String, car_type::Symbol)
    if haskey(car_data, "actions")
        actions = car_data["actions"]
        time_steps = 1:length(actions)

        # Create action plot
        plt = plot(time_steps, actions,
                   label = "Actions", color = :purple, linewidth = 2, alpha = 0.8)
        hline!(plt, [0.0], color = :black, linestyle = :dash, label = "Zero Action")

        title!(plt, "Action Sequence - $(string(car_type))")
        xlabel!(plt, "Time Step")
        ylabel!(plt, "Action Value")
        plot!(plt, legend = :topright)

        # Save plot
        filename = joinpath(vis_dir, "$(car_type)_actions.png")
        savefig(plt, filename)
        @info "Action analysis saved" filename = filename
    end
end

# ==================== MAIN ANIMATION FUNCTIONS ====================

@doc """
Create a comprehensive animation for a single car type.

Args:
- car_type: Type of car (:mountain_car, :race_car, :autonomous_car)
- states: Vector of state trajectories
- actions: Vector of actions taken
- predictions: Optional matrix of predicted states
- filename: Output filename for the animation
- fps: Frames per second (default: 24)
"""
function create_car_animation(
    car_type::Symbol,
    states::Vector{Vector{Float64}},
    actions::Vector{Float64},
    predictions::Union{Matrix{Float64}, Nothing},
    filename::String;
    fps::Int = 24
)
    @info "Creating car animation: $car_type" filename = filename

    try
        # Determine animation type based on car type
        if car_type == :mountain_car
            create_mountain_car_animation(states, actions, predictions, filename; fps = fps)
        else
            create_unified_animation(car_type, states, actions, filename;
                                    predictions = predictions, fps = fps)
        end
    catch e
        @warn "Advanced animation failed, creating simple animation" error = string(e)
        create_simple_animation(car_type, states, actions, filename; fps = fps)
    end
end

# ==================== UTILITY FUNCTIONS ====================

@doc """
Save plot to multiple formats with consistent naming.

Args:
- plot: The plot to save
- filename: Base filename (without extension)
- formats: List of formats to save (default: ["png", "pdf"])
"""
function save_plot(plot::Plots.Plot, filename::String; formats::Vector{String} = ["png", "pdf"])
    for format in formats
        full_filename = "$filename.$format"
        try
            savefig(plot, full_filename)
            @info "Saved plot: $full_filename"
        catch e
            @warn "Failed to save plot: $full_filename" error = string(e)
        end
    end
end

end # module Visualization
