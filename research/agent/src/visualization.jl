# Visualization Module for Active Inference Agents
# Comprehensive plotting and animation for agent-environment simulations
#
# Provides comprehensive visualization including:
# - Trajectory plots (states, actions, observations)
# - Belief evolution plots
# - Performance metrics visualization
# - Animated trajectories
# - State space visualization

module Visualization

using Plots
using Statistics
using Printf

# Import types from Main (they should be loaded before this module)
import Main: StateVector, ActionVector, ObservationVector

# Import Diagnostics module
import Main.Diagnostics

"""
plot_trajectory_1d(result, output_dir; title="1D Navigation Trajectory")

Plot 1D navigation trajectory (position over time).
"""
function plot_trajectory_1d(result, output_dir::String; title="1D Navigation Trajectory")
    positions = [s[1] for s in result.states]
    actions = [a[1] for a in result.actions]
    
    # Position plot
    p1 = plot(
        1:length(positions),
        positions,
        xlabel="Step",
        ylabel="Position",
        title=title,
        legend=false,
        linewidth=2,
        color=:blue
    )
    
    # Action plot
    p2 = plot(
        1:length(actions),
        actions,
        xlabel="Step",
        ylabel="Action (Velocity)",
        title="Actions Over Time",
        legend=false,
        linewidth=2,
        color=:red
    )
    
    # Combined plot
    combined = plot(p1, p2, layout=(2,1), size=(800, 600))
    
    savefig(combined, joinpath(output_dir, "trajectory_1d.png"))
    return combined
end

"""
plot_trajectory_2d(result, output_dir; title="2D State Trajectory")

Plot 2D state trajectory (e.g., Mountain Car position-velocity).
"""
function plot_trajectory_2d(result, output_dir::String; title="2D State Trajectory")
    positions = [s[1] for s in result.states]
    velocities = [s[2] for s in result.states]
    actions = [a[1] for a in result.actions]
    
    # Position over time
    p1 = plot(
        1:length(positions),
        positions,
        xlabel="Step",
        ylabel="Position",
        title="Position Over Time",
        legend=false,
        linewidth=2,
        color=:blue
    )
    
    # Velocity over time
    p2 = plot(
        1:length(velocities),
        velocities,
        xlabel="Step",
        ylabel="Velocity",
        title="Velocity Over Time",
        legend=false,
        linewidth=2,
        color=:green
    )
    
    # Phase space (position vs velocity)
    p3 = plot(
        positions,
        velocities,
        xlabel="Position",
        ylabel="Velocity",
        title="Phase Space",
        legend=false,
        linewidth=2,
        color=:purple,
        marker=:circle,
        markersize=3,
        alpha=0.6
    )
    
    # Actions over time
    p4 = plot(
        1:length(actions),
        actions,
        xlabel="Step",
        ylabel="Action (Force)",
        title="Actions Over Time",
        legend=false,
        linewidth=2,
        color=:red
    )
    
    # Combined plot
    combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
    
    savefig(combined, joinpath(output_dir, "trajectory_2d.png"))
    return combined
end

"""
plot_diagnostics(diagnostics, output_dir)

Plot comprehensive diagnostics (memory, performance, beliefs).
"""
function plot_diagnostics(diagnostics, output_dir::String)
    summary = Diagnostics.get_comprehensive_summary(diagnostics)
    
    plots_list = []
    
    # Memory plot
    if summary["memory"]["measurements"] > 0
        mem_data = diagnostics.memory_tracer.memory_mb
        p_mem = plot(
            1:length(mem_data),
            mem_data,
            xlabel="Measurement",
            ylabel="Memory (MB)",
            title="Memory Usage",
            legend=false,
            linewidth=2,
            color=:orange
        )
        push!(plots_list, p_mem)
    end
    
    # Inference time plot
    if haskey(summary["performance"], "operations") && 
       haskey(summary["performance"]["operations"], "inference")
        times = diagnostics.performance_profiler.operation_times["inference"]
        p_time = plot(
            1:length(times),
            times,
            xlabel="Step",
            ylabel="Time (s)",
            title="Inference Time per Step",
            legend=false,
            linewidth=2,
            color=:red,
            yscale=:log10
        )
        push!(plots_list, p_time)
    end
    
    # Uncertainty plot (if available)
    if summary["beliefs"]["measurements"] > 0
        uncertainty = diagnostics.belief_tracker.uncertainty_trace
        p_unc = plot(
            1:length(uncertainty),
            uncertainty,
            xlabel="Step",
            ylabel="Uncertainty (trace of covariance)",
            title="Belief Uncertainty",
            legend=false,
            linewidth=2,
            color=:purple
        )
        push!(plots_list, p_unc)
    end
    
    if !isempty(plots_list)
        n_plots = length(plots_list)
        layout = n_plots == 1 ? (1,1) : n_plots == 2 ? (1,2) : (2,2)
        combined = plot(plots_list..., layout=layout, size=(1200, 600))
        savefig(combined, joinpath(output_dir, "diagnostics.png"))
        return combined
    end
    
    return nothing
end

"""
animate_trajectory_1d(result, output_dir; fps=10, title="1D Navigation Animation")

Create animated visualization of 1D navigation.
"""
function animate_trajectory_1d(result, output_dir::String; fps=10, title="1D Navigation Animation")
    positions = [s[1] for s in result.states]
    
    anim = @animate for t in 1:length(positions)
        # Current position
        plot(
            [t],
            [positions[t]],
            xlabel="Step",
            ylabel="Position",
            title="$title (Step $t/$(length(positions)))",
            xlim=(1, length(positions)),
            ylim=(minimum(positions) - 0.1, maximum(positions) + 0.1),
            marker=:circle,
            markersize=10,
            color=:blue,
            legend=false
        )
        
        # Add trajectory so far
        if t > 1
            plot!(
                1:t,
                positions[1:t],
                linewidth=2,
                color=:blue,
                alpha=0.5
            )
        end
    end
    
    gif(anim, joinpath(output_dir, "trajectory_1d.gif"), fps=fps)
    return anim
end

"""
animate_trajectory_2d(result, output_dir; fps=10, title="2D Trajectory Animation")

Create animated visualization of 2D state trajectory (e.g., Mountain Car).
"""
function animate_trajectory_2d(result, output_dir::String; fps=10, title="2D Trajectory Animation")
    positions = [s[1] for s in result.states]
    velocities = [s[2] for s in result.states]
    actions = [a[1] for a in result.actions]
    
    anim = @animate for t in 1:length(positions)
        # Position subplot
        p1 = plot(
            1:t,
            positions[1:t],
            xlabel="Step",
            ylabel="Position",
            title="Position (Step $t/$(length(positions)))",
            xlim=(1, length(positions)),
            ylim=(minimum(positions) - 0.1, maximum(positions) + 0.1),
            linewidth=2,
            color=:blue,
            legend=false
        )
        
        # Velocity subplot
        p2 = plot(
            1:t,
            velocities[1:t],
            xlabel="Step",
            ylabel="Velocity",
            title="Velocity",
            xlim=(1, length(velocities)),
            ylim=(minimum(velocities) - 0.01, maximum(velocities) + 0.01),
            linewidth=2,
            color=:green,
            legend=false
        )
        
        # Phase space with current position marked
        p3 = plot(
            positions[1:t],
            velocities[1:t],
            xlabel="Position",
            ylabel="Velocity",
            title="Phase Space",
            xlim=(minimum(positions) - 0.1, maximum(positions) + 0.1),
            ylim=(minimum(velocities) - 0.01, maximum(velocities) + 0.01),
            linewidth=2,
            color=:purple,
            alpha=0.5,
            legend=false
        )
        scatter!(
            [positions[t]],
            [velocities[t]],
            markersize=10,
            color=:red
        )
        
        # Actions with current action marked
        if t > 1
            p4 = plot(
                1:t-1,
                actions[1:t-1],
                xlabel="Step",
                ylabel="Action",
                title="Actions",
                xlim=(1, length(actions)),
                ylim=(minimum(actions) - 0.1, maximum(actions) + 0.1),
                linewidth=2,
                color=:red,
                legend=false
            )
        else
            p4 = plot(
                xlabel="Step",
                ylabel="Action",
                title="Actions",
                xlim=(1, length(actions)),
                ylim=(minimum(actions) - 0.1, maximum(actions) + 0.1),
                legend=false
            )
        end
        
        plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
    end
    
    gif(anim, joinpath(output_dir, "trajectory_2d.gif"), fps=fps)
    return anim
end

"""
plot_mountain_car_landscape(result, output_dir)

Plot Mountain Car trajectory on the actual landscape.
"""
function plot_mountain_car_landscape(result, output_dir::String)
    positions = [s[1] for s in result.states]
    
    # Mountain car landscape function
    x_range = range(-1.2, 0.6, length=200)
    heights = -sin.(3 * x_range)  # Simplified landscape
    
    p = plot(
        x_range,
        heights,
        xlabel="Position",
        ylabel="Height",
        title="Mountain Car Trajectory",
        label="Landscape",
        linewidth=2,
        color=:gray,
        fillrange=minimum(heights),
        fillalpha=0.3,
        fillcolor=:lightgray
    )
    
    # Plot trajectory as dots on landscape
    traj_heights = -sin.(3 * positions)
    scatter!(
        positions,
        traj_heights,
        label="Agent Trajectory",
        markersize=4,
        color=:blue,
        alpha=0.6
    )
    
    # Mark start and end
    scatter!(
        [positions[1]],
        [traj_heights[1]],
        label="Start",
        markersize=10,
        color=:green,
        marker=:star
    )
    scatter!(
        [positions[end]],
        [traj_heights[end]],
        label="End",
        markersize=10,
        color=:red,
        marker=:star
    )
    
    # Mark goal
    vline!(
        [0.5],
        label="Goal",
        linewidth=2,
        linestyle=:dash,
        color=:gold
    )
    
    savefig(p, joinpath(output_dir, "mountain_car_landscape.png"))
    return p
end

"""
generate_all_visualizations(result, output_dir, state_dim)

Generate all appropriate visualizations based on state dimensionality.
"""
function generate_all_visualizations(result, output_dir::String, state_dim::Int)
    mkpath(output_dir)
    
    println("Generating visualizations...")
    
    plots_created = []
    
    # Trajectory plots (static only - animations handled separately)
    if state_dim == 1
        println("  • Creating 1D trajectory plot...")
        plot_trajectory_1d(result, output_dir)
        push!(plots_created, "trajectory_1d.png")
        
    elseif state_dim == 2
        println("  • Creating 2D trajectory plot...")
        plot_trajectory_2d(result, output_dir)
        push!(plots_created, "trajectory_2d.png")
        
        println("  • Creating mountain car landscape plot...")
        plot_mountain_car_landscape(result, output_dir)
        push!(plots_created, "mountain_car_landscape.png")
    end
    
    # Diagnostics plots
    if result.diagnostics !== nothing
        println("  • Creating diagnostics plots...")
        plot_diagnostics(result.diagnostics, output_dir)
        push!(plots_created, "diagnostics.png")
    end
    
    println("  ✓ Created $(length(plots_created)) visualization(s)")
    return plots_created
end

# Export public API
export plot_trajectory_1d, plot_trajectory_2d
export plot_diagnostics, plot_mountain_car_landscape
export animate_trajectory_1d, animate_trajectory_2d
export generate_all_visualizations

end # module Visualization

