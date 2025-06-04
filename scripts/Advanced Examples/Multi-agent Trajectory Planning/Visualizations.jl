module Visualizations

using Plots
using ..Environment

export plot_marker_at_position!, plot_agent_naive_plan!, animate_paths
# Add new exports for functions moved from Environment.jl
export plot_rectangle!, plot_environment!, plot_environment
# Add export for the new ELBO convergence function
export plot_elbo_convergence

"""
    plot_marker_at_position!(p, radius, position; color="red", markersize=10.0, alpha=1.0, label="")

Plot a circular marker at the specified position with the given radius.

# Arguments
- `p`: The plot to add the marker to
- `radius`: Radius of the circular marker
- `position`: A tuple (x, y) representing the center position
- `color`: Color of the marker (default: "red")
- `markersize`: Size of the marker (default: 10.0)
- `alpha`: Transparency of the marker (default: 1.0)
- `label`: Label for the marker in the plot legend (default: "")

# Returns
- The updated plot object
"""
function plot_marker_at_position!(p, radius, position; color="red", markersize=10.0, alpha=1.0, label="")
    # Draw the agent as a circle with the given radius
    θ = range(0, 2π, 100)
    
    x_coords = position[1] .+ radius .* cos.(θ)
    y_coords = position[2] .+ radius .* sin.(θ)
    
    plot!(p, Shape(x_coords, y_coords); color=color, label=label, alpha=alpha)
    return p
end

"""
    plot_agent_naive_plan!(p, agent; color = "blue")

Plot an agent's naive plan (straight line from initial to target position).

# Arguments
- `p`: The plot to add the plan to
- `agent`: The agent object containing initial and target positions
- `color`: Color to use for the agent (default: "blue")

# Returns
- The updated plot object
"""
function plot_agent_naive_plan!(p, agent; color = "blue")
    plot_marker_at_position!(p, agent.radius, agent.initial_position, color = color)
    plot_marker_at_position!(p, agent.radius, agent.target_position, color = color, alpha = 0.1)
    quiver!(p, [ agent.initial_position[1] ], [ agent.initial_position[2] ], 
            quiver = ([ agent.target_position[1] - agent.initial_position[1] ], 
                     [ agent.target_position[2] -  agent.initial_position[2] ]))
end

"""
    plot_agent_plans!(p, agents)

Plot naive plans for multiple agents.

# Arguments
- `p`: The plot to add the plans to
- `agents`: List of agent objects

# Returns
- The updated plot object
"""
function plot_agent_plans!(p, agents)
    colors = Plots.palette(:tab10)
    
    for (k, agent) in enumerate(agents)
        plot_agent_naive_plan!(p, agent, color = colors[k])
    end
    
    return p
end

"""
    animate_paths(environment, agents, paths; filename = "result.gif", fps = 15, x_limits = (-20, 20), y_limits = (-20, 20), plot_size = (800, 400), show_targets = true, path_alpha = 0.8)

Create an animation of agent paths through the environment.

# Arguments
- `environment`: The environment containing obstacles
- `agents`: List of agent objects
- `paths`: Matrix of agent positions over time, dimensions: (nr_agents, nr_steps)
- `filename`: Output filename for the GIF (default: "result.gif")
- `fps`: Frames per second for the animation (default: 15)
- `x_limits`: Tuple with x-axis limits (default: (-20, 20))
- `y_limits`: Tuple with y-axis limits (default: (-20, 20))
- `plot_size`: Tuple with plot width and height (default: (800, 400))
- `show_targets`: Whether to show target positions (default: true)
- `path_alpha`: Alpha value for path lines (default: 0.8)

# Returns
- `nothing`
"""
function animate_paths(environment, agents, paths; 
                      filename = "result.gif", 
                      fps = 15, 
                      x_limits = (-20, 20), 
                      y_limits = (-20, 20), 
                      plot_size = (800, 400), 
                      show_targets = true,
                      path_alpha = 0.8)
    nr_agents, nr_steps = size(paths)
    colors = Plots.palette(:tab10)

    println("Generating animation frames...")
    animation = @animate for t in 1:nr_steps
        frame = plot_environment(environment, 
                                x_limits = x_limits, 
                                y_limits = y_limits, 
                                plot_size = plot_size)
    
        for k in 1:nr_agents
            position = paths[k, t]          
            path = paths[k, 1:t]
            
            plot_marker_at_position!(frame, agents[k].radius, position, color = colors[k])
            
            if show_targets
                plot_marker_at_position!(frame, agents[k].radius, agents[k].target_position, color = colors[k], alpha = 0.2)
            end
            
            plot!(frame, getindex.(path, 1), getindex.(path, 2); 
                 linestyle = :dash, 
                 label = "", 
                 color = colors[k], 
                 alpha = path_alpha)
        end

        frame
    end

    # Save the animation
    println("Saving animation to $filename...")
    gif(animation, filename, fps=fps, show_msg = false)
    println("Animation saved successfully.")
    
    return nothing
end

"""
    plot_elbo_convergence(elbo_values; filename = "convergence.png")

Create a plot showing the ELBO convergence during inference.

# Arguments
- `elbo_values`: Vector of ELBO values from inference iterations
- `filename`: Output filename for the plot (default: "convergence.png")

# Returns
- The plot object
"""
function plot_elbo_convergence(elbo_values; filename = "convergence.png")
    p = plot(elbo_values, 
             xlabel = "Iteration", 
             ylabel = "ELBO", 
             title = "Convergence of Inference",
             legend = false, 
             linewidth = 2,
             grid = true,
             size = (800, 400))
    
    # Add a smoothed trend line
    if length(elbo_values) > 10
        window_size = max(5, div(length(elbo_values), 20))
        smoothed = movmean(elbo_values, window_size)
        plot!(p, smoothed, linewidth = 3, color = :red, alpha = 0.7, label = "Moving Average")
    end
    
    # Save the plot
    println("Saving convergence plot to $filename...")
    savefig(p, filename)
    println("Convergence plot saved successfully.")
    
    return p
end

# Helper function for moving average
function movmean(arr, window_size)
    n = length(arr)
    result = similar(arr)
    
    for i in 1:n
        start_idx = max(1, i - div(window_size, 2))
        end_idx = min(n, i + div(window_size, 2))
        result[i] = mean(arr[start_idx:end_idx])
    end
    
    return result
end

# Functions moved from Environment.jl

"""
    plot_rectangle!(p, rect::Rectangle)

Plot a rectangle on an existing plot.

# Arguments
- `p`: The plot to add the rectangle to
- `rect`: Rectangle object with center and size properties

# Returns
- The updated plot object
"""
function plot_rectangle!(p, rect::Rectangle)
    # Calculate the x-coordinates of the four corners
    x_coords = rect.center[1] .+ rect.size[1]/2 * [-1, 1, 1, -1, -1]
    # Calculate the y-coordinates of the four corners
    y_coords = rect.center[2] .+ rect.size[2]/2 * [-1, -1, 1, 1, -1]
    
    # Plot the rectangle with a black fill
    plot!(p, Shape(x_coords, y_coords), 
          label = "", 
          color = :black, 
          alpha = 0.5,
          linewidth = 1.5,
          fillalpha = 0.3)
end

"""
    plot_environment!(p, env::Environment)

Add all obstacles from an environment to an existing plot.

# Arguments
- `p`: The plot to add the environment obstacles to
- `env`: Environment object containing obstacles

# Returns
- The updated plot object
"""
function plot_environment!(p, env::Environment)
    for obstacle in env.obstacles
        plot_rectangle!(p, obstacle)
    end
    return p
end

"""
    plot_environment(env::Environment; x_limits=(-20, 20), y_limits=(-20, 20), plot_size=(800, 400))

Create a new plot showing the environment.

# Arguments
- `env`: Environment object containing obstacles
- `x_limits`: Tuple with x-axis limits (default: (-20, 20))
- `y_limits`: Tuple with y-axis limits (default: (-20, 20))
- `plot_size`: Tuple with plot width and height (default: (800, 400))

# Returns
- A new plot showing the environment
"""
function plot_environment(env::Environment; x_limits=(-20, 20), y_limits=(-20, 20), plot_size=(800, 400))
    p = Plots.plot(size = plot_size, xlims = x_limits, ylims = y_limits, aspect_ratio = :equal)
    plot_environment!(p, env)
    return p
end

end # module 