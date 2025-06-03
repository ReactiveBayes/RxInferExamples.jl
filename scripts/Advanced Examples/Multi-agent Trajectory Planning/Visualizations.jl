module Visualizations

using Plots
using ..Environment

export plot_marker_at_position!, plot_agent_naive_plan!, animate_paths

function plot_marker_at_position!(p, radius, position; color="red", markersize=10.0, alpha=1.0, label="")
    # Draw the agent as a circle with the given radius
    θ = range(0, 2π, 100)
    
    x_coords = position[1] .+ radius .* cos.(θ)
    y_coords = position[2] .+ radius .* sin.(θ)
    
    plot!(p, Shape(x_coords, y_coords); color=color, label=label, alpha=alpha)
    return p
end

function plot_agent_naive_plan!(p, agent; color = "blue")
    plot_marker_at_position!(p, agent.radius, agent.initial_position, color = color)
    plot_marker_at_position!(p, agent.radius, agent.target_position, color = color, alpha = 0.1)
    quiver!(p, [ agent.initial_position[1] ], [ agent.initial_position[2] ], 
            quiver = ([ agent.target_position[1] - agent.initial_position[1] ], 
                     [ agent.target_position[2] -  agent.initial_position[2] ]))
end

function plot_agent_plans!(p, agents)
    colors = Plots.palette(:tab10)
    
    for (k, agent) in enumerate(agents)
        plot_agent_naive_plan!(p, agent, color = colors[k])
    end
    
    return p
end

function animate_paths(environment, agents, paths; filename = "result.gif", fps = 15)
    nr_agents, nr_steps = size(paths)
    colors = Plots.palette(:tab10)

    println("Generating animation frames...")
    animation = @animate for t in 1:nr_steps
        frame = plot_environment(environment)
    
        for k in 1:nr_agents
            position = paths[k, t]          
            path = paths[k, 1:t]
            
            plot_marker_at_position!(frame, agents[k].radius, position, color = colors[k])
            plot_marker_at_position!(frame, agents[k].radius, agents[k].target_position, color = colors[k], alpha = 0.2)
            plot!(frame, getindex.(path, 1), getindex.(path, 2); linestyle=:dash, label="", color=colors[k])
        end

        frame
    end

    # Save the animation
    println("Saving animation to $filename...")
    gif(animation, filename, fps=fps, show_msg = false)
    println("Animation saved successfully.")
    
    return nothing
end

end # module 