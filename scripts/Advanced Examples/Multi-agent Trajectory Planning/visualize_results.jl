using Plots
using DelimitedFiles
using Dates
using Statistics
using LinearAlgebra

# Add the current directory to the load path
if !isinteractive()
    push!(LOAD_PATH, dirname(@__FILE__))
end

# Import TrajectoryPlanning module
include("TrajectoryPlanning.jl")
using .TrajectoryPlanning

"""
    visualize_results(results_dir)

Generate visualizations from results stored in the specified directory.

# Arguments
- `results_dir`: Directory containing result files (paths.csv, controls.csv, etc.)

# Returns
- Nothing. Visualizations are saved to the results directory.
"""
function visualize_results(results_dir)
    println("Visualizing results from $results_dir...")
    
    # Check if directory exists
    if !isdir(results_dir)
        error("Results directory not found: $results_dir")
    end
    
    # Load path data if available
    paths_file = joinpath(results_dir, "paths.csv")
    if isfile(paths_file)
        path_data = readdlm(paths_file, ',', Float64)
        println("Loaded path data: $(size(path_data, 1)) points")
        
        # Process path data
        nr_agents = Int(maximum(path_data[:, 1]))
        nr_steps = Int(maximum(path_data[:, 2]))
        
        # Reshape data into matrices
        paths = Array{Vector{Float64}}(undef, nr_agents, nr_steps)
        for i in 1:size(path_data, 1)
            agent = Int(path_data[i, 1])
            step = Int(path_data[i, 2])
            paths[agent, step] = [path_data[i, 3], path_data[i, 4]]
        end
        
        # Plot paths
        p = plot(title="Agent Paths", xlabel="X", ylabel="Y", aspect_ratio=:equal,
                 legend=:topleft, size=(800, 600))
        
        colors = Plots.palette(:tab10)
        for k in 1:nr_agents
            x_coords = [paths[k, t][1] for t in 1:nr_steps]
            y_coords = [paths[k, t][2] for t in 1:nr_steps]
            
            plot!(p, x_coords, y_coords, linewidth=2, 
                  marker=:circle, markersize=3, 
                  label="Agent $k", color=colors[k])
            
            # Mark start and end points
            scatter!([x_coords[1]], [y_coords[1]], color=colors[k], 
                     marker=:star, markersize=8, label="")
            scatter!([x_coords[end]], [y_coords[end]], color=colors[k], 
                     marker=:square, markersize=8, label="")
        end
        
        savefig(p, joinpath(results_dir, "path_visualization.png"))
        println("Saved path visualization.")
    else
        println("Warning: Path data file not found.")
    end
    
    # Load control data if available
    controls_file = joinpath(results_dir, "controls.csv")
    if isfile(controls_file)
        control_data = readdlm(controls_file, ',', Float64)
        println("Loaded control data: $(size(control_data, 1)) points")
        
        # Process control data
        nr_agents = Int(maximum(control_data[:, 1]))
        nr_steps = Int(maximum(control_data[:, 2]))
        
        # Reshape data into matrices
        controls = Array{Vector{Float64}}(undef, nr_agents, nr_steps)
        for i in 1:size(control_data, 1)
            agent = Int(control_data[i, 1])
            step = Int(control_data[i, 2])
            controls[agent, step] = [control_data[i, 3], control_data[i, 4]]
        end
        
        # Plot control magnitudes
        p = plot(title="Control Magnitudes", xlabel="Step", ylabel="Magnitude", 
                 legend=:topleft, size=(800, 400))
        
        colors = Plots.palette(:tab10)
        for k in 1:nr_agents
            magnitudes = [norm(controls[k, t]) for t in 1:nr_steps]
            plot!(p, 1:nr_steps, magnitudes, linewidth=2, 
                  marker=:circle, markersize=3, 
                  label="Agent $k", color=colors[k])
        end
        
        savefig(p, joinpath(results_dir, "control_magnitudes.png"))
        println("Saved control magnitude visualization.")
    else
        println("Warning: Control data file not found.")
    end
    
    # Load convergence data if available
    convergence_file = joinpath(results_dir, "convergence_metrics.csv")
    if isfile(convergence_file)
        convergence_data = readdlm(convergence_file, ',', Float64)
        println("Loaded convergence data: $(size(convergence_data, 1)) points")
        
        iterations = convergence_data[:, 1]
        elbo_values = convergence_data[:, 2]
        
        # Plot convergence
        p = plot(iterations, elbo_values, 
                 title="ELBO Convergence", xlabel="Iteration", ylabel="ELBO",
                 legend=false, linewidth=2, size=(800, 400))
        
        savefig(p, joinpath(results_dir, "elbo_convergence.png"))
        println("Saved ELBO convergence visualization.")
    else
        println("Warning: Convergence metrics file not found.")
    end
    
    println("Visualization complete.")
    return nothing
end

"""
    visualize_environment(environment_name, output_dir=".")

Generate a visualization of the specified environment.

# Arguments
- `environment_name`: Name of the environment ("door", "wall", or "combined")
- `output_dir`: Directory to save the visualization (default: current directory)

# Returns
- Nothing. Visualization is saved to the output directory.
"""
function visualize_environment(environment_name, output_dir=".")
    println("Visualizing $environment_name environment...")
    
    # Create the environment based on name
    environment = if environment_name == "door"
        TrajectoryPlanning.create_door_environment()
    elseif environment_name == "wall"
        TrajectoryPlanning.create_wall_environment()
    elseif environment_name == "combined"
        TrajectoryPlanning.create_combined_environment()
    else
        error("Unknown environment: $environment_name")
    end
    
    # Create a plot with the environment
    p = plot(title="$environment_name Environment", 
             xlims=(-20, 20), ylims=(-20, 20), 
             aspect_ratio=:equal, size=(800, 600))
    
    # Plot each obstacle
    for obstacle in environment.obstacles
        # Calculate the x-coordinates of the four corners
        x_coords = obstacle.center[1] .+ obstacle.size[1]/2 * [-1, 1, 1, -1, -1]
        # Calculate the y-coordinates of the four corners
        y_coords = obstacle.center[2] .+ obstacle.size[2]/2 * [-1, -1, 1, 1, -1]
        
        # Plot the rectangle with a black fill
        plot!(p, Shape(x_coords, y_coords), 
              label = "", 
              color = :black, 
              alpha = 0.5,
              linewidth = 1.5,
              fillalpha = 0.3)
    end
    
    # Save the plot
    filename = joinpath(output_dir, "$(environment_name)_environment.png")
    savefig(p, filename)
    println("Saved environment visualization to $filename")
    
    return nothing
end

"""
    compare_environments(output_dir=".")

Generate a comparison visualization of all standard environments.

# Arguments
- `output_dir`: Directory to save the visualization (default: current directory)

# Returns
- Nothing. Visualization is saved to the output directory.
"""
function compare_environments(output_dir=".")
    println("Comparing all environments...")
    
    # Create environments
    door_env = TrajectoryPlanning.create_door_environment()
    wall_env = TrajectoryPlanning.create_wall_environment()
    combined_env = TrajectoryPlanning.create_combined_environment()
    
    # Create a plot layout with 3 subplots
    p = plot(layout=(1,3), size=(1200, 400), title=["Door Environment" "Wall Environment" "Combined Environment"])
    
    # Helper function to plot an environment in a subplot
    function plot_env_in_subplot!(p, env, idx)
        plot!(p, subplot=idx, xlims=(-20, 20), ylims=(-20, 20), aspect_ratio=:equal)
        
        for obstacle in env.obstacles
            # Calculate the x-coordinates of the four corners
            x_coords = obstacle.center[1] .+ obstacle.size[1]/2 * [-1, 1, 1, -1, -1]
            # Calculate the y-coordinates of the four corners
            y_coords = obstacle.center[2] .+ obstacle.size[2]/2 * [-1, -1, 1, 1, -1]
            
            # Plot the rectangle with a black fill
            plot!(p, subplot=idx, Shape(x_coords, y_coords), 
                  label = "", 
                  color = :black, 
                  alpha = 0.5,
                  linewidth = 1.5,
                  fillalpha = 0.3)
        end
    end
    
    # Plot each environment in its subplot
    plot_env_in_subplot!(p, door_env, 1)
    plot_env_in_subplot!(p, wall_env, 2)
    plot_env_in_subplot!(p, combined_env, 3)
    
    # Save the plot
    filename = joinpath(output_dir, "environment_comparison.png")
    savefig(p, filename)
    println("Saved environment comparison to $filename")
    
    return nothing
end

"""
    visualize_latest_results()

Visualize results from the most recent experiment.

# Returns
- Nothing. Visualizations are saved to the results directory.
"""
function visualize_latest_results()
    # Try to find results directory
    if isdir("results")
        subdirs = filter(d -> isdir(joinpath("results", d)), readdir("results"))
        if !isempty(subdirs)
            # Sort by directory name (assumes timestamp format)
            sort!(subdirs, rev=true)
            latest_dir = joinpath("results", subdirs[1])
            
            println("Found latest results directory: $latest_dir")
            visualize_results(latest_dir)
        else
            println("No result directories found.")
        end
    else
        println("Results directory not found.")
    end
    
    return nothing
end

# If run as a script, visualize the latest results
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) > 0
        # If a directory is provided, visualize results from that directory
        visualize_results(ARGS[1])
    else
        # Otherwise, visualize the latest results
        visualize_latest_results()
    end
end 