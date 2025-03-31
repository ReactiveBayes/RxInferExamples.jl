#!/usr/bin/env julia

# RSLDS Visualization Script
# This script can be used to visualize saved RSLDS results
# Usage: julia visualize_rslds_results.jl <results_directory>

# Install required packages if not already installed
import Pkg
for pkg in ["Plots", "DelimitedFiles", "LinearAlgebra", "Statistics", "Dates", "ArgParse"]
    try
        @eval import $pkg
    catch
        println("Installing $pkg...")
        Pkg.add(pkg)
    end
end

using Plots, DelimitedFiles, LinearAlgebra, Statistics, Dates, ArgParse

function parse_commandline()
    s = ArgParseSettings(
        description = "Visualization tool for RSLDS results",
        prog = "visualize_rslds_results.jl"
    )
    
    @add_arg_table s begin
        "results_dir"
            help = "Directory containing RSLDS results data files"
            required = true
        "--output", "-o"
            help = "Output directory for visualizations (default: results_dir/visualizations)"
            default = ""
        "--format", "-f"
            help = "Output format for plots (png, pdf, svg)"
            default = "png"
        "--dpi"
            help = "DPI for output images"
            arg_type = Int
            default = 300
        "--no-lens", "-n"
            help = "Disable lens/zoom effect on plots"
            action = :store_true
        "--animate", "-a"
            help = "Create animations to visualize the system"
            action = :store_true
        "--fps"
            help = "Frames per second for animations"
            arg_type = Int
            default = 10
        "--inference-animation", "-i"
            help = "Create animation showing inference convergence (requires iteration_* subdirectories)"
            action = :store_true
    end
    
    return parse_args(s)
end

"""
    read_model_info(results_dir)
    
Extract model information from README.txt file if it exists.
"""
function read_model_info(results_dir)
    readme_path = joinpath(results_dir, "README.txt")
    
    model_info = Dict(
        "timestamp" => "Unknown",
        "switching_states" => "Unknown",
        "latent_dim" => "Unknown",
        "obs_dim" => "Unknown"
    )
    
    if isfile(readme_path)
        readme_content = read(readme_path, String)
        
        # Parse timestamp
        timestamp_match = match(r"Generated on: (.*)", readme_content)
        if timestamp_match !== nothing
            model_info["timestamp"] = timestamp_match.captures[1]
        end
        
        # Parse switching states
        states_match = match(r"Number of switching states: (\d+)", readme_content)
        if states_match !== nothing
            model_info["switching_states"] = states_match.captures[1]
        end
        
        # Parse latent dimension
        latent_match = match(r"Latent state dimension: (\d+)", readme_content)
        if latent_match !== nothing
            model_info["latent_dim"] = latent_match.captures[1]
        end
        
        # Parse observation dimension
        obs_match = match(r"Observation dimension: (\d+)", readme_content)
        if obs_match !== nothing
            model_info["obs_dim"] = obs_match.captures[1]
        end
    end
    
    return model_info
end

function create_visualizations(results_dir, output_dir, format="png", dpi=300; use_lens=true, create_animations=false, fps=10, inference_animation=false)
    # Create output directory if it doesn't exist
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    println("Loading RSLDS results from: $results_dir")
    
    # Read model information if available
    model_info = read_model_info(results_dir)
    println("Model information:")
    println("- Generated on: $(model_info["timestamp"])")
    println("- Switching states: $(model_info["switching_states"])")
    println("- Latent dimension: $(model_info["latent_dim"])")
    println("- Observation dimension: $(model_info["obs_dim"])")
    
    # Load data files
    switching_states_file = joinpath(results_dir, "switching_states.csv")
    continuous_mean_file = joinpath(results_dir, "continuous_states_mean.csv")
    continuous_var_file = joinpath(results_dir, "continuous_states_var.csv")
    observations_file = joinpath(results_dir, "observations.csv")
    
    # Check if files exist
    all_required_files_exist = isfile(switching_states_file) && 
                              isfile(continuous_mean_file) && 
                              isfile(continuous_var_file) && 
                              isfile(observations_file)
    
    if !all_required_files_exist
        error("Required data files not found in $results_dir")
    end
    
    # Load data
    switching_states = vec(readdlm(switching_states_file))
    continuous_states_mean = readdlm(continuous_mean_file)
    continuous_states_var = readdlm(continuous_var_file)
    observations = readdlm(observations_file)
    
    # Check if true states are available
    true_states_file = joinpath(results_dir, "true_states.csv")
    has_true_states = isfile(true_states_file)
    
    true_states = has_true_states ? readdlm(true_states_file) : nothing
    
    # Load transition matrices if available
    transition_matrices = []
    i = 1
    while isfile(joinpath(results_dir, "transition_matrix_$(i).csv"))
        push!(transition_matrices, readdlm(joinpath(results_dir, "transition_matrix_$(i).csv")))
        i += 1
    end
    
    # Load discrete transition matrix if available
    discrete_matrix_file = joinpath(results_dir, "discrete_transition_matrix.csv")
    has_discrete_matrix = isfile(discrete_matrix_file)
    
    discrete_matrix = has_discrete_matrix ? readdlm(discrete_matrix_file) : nothing
    
    # Load free energy if available
    free_energy_file = joinpath(results_dir, "free_energy.csv")
    has_free_energy = isfile(free_energy_file)
    
    free_energy = has_free_energy ? vec(readdlm(free_energy_file)) : nothing
    
    # Get dimensions
    n_dims = size(continuous_states_mean, 2)
    n_timepoints = size(continuous_states_mean, 1)
    
    println("Data loaded: $n_timepoints time points, $n_dims dimensions")
    
    theme(:default)
    default_font = "Computer Modern"
    default(fontfamily=default_font, 
            linewidth=2, 
            framestyle=:box, 
            label=nothing, 
            grid=false,
            legendfontsize=10)
    
    # Create visualizations
    
    # 1. Switching state visualization
    p1 = scatter(switching_states, 
                label="Estimated Regimes", 
                color="blue", 
                linewidth=2,
                xlabel="Time", 
                ylabel="Regime", 
                title="Estimated Switching States")
    savefig(p1, joinpath(output_dir, "switching_states.$format"))
    
    # 2. Continuous state visualization for each dimension
    for dim in 1:n_dims
        p = plot(continuous_states_mean[:, dim], 
                ribbon=sqrt.(continuous_states_var[:, dim]), 
                label="Estimated States", 
                color="blue", 
                fillalpha=0.2, 
                linewidth=2,
                xlabel="Time", 
                ylabel="State Value", 
                title="Dimension $dim State Estimation")
        
        if has_true_states
            plot!(p, true_states[:, dim], 
                 label="True States", 
                 color="green", 
                 linewidth=1)
        end
        
        scatter!(p, observations[:, dim], 
                label="Observed Data", 
                color="black", 
                ms=1.3)
        
        # Add a lens to highlight a section of the data if enabled
        if use_lens
            lens_start = max(1, div(n_timepoints, 4))
            lens_end = min(n_timepoints, lens_start + 40)
            lens!(p, lens_start:lens_end, [-3, 3], 
                 inset=(1, bbox(0.07, 0.6, 0.3, 0.3)))
        end
        
        savefig(p, joinpath(output_dir, "continuous_state_dim$(dim).$format"))
    end
    
    # 3. Transition matrices visualization
    for (i, matrix) in enumerate(transition_matrices)
        p = heatmap(matrix, 
                   title="Continuous Transition Matrix $i",
                   xlabel="To", 
                   ylabel="From", 
                   color=:viridis, 
                   aspect_ratio=1,
                   clim=(-1, 1),  # Adjust color limits as needed
                   annotate=[(j, i, text(round(matrix[i,j], digits=2), 8, :white)) 
                            for i in 1:size(matrix, 1) for j in 1:size(matrix, 2)])
        savefig(p, joinpath(output_dir, "transition_matrix_$(i).$format"))
    end
    
    # 4. Discrete transition matrix visualization
    if has_discrete_matrix
        p = heatmap(discrete_matrix, 
                   title="Discrete Transition Matrix",
                   xlabel="To", 
                   ylabel="From", 
                   color=:viridis, 
                   aspect_ratio=1,
                   annotate=[(j, i, text(round(discrete_matrix[i,j], digits=3), 8, :white)) 
                            for i in 1:size(discrete_matrix, 1) for j in 1:size(discrete_matrix, 2)])
        savefig(p, joinpath(output_dir, "discrete_transition_matrix.$format"))
    end
    
    # 5. Free energy convergence
    if has_free_energy
        p = plot(free_energy, 
                label="", 
                title="Free Energy Convergence",
                xlabel="Iteration", 
                ylabel="Free Energy",
                marker=:circle, 
                markersize=3, 
                linewidth=2)
        savefig(p, joinpath(output_dir, "free_energy.$format"))
    end
    
    # 6. Combined view of states and observations (first 200 points)
    viz_length = min(n_timepoints, 200)
    p = plot(layout=(n_dims+1, 1), size=(800, 200*(n_dims+1)))
    
    # First plot is switching states
    plot!(p[1], switching_states[1:viz_length], 
          color="blue", 
          linewidth=2, 
          label="", 
          title="Switching States", 
          ylabel="Regime")
    
    # Remaining plots are continuous states for each dimension
    for dim in 1:n_dims
        plot!(p[dim+1], continuous_states_mean[1:viz_length, dim], 
              ribbon=sqrt.(continuous_states_var[1:viz_length, dim]), 
              label="Estimated", 
              color="blue", 
              fillalpha=0.2, 
              linewidth=2,
              title="Dimension $dim", 
              ylabel="Value")
        
        if has_true_states
            plot!(p[dim+1], true_states[1:viz_length, dim], 
                 label="True", 
                 color="green", 
                 linewidth=1)
        end
        
        scatter!(p[dim+1], observations[1:viz_length, dim], 
                label="Observed", 
                color="black", 
                ms=1.0)
    end
    
    # Only show x-axis label on the bottom plot
    plot!(p[n_dims+1], xlabel="Time")
    
    savefig(p, joinpath(output_dir, "combined_view.$format"))
    
    # 7. Generate a combined dashboard
    plot_width = 500
    plot_height = 400
    
    # Create a dashboard with 2x2 layout
    dashboard = plot(layout=(2,2), size=(2*plot_width, 2*plot_height), 
                   margin=5Plots.mm)
    
    # Switching state plot
    plot!(dashboard[1], switching_states, 
          color="blue", 
          linewidth=2,
          xlabel="Time", 
          ylabel="Regime", 
          title="Switching States")
    
    # Continuous states (first dimension)
    plot!(dashboard[2], continuous_states_mean[:, 1], 
          ribbon=sqrt.(continuous_states_var[:, 1]), 
          label="Estimated", 
          color="blue", 
          fillalpha=0.2, 
          linewidth=2,
          xlabel="Time", 
          ylabel="Value", 
          title="Dimension 1 State")
    
    if has_true_states
        plot!(dashboard[2], true_states[:, 1], 
             label="True", 
             color="green", 
             linewidth=1)
    end
    
    scatter!(dashboard[2], observations[:, 1], 
            label="Observed", 
            color="black", 
            ms=0.8)
    
    # Transition matrices
    if length(transition_matrices) > 0
        heatmap!(dashboard[3], transition_matrices[1], 
                title="Transition Matrix 1",
                xlabel="To", 
                ylabel="From", 
                color=:viridis, 
                aspect_ratio=1,
                clim=(-1, 1),
                annotate=[(j, i, text(round(transition_matrices[1][i,j], digits=2), 7, :white)) 
                        for i in 1:size(transition_matrices[1], 1) for j in 1:size(transition_matrices[1], 2)])
    end
    
    # Discrete transition matrix
    if has_discrete_matrix
        heatmap!(dashboard[4], discrete_matrix, 
                title="Discrete Transition Matrix",
                xlabel="To", 
                ylabel="From", 
                color=:viridis, 
                aspect_ratio=1,
                annotate=[(j, i, text(round(discrete_matrix[i,j], digits=2), 7, :white)) 
                        for i in 1:size(discrete_matrix, 1) for j in 1:size(discrete_matrix, 2)])
    elseif has_free_energy
        plot!(dashboard[4], free_energy, 
             label="", 
             title="Free Energy",
             xlabel="Iteration", 
             ylabel="Value",
             marker=:circle, 
             markersize=3, 
             linewidth=2)
    end
    
    savefig(dashboard, joinpath(output_dir, "dashboard.$format"))
    
    # 8. Generate an HTML report if requested
    html_report_path = joinpath(output_dir, "report.html")
    open(html_report_path, "w") do io
        println(io, """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RSLDS Analysis Results</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
                h1, h2, h3 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                .report-header { margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #eee; }
                .report-section { margin-bottom: 30px; }
                .plot-gallery { display: flex; flex-wrap: wrap; gap: 20px; }
                .plot-item { flex: 1; min-width: 300px; margin-bottom: 20px; }
                .plot-item img { width: 100%; border: 1px solid #ddd; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="report-header">
                    <h1>RSLDS Analysis Results</h1>
                    <p>Generated on: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))</p>
                    <p>Original analysis: $(model_info["timestamp"])</p>
                </div>
                
                <div class="report-section">
                    <h2>Model Information</h2>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Number of switching states</td><td>$(model_info["switching_states"])</td></tr>
                        <tr><td>Latent state dimension</td><td>$(model_info["latent_dim"])</td></tr>
                        <tr><td>Observation dimension</td><td>$(model_info["obs_dim"])</td></tr>
                        <tr><td>Time points</td><td>$n_timepoints</td></tr>
                    </table>
                </div>
                
                <div class="report-section">
                    <h2>Dashboard</h2>
                    <img src="dashboard.$format" alt="Dashboard" style="width:100%; max-width:1000px;">
                </div>
                
                <div class="report-section">
                    <h2>Switching States</h2>
                    <div class="plot-item">
                        <img src="switching_states.$format" alt="Switching States">
                    </div>
                    <p>This plot shows the estimated regime switches over time.</p>
                </div>
                
                <div class="report-section">
                    <h2>Continuous States</h2>
                    <div class="plot-gallery">
        """)
        
        for dim in 1:n_dims
            println(io, """
                <div class="plot-item">
                    <img src="continuous_state_dim$(dim).$format" alt="Dimension $dim">
                    <p>Dimension $dim state estimation with uncertainty</p>
                </div>
            """)
        end
        
        println(io, """
                    </div>
                </div>
                
                <div class="report-section">
                    <h2>Transition Matrices</h2>
                    <div class="plot-gallery">
        """)
        
        for i in 1:length(transition_matrices)
            println(io, """
                <div class="plot-item">
                    <img src="transition_matrix_$(i).$format" alt="Transition Matrix $i">
                    <p>Continuous transition matrix $i</p>
                </div>
            """)
        end
        
        if has_discrete_matrix
            println(io, """
                <div class="plot-item">
                    <img src="discrete_transition_matrix.$format" alt="Discrete Transition Matrix">
                    <p>Discrete transition matrix for the switching process</p>
                </div>
            """)
        end
        
        println(io, """
                    </div>
                </div>
        """)
        
        if has_free_energy
            println(io, """
                <div class="report-section">
                    <h2>Inference Convergence</h2>
                    <div class="plot-item">
                        <img src="free_energy.$format" alt="Free Energy">
                        <p>Free energy convergence during inference</p>
                    </div>
                </div>
            """)
        end
        
        println(io, """
                <div class="report-section">
                    <h2>Combined View</h2>
                    <div class="plot-item">
                        <img src="combined_view.$format" alt="Combined View">
                        <p>Combined view of all states and observations</p>
                    </div>
                </div>
            </div>
        """)
        
        if create_animations
            println(io, """
                <div class="report-section">
                    <h2>Animations</h2>
                    <div class="plot-gallery">
                        <div class="plot-item">
                            <img src="animation_states.gif" alt="States Animation">
                            <p>Animation of the switching states and continuous states over time</p>
                        </div>
                        <div class="plot-item">
                            <img src="animation_phase.gif" alt="Phase Space Animation">
                            <p>Animation of the system trajectory in phase space</p>
                        </div>
                    </div>
                </div>
            """)
        end
        
        println(io, """
                </div>
            </body>
        </html>
        """)
    end
    
    # 9. Create animations if requested
    if create_animations
        try
            create_rslds_animations(switching_states, continuous_states_mean, continuous_states_var, 
                                   observations, true_states, output_dir, fps)
        catch e
            println("Error creating animations: $e")
            println("Continuing with other visualizations...")
        end
    end
    
    # 10. Create inference animation if requested
    if inference_animation
        println("Looking for iteration snapshots...")
        iteration_dirs = filter(d -> startswith(d, "iteration_") && isdir(joinpath(results_dir, d)), 
                              readdir(results_dir))
        
        if !isempty(iteration_dirs)
            try
                create_inference_animation(results_dir, iteration_dirs, output_dir, fps)
            catch e
                println("Error creating inference animation: $e")
                println("Continuing with other visualizations...")
            end
        else
            println("No iteration snapshots found. Skipping inference animation.")
        end
    end
    
    println("All visualizations saved to $output_dir")
    println("HTML report generated at: $html_report_path")
end

"""
    create_rslds_animations(switching_states, continuous_states_mean, continuous_states_var, 
                           observations, true_states, output_dir, fps)
    
Creates animations to visualize the RSLDS model dynamics.

# Arguments
- `switching_states`: Vector of switching states
- `continuous_states_mean`: Matrix of continuous state means
- `continuous_states_var`: Matrix of continuous state variances
- `observations`: Matrix of observations
- `true_states`: Matrix of true states (if available, can be nothing)
- `output_dir`: Directory to save animations
- `fps`: Frames per second for animations
"""
function create_rslds_animations(switching_states, continuous_states_mean, continuous_states_var, 
                               observations, true_states, output_dir, fps)
    println("Creating animations...")
    
    n_dims = size(continuous_states_mean, 2)
    n_frames = min(size(continuous_states_mean, 1), 500)  # Limit to 500 frames max
    
    # 1. Create animation of states over time
    println("Creating state animation...")
    anim_states = @animate for t in 1:n_frames
        # Use a layout with switching states on top and continuous states below
        p = plot(layout=(n_dims+1, 1), size=(800, 200*(n_dims+1)))
        
        # Top plot: Switching states with current position highlighted
        plot!(p[1], switching_states[1:t], color="blue", linewidth=2, 
              label="", title="Switching States", ylabel="Regime")
        scatter!(p[1], [t], [switching_states[t]], color="red", markersize=5, label="Current")
        
        # Continuous states for each dimension
        for dim in 1:n_dims
            # Plot state evolution
            plot!(p[dim+1], continuous_states_mean[1:t, dim], 
                  ribbon=sqrt.(continuous_states_var[1:t, dim]), 
                  color="blue", fillalpha=0.2, linewidth=2,
                  title="Dimension $dim", ylabel="Value", label="Estimated")
            
            # Highlight current position
            scatter!(p[dim+1], [t], [continuous_states_mean[t, dim]], 
                    color="red", markersize=5, label="Current")
            
            # Plot true states if available
            if true_states !== nothing
                plot!(p[dim+1], true_states[1:t, dim], color="green", linewidth=1, label="True")
            end
            
            # Plot observations
            scatter!(p[dim+1], 1:t, observations[1:t, dim], 
                    color="black", markersize=1.5, label="Observed")
        end
        
        # Only show x-axis label on the bottom plot
        plot!(p[n_dims+1], xlabel="Time")
    end
    
    try
        gif(anim_states, joinpath(output_dir, "animation_states.gif"), fps=fps)
        println("States animation created successfully.")
    catch e
        println("Error creating states animation: $e")
    end
    
    # 2. Create phase space animation (for 2D systems)
    if n_dims >= 2
        println("Creating phase space animation...")
        anim_phase = @animate for t in 1:n_frames
            # Main plot: Phase space
            p = plot(size=(800, 600), title="Phase Space Trajectory")
            
            # Plot trajectory so far
            plot!(continuous_states_mean[1:t, 1], continuous_states_mean[1:t, 2], 
                  color="blue", linewidth=2, label="Estimated")
            
            # Highlight the current state with color based on switching state
            state_colors = [:red, :orange, :green, :purple, :brown]
            current_color = state_colors[mod1(Int(switching_states[t]), length(state_colors))]
            
            scatter!([continuous_states_mean[t, 1]], [continuous_states_mean[t, 2]], 
                    color=current_color, markersize=8, label="Current (Regime $(Int(switching_states[t])))")
            
            # Plot true trajectory if available
            if true_states !== nothing
                plot!(true_states[1:t, 1], true_states[1:t, 2], 
                     color="green", linewidth=1, label="True")
            end
            
            # Plot observations
            scatter!(observations[1:t, 1], observations[1:t, 2], 
                    color="black", markersize=1.5, label="Observed")
            
            xlabel!("Dimension 1")
            ylabel!("Dimension 2")
        end
        
        try
            gif(anim_phase, joinpath(output_dir, "animation_phase.gif"), fps=fps)
            println("Phase space animation created successfully.")
        catch e
            println("Error creating phase space animation: $e")
        end
    end
end

"""
    create_inference_animation(results_dir, iteration_dirs, output_dir, fps)
    
Creates an animation showing the convergence of the inference over iterations.

# Arguments
- `results_dir`: Base directory containing results
- `iteration_dirs`: Array of iteration snapshot directories
- `output_dir`: Directory to save the animation
- `fps`: Frames per second for the animation
"""
function create_inference_animation(results_dir, iteration_dirs, output_dir, fps)
    println("Creating inference convergence animation...")
    
    # Sort iteration directories by iteration number
    sort!(iteration_dirs, by=dir -> parse(Int, replace(dir, "iteration_" => "")))
    
    # Extract data from each iteration
    iteration_data = []
    
    for dir in iteration_dirs
        # Load data for this iteration
        iter_path = joinpath(results_dir, dir)
        
        # Check if required files exist
        files_exist = isfile(joinpath(iter_path, "switching_states.csv")) &&
                     isfile(joinpath(iter_path, "continuous_states_mean.csv"))
        
        if files_exist
            try
                # Read switching states and continuous states
                switching_states = vec(readdlm(joinpath(iter_path, "switching_states.csv")))
                continuous_mean = readdlm(joinpath(iter_path, "continuous_states_mean.csv"))
                
                # Extract iteration number
                iter_num = parse(Int, replace(dir, "iteration_" => ""))
                
                push!(iteration_data, (iter_num, switching_states, continuous_mean))
            catch e
                println("Error loading data from $dir: $e")
            end
        end
    end
    
    # Create animation if we have data
    if isempty(iteration_data)
        println("No valid data found in iteration directories.")
        return
    end
    
    # Sort by iteration number
    sort!(iteration_data, by=item -> item[1])
    
    # Extract time points and dimensions
    n_timepoints = size(iteration_data[1][3], 1)
    n_dims = size(iteration_data[1][3], 2)
    
    # Create animation
    println("Creating inference animation with $(length(iteration_data)) frames...")
    
    try
        anim_inference = @animate for (i, (iter_num, states, means)) in enumerate(iteration_data)
            p = plot(layout=(n_dims+1, 1), size=(800, 200*(n_dims+1)),
                    title="Inference Progress: Iteration $iter_num")
            
            # Switching states
            plot!(p[1], states, color="blue", linewidth=2, 
                  title="Switching States", ylabel="Regime", label="")
            
            # Continuous states for each dimension
            for dim in 1:n_dims
                plot!(p[dim+1], means[:, dim], color="blue", linewidth=2,
                      title="Dimension $dim", ylabel="Value", label="")
            end
            
            # Only show x-axis label on the bottom plot
            plot!(p[n_dims+1], xlabel="Time")
        end
        
        gif(anim_inference, joinpath(output_dir, "inference_convergence.gif"), fps=fps)
        println("Inference convergence animation created.")
        
        # Add to HTML report
        html_report_path = joinpath(output_dir, "report.html")
        if isfile(html_report_path)
            try
                # Read the existing HTML
                content = read(html_report_path, String)
                
                # Find the insertion point (right before the closing div.container)
                insertion_point = findlast("</div>\n</body>", content)
                
                if insertion_point !== nothing
                    # Split the content
                    before = content[1:first(insertion_point)-1]
                    after = content[first(insertion_point):end]
                    
                    # Insert the new section
                    new_section = """
                    <div class="report-section">
                        <h2>Inference Convergence</h2>
                        <div class="plot-item">
                            <img src="inference_convergence.gif" alt="Inference Convergence">
                            <p>Animation showing how the model estimates evolved during inference iterations</p>
                        </div>
                    </div>
                    """
                    
                    # Write the updated HTML
                    open(html_report_path, "w") do io
                        write(io, before * new_section * after)
                    end
                    
                    println("Added inference animation to HTML report.")
                end
            catch e
                println("Error updating HTML report: $e")
            end
        end
    catch e
        println("Error creating inference animation: $e")
    end
end

function main()
    args = parse_commandline()
    
    results_dir = args["results_dir"]
    output_dir = args["output"] == "" ? joinpath(results_dir, "visualizations") : args["output"]
    format = args["format"]
    dpi = args["dpi"]
    use_lens = !args["no-lens"]
    create_animations = args["animate"]
    fps = args["fps"]
    inference_animation = args["inference-animation"]
    
    create_visualizations(results_dir, output_dir, format, dpi; 
                         use_lens=use_lens, 
                         create_animations=create_animations,
                         fps=fps,
                         inference_animation=inference_animation)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# Usage:
# julia visualize_rslds_results.jl path/to/results_directory
# julia visualize_rslds_results.jl path/to/results_directory --animate --fps 15 