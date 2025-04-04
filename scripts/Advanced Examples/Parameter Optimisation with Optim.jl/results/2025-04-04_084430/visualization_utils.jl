#!/usr/bin/env julia
# Visualization utilities for Parameter Optimisation with Optim.jl

using Plots, Dates, StatsPlots, Distributions, LaTeXStrings, Measures

"""
    plot_univariate_data(signal, data, v)

Plot the univariate data and the true signal.
"""
function plot_univariate_data(signal, data, v; output_dir=nothing)
    p = plot(title="Univariate State Space Model Data", 
             size=(800, 500), dpi=300, margin=5mm)
    plot!(p, 1:length(signal), signal, label="True signal", linewidth=2, 
          color=:blue, legend=:topleft)
    scatter!(p, 1:length(data), data, label="Observations", 
             alpha=0.5, markersize=3, color=:red)
    plot!(p, xlabel="Time index", ylabel="Value")
    
    if !isnothing(output_dir)
        savefig(p, joinpath(output_dir, "univariate_data.png"))
    end
    
    return p
end

"""
    plot_univariate_results(signal, data, x_posterior, v; output_dir=nothing)

Plot the univariate inference results.
"""
function plot_univariate_results(signal, data, x_posterior, v; output_dir=nothing)
    means = mean.(x_posterior)
    stds = sqrt.(var.(x_posterior))
    
    p = plot(title="Univariate State Space Model Results", 
             size=(800, 500), dpi=300, margin=5mm)
    
    # True signal
    plot!(p, 1:length(signal), signal, label="True signal", 
          linewidth=2, color=:blue, alpha=0.7, legend=:topleft)
    
    # Inferred signal with uncertainty
    plot!(p, 1:length(means), means, ribbon=stds, label="Inferred signal",
          linewidth=2, color=:green, fillalpha=0.3)
    
    # Data points
    scatter!(p, 1:length(data), data, label="Observations", 
             alpha=0.5, markersize=3, color=:red)
    
    plot!(p, xlabel="Time index", ylabel="Value")
    
    if !isnothing(output_dir)
        savefig(p, joinpath(output_dir, "univariate_results.png"))
    end
    
    return p
end

"""
    plot_multivariate_data(x, Q)

Plot the multivariate data.
"""
function plot_multivariate_data(x, Q; output_dir=nothing)
    px = plot(title="Multivariate State Space Model Data", 
              size=(800, 500), dpi=300, margin=5mm)
    px = plot!(px, getindex.(x, 1), ribbon = diag(Q)[1] .|> sqrt, 
               fillalpha = 0.2, label = "x₁", color=:blue, linewidth=2)
    px = plot!(px, getindex.(x, 2), ribbon = diag(Q)[2] .|> sqrt, 
               fillalpha = 0.2, label = "x₂", color=:red, linewidth=2)
    px = plot!(px, xlabel="Time index", ylabel="Value")
    
    # Also create a 2D phase plot
    p2 = scatter(getindex.(x, 1), getindex.(x, 2), 
                title="Phase Plot of State Variables", 
                xlabel="x₁", ylabel="x₂", 
                markersize=3, markerstrokewidth=0, 
                alpha=0.6, color=:blue,
                size=(600, 600), dpi=300, margin=5mm)
    
    # Add arrows to show direction
    step = max(1, div(length(x), 20))
    for i in 1:step:length(x)-step
        plot!(p2, [x[i][1], x[i+step][1]], [x[i][2], x[i+step][2]], 
              arrow=true, color=:black, linewidth=1, label=nothing)
    end
    
    if !isnothing(output_dir)
        savefig(px, joinpath(output_dir, "multivariate_data.png"))
        savefig(p2, joinpath(output_dir, "multivariate_phase.png"))
    end
    
    return px, p2
end

"""
    plot_multivariate_results(x, Q, xmarginals; output_dir=nothing)

Plot the multivariate model inference results.
"""
function plot_multivariate_results(x, Q, xmarginals; output_dir=nothing)
    px = plot(title="Multivariate Model Results", 
              size=(800, 500), dpi=300, margin=5mm)
    
    # True trajectories with uncertainty
    px = plot!(px, getindex.(x, 1), ribbon = diag(Q)[1] .|> sqrt, 
               fillalpha = 0.2, label = "True x₁", color=:blue, linewidth=2)
    px = plot!(px, getindex.(x, 2), ribbon = diag(Q)[2] .|> sqrt, 
               fillalpha = 0.2, label = "True x₂", color=:red, linewidth=2)
    
    # Inferred trajectories with uncertainty
    px = plot!(px, getindex.(mean.(xmarginals), 1), 
               ribbon = getindex.(var.(xmarginals), 1) .|> sqrt, 
               fillalpha = 0.5, label = "Inferred x₁", color=:green, linewidth=2)
    px = plot!(px, getindex.(mean.(xmarginals), 2), 
               ribbon = getindex.(var.(xmarginals), 2) .|> sqrt, 
               fillalpha = 0.5, label = "Inferred x₂", color=:orange, linewidth=2)
    
    px = plot!(px, xlabel="Time index", ylabel="Value")
    
    # Also create a phase space plot comparing true vs inferred
    p2 = scatter(getindex.(x, 1), getindex.(x, 2), 
                 label="True trajectory", 
                 markersize=3, markerstrokewidth=0, 
                 alpha=0.6, color=:blue,
                 title="Phase Space Comparison", 
                 xlabel=L"x_1", ylabel=L"x_2",
                 size=(600, 600), dpi=300, margin=5mm)
    
    # Add inferred trajectory
    scatter!(p2, getindex.(mean.(xmarginals), 1), getindex.(mean.(xmarginals), 2),
             label="Inferred trajectory", markersize=3, 
             markerstrokewidth=0, alpha=0.6, color=:red)
    
    if !isnothing(output_dir)
        savefig(px, joinpath(output_dir, "multivariate_results.png"))
        savefig(p2, joinpath(output_dir, "multivariate_phase_comparison.png"))
    end
    
    return px, p2
end

"""
    animate_optimization(trace, params_history, true_params; output_dir=nothing)

Create an animation of the optimization process.
"""
function animate_optimization(trace, params_history, true_params; output_dir=nothing)
    # Check if we have enough data for animation
    if length(trace) <= 1 || length(params_history) <= 1
        @warn "Not enough data points for animation"
        return nothing
    end
    
    # Create more frames for smoother animation by interpolating
    # between existing points if needed
    n_frames = min(100, length(trace) * 3)
    
    # Plot the trace with a moving point
    anim = @animate for frame_idx in 1:n_frames
        # Map frame index to data index (potentially fractional)
        i_float = 1 + (length(trace) - 1) * (frame_idx - 1) / (n_frames - 1)
        i = min(length(trace), ceil(Int, i_float))
        
        # For smooth transitions
        progress = i_float - floor(i_float)
        
        # Create the trace plot
        p1 = plot(1:length(trace), trace, 
                 label="Free Energy", linewidth=3, 
                 xlabel="Iterations", ylabel="Free Energy",
                 title="Optimization Progress ($(frame_idx)/$(n_frames))",
                 size=(800, 400), dpi=200, margin=5mm,
                 framestyle=:box)
        
        # Add marker at current position
        curr_x = min(length(trace), max(1, i_float))
        curr_y = if floor(Int, curr_x) == ceil(Int, curr_x)
            trace[floor(Int, curr_x)]
        else
            # Interpolate between points for smoother animation
            i_floor = floor(Int, curr_x)
            i_ceil = ceil(Int, curr_x)
            if i_ceil > length(trace)
                trace[i_floor]
            else
                trace[i_floor] * (1-progress) + trace[i_ceil] * progress
            end
        end
        
        scatter!(p1, [curr_x], [curr_y], color=:red, markersize=8, 
                label="Current", markerstrokewidth=1)
        
        # Add gradient arrow to show direction of optimization
        if i < length(trace)
            x1, y1 = curr_x, curr_y
            x2, y2 = curr_x + 0.5, trace[min(length(trace), i+1)]
            plot!(p1, [x1, x2], [y1, y2], arrow=true, color=:black, 
                  linewidth=2, label=nothing)
        end
        
        # Plot the parameter convergence
        p2 = plot(title="Parameter Convergence", 
                  ylabel="Parameter Value", xlabel="Iteration",
                  size=(800, 400), dpi=200, margin=5mm,
                  framestyle=:box)
        
        for j in 1:length(true_params)
            # Get parameter values up to current frame
            param_indices = 1:min(length(params_history), i)
            if isempty(param_indices)
                continue
            end
            
            param_values = [p[j] for p in params_history[param_indices]]
            
            # Plot parameter trajectory
            plot!(p2, 1:length(param_indices), param_values, 
                  label="Parameter $(j)", linewidth=3)
            
            # Add true parameter value as horizontal line
            hline!(p2, [true_params[j]], 
                   linestyle=:dash, color=j, label="True $(j)")
            
            # Add marker at current position
            if !isempty(param_indices)
                scatter!(p2, [length(param_indices)], [param_values[end]], 
                        color=j, markersize=8, label=nothing, markerstrokewidth=1)
            end
        end
        
        plot(p1, p2, layout=(2,1), size=(800, 800))
    end every 1  # Include all frames for smoothness
    
    if !isnothing(output_dir) && length(trace) > 1
        gif(anim, joinpath(output_dir, "optimization_animation.gif"), fps=30)
    end
    
    return anim
end

"""
    visualize_free_energy_landscape(f, param_ranges, true_params; output_dir=nothing, n_points=50)

Create a 2D or 3D visualization of the free energy landscape.
"""
function visualize_free_energy_landscape(f, param_ranges, true_params; output_dir=nothing, n_points=50)
    if length(param_ranges) == 2
        # For 2 parameters, create a 3D surface plot
        p1_range = range(param_ranges[1][1], param_ranges[1][2], length=n_points)
        p2_range = range(param_ranges[2][1], param_ranges[2][2], length=n_points)
        
        # Calculate free energy at each point
        fe_grid = Array{Float64}(undef, n_points, n_points)
        min_fe = Inf
        min_params = [0.0, 0.0]
        
        for (i, p1) in enumerate(p1_range)
            for (j, p2) in enumerate(p2_range)
                try
                    fe = f([p1, p2])
                    fe_grid[i, j] = fe
                    if fe < min_fe
                        min_fe = fe
                        min_params = [p1, p2]
                    end
                catch
                    fe_grid[i, j] = NaN
                end
            end
        end
        
        # Create a surface plot
        p_surface = surface(p1_range, p2_range, fe_grid, 
                          xlabel="Parameter 1", ylabel="Parameter 2",
                          zlabel="Free Energy", 
                          title="Free Energy Landscape",
                          size=(800, 600), dpi=300, margin=5mm,
                          c=:viridis, alpha=0.8)
        
        # Add true parameter values
        scatter!(p_surface, [true_params[1]], [true_params[2]], [f(true_params)], 
                color=:red, markersize=5, label="True")
        
        # Add minimum found
        scatter!(p_surface, [min_params[1]], [min_params[2]], [min_fe], 
                color=:green, markersize=5, label="Minimum")
        
        # Create a contour plot of the same data
        p_contour = contour(p1_range, p2_range, fe_grid',
                           xlabel="Parameter 1", ylabel="Parameter 2",
                           title="Free Energy Contours",
                           size=(800, 600), dpi=300, margin=5mm,
                           c=:viridis, fill=true)
        
        # Add true parameter values
        scatter!(p_contour, [true_params[1]], [true_params[2]], 
                color=:red, markersize=5, label="True")
        
        # Add minimum found
        scatter!(p_contour, [min_params[1]], [min_params[2]], 
                color=:green, markersize=5, label="Minimum")
        
        if !isnothing(output_dir)
            savefig(p_surface, joinpath(output_dir, "free_energy_surface.png"))
            savefig(p_contour, joinpath(output_dir, "free_energy_contours.png"))
        end
        
        return p_surface, p_contour
    else
        # For 1 parameter, create a line plot
        p_range = range(param_ranges[1][1], param_ranges[1][2], length=n_points)
        
        # Calculate free energy at each point
        fe_values = [f([p]) for p in p_range]
        
        # Create a line plot
        p = plot(p_range, fe_values, 
                xlabel="Parameter", ylabel="Free Energy",
                title="Free Energy Landscape", 
                size=(800, 600), dpi=300, margin=5mm,
                linewidth=2, legend=:topright)
        
        # Add true parameter value
        scatter!(p, [true_params[1]], [f([true_params[1]])], 
                color=:red, markersize=5, label="True")
        
        # Add minimum found
        min_idx = argmin(fe_values)
        scatter!(p, [p_range[min_idx]], [fe_values[min_idx]], 
                color=:green, markersize=5, label="Minimum")
        
        if !isnothing(output_dir)
            savefig(p, joinpath(output_dir, "free_energy_1d.png"))
        end
        
        return p
    end
end

"""
    animate_free_energy_descent(f, param_ranges, params_history, true_params; output_dir=nothing, n_points=50)

Create an animation of the optimization process in parameter space with free energy landscape.
"""
function animate_free_energy_descent(f, param_ranges, params_history, true_params; output_dir=nothing, n_points=50)
    if length(params_history) <= 1
        @warn "Not enough points for animation"
        return nothing
    end
    
    if length(param_ranges) != 2 || length(params_history[1]) != 2
        @warn "Animation requires exactly 2 parameters"
        return nothing
    end
    
    # Create grid for contour plot
    p1_range = range(param_ranges[1][1], param_ranges[1][2], length=n_points)
    p2_range = range(param_ranges[2][1], param_ranges[2][2], length=n_points)
    
    # Calculate free energy at each point
    fe_grid = Array{Float64}(undef, n_points, n_points)
    for (i, p1) in enumerate(p1_range)
        for (j, p2) in enumerate(p2_range)
            try
                fe_grid[i, j] = f([p1, p2])
            catch
                fe_grid[i, j] = NaN
            end
        end
    end
    
    # Create more frames for smoother animation
    n_frames = min(100, length(params_history) * 3)
    
    # Create the animation
    anim = @animate for frame_idx in 1:n_frames
        # Map frame index to data index (potentially fractional)
        i_float = 1 + (length(params_history) - 1) * (frame_idx - 1) / (n_frames - 1)
        i = min(length(params_history), ceil(Int, i_float))
        
        # Create the contour plot
        p = contour(p1_range, p2_range, fe_grid',
                   xlabel="Parameter 1", ylabel="Parameter 2",
                   title="Optimization Trajectory ($(frame_idx)/$(n_frames))",
                   size=(800, 800), dpi=200, margin=5mm,
                   c=:viridis, fill=true, colorbar_title="Free Energy")
        
        # Add true parameter values
        scatter!(p, [true_params[1]], [true_params[2]], 
                color=:red, markersize=8, label="True")
        
        # Add optimization trajectory
        param1_values = [p[1] for p in params_history[1:min(length(params_history), i)]]
        param2_values = [p[2] for p in params_history[1:min(length(params_history), i)]]
        
        plot!(p, param1_values, param2_values, 
             linewidth=3, color=:white, label="Trajectory")
        
        # Add current position with larger marker
        if !isempty(param1_values)
            scatter!(p, [param1_values[end]], [param2_values[end]], 
                    color=:green, markersize=10, label="Current")
            
            # Add gradient arrow
            if i < length(params_history)
                next_param = params_history[min(length(params_history), i+1)]
                curr_param = params_history[min(length(params_history), i)]
                
                # Scale the arrow to make it visible
                arrow_scale = 0.5
                arrow_x = curr_param[1] + arrow_scale * (next_param[1] - curr_param[1])
                arrow_y = curr_param[2] + arrow_scale * (next_param[2] - curr_param[2])
                
                plot!(p, [curr_param[1], arrow_x], [curr_param[2], arrow_y], 
                     arrow=true, color=:yellow, linewidth=2, label=nothing)
            end
        end
    end every 1  # Include all frames for smoothness
    
    if !isnothing(output_dir)
        gif(anim, joinpath(output_dir, "free_energy_descent.gif"), fps=30)
    end
    
    return anim
end

"""
    animate_free_energy_evolution(trace, params_history; output_dir=nothing)

Create an animation showing how free energy changes during optimization.
"""
function animate_free_energy_evolution(trace, params_history; output_dir=nothing)
    if length(trace) <= 1 || length(params_history) <= 1
        @warn "Not enough data points for animation"
        return nothing
    end
    
    # Create more frames for smoother animation
    n_frames = min(100, length(trace) * 3)
    
    # Create the animation
    anim = @animate for frame_idx in 1:n_frames
        # Map frame index to data index (potentially fractional)
        i_float = 1 + (length(trace) - 1) * (frame_idx - 1) / (n_frames - 1)
        i = min(length(trace), ceil(Int, i_float))
        
        layout = @layout [a{0.7h}; b{0.3h}]
        
        # Create the trace plot
        p1 = plot(1:i, trace[1:i], 
                 label="Free Energy", linewidth=3, 
                 xlabel="Iterations", ylabel="Free Energy",
                 title="Free Energy Evolution ($(frame_idx)/$(n_frames))",
                 size=(800, 400), dpi=200, margin=5mm,
                 framestyle=:box, legend=:topright)
        
        # Highlight the current point
        scatter!(p1, [i], [trace[i]], color=:red, markersize=8, 
                label="Current", markerstrokewidth=1)
        
        # Add relative improvement text
        if i > 1
            improvement = (trace[1] - trace[i]) / trace[1] * 100
            annotate!(p1, i, trace[i], text("$(round(improvement, digits=2))% improved", 
                                           10, :bottom, :right))
        end
        
        # Calculate convergence rate using exponential fit if enough points
        if i >= 5
            # Normalize for better visualization
            normalized_trace = (trace[1:i] .- minimum(trace[1:i])) ./ 
                              (maximum(trace[1:i]) - minimum(trace[1:i]))
            
            # Create convergence plot
            p2 = plot(1:i, normalized_trace, 
                     label="Normalized FE", linewidth=3, 
                     xlabel="Iterations", ylabel="Normalized Value",
                     size=(800, 200), dpi=200, margin=5mm,
                     framestyle=:box, legend=:topright, color=:red)
            
            # Add a trendline
            if i >= 10
                # Fit exponential decay: a * exp(-b * x) + c
                x = 1:i
                y = normalized_trace
                
                # Simple exponential fit visualization
                if maximum(y) - minimum(y) > 0.01  # Only if enough variation
                    trend = exp.(-0.1 .* x) 
                    trend = (trend .- minimum(trend)) ./ (maximum(trend) - minimum(trend))
                    trend = trend .* (maximum(y) - minimum(y)) .+ minimum(y)
                    
                    plot!(p2, x, trend, color=:black, linestyle=:dash, 
                         linewidth=2, label="Trend")
                end
            end
            
            plot(p1, p2, layout=layout, size=(800, 600))
        else
            plot(p1, size=(800, 600))
        end
    end every 1  # Include all frames for smoothness
    
    if !isnothing(output_dir)
        gif(anim, joinpath(output_dir, "free_energy_evolution.gif"), fps=30)
    end
    
    return anim
end

"""
    create_stacked_free_energy_visualization(trace, params_history, true_params; output_dir=nothing)

Create a stacked visualization of free energy and parameters over iterations.
"""
function create_stacked_free_energy_visualization(trace, params_history, true_params; output_dir=nothing)
    if length(trace) <= 1 || length(params_history) <= 1
        @warn "Not enough data points for visualization"
        return nothing
    end
    
    # Create layout for stacked plots
    layout = @layout [a; b]
    
    # Create the trace plot
    p1 = plot(1:length(trace), trace, 
             label="Free Energy", linewidth=3, 
             xlabel="Iterations", ylabel="Free Energy",
             title="Free Energy Optimization",
             size=(800, 400), dpi=300, margin=5mm,
             framestyle=:box, legend=:topright)
    
    # Add rate of decrease as a percentage
    decreases = diff(trace)
    percent_decreases = abs.(decreases) ./ trace[1:end-1] * 100
    
    # Add decrease rate on secondary axis
    p1_twin = twinx(p1)
    plot!(p1_twin, 2:length(trace), percent_decreases, 
          color=:red, linewidth=2, linestyle=:dash,
          ylabel="% Change", label="% Change/Iteration")
    
    # Create parameter plot
    p2 = plot(title="Parameter Evolution", 
              ylabel="Parameter Value", xlabel="Iteration",
              size=(800, 400), dpi=300, margin=5mm,
              framestyle=:box, legend=:topright)
    
    # Plot parameter trajectories
    for j in 1:length(true_params)
        param_values = [p[j] for p in params_history]
        
        plot!(p2, 1:length(param_values), param_values, 
              label="Parameter $(j)", linewidth=3)
        
        # Add true parameter value as horizontal line
        hline!(p2, [true_params[j]], 
               linestyle=:dash, color=j, label="True $(j)")
        
        # Calculate final percentage error
        if !isempty(param_values)
            error_pct = abs(param_values[end] - true_params[j]) / abs(true_params[j]) * 100
            if abs(true_params[j]) > 1e-10  # Avoid division by near-zero
                annotate!(p2, length(param_values), param_values[end], 
                         text("$(round(error_pct, digits=2))% error", 8, :right))
            end
        end
    end
    
    # Combine plots
    p = plot(p1, p2, layout=layout, size=(800, 800))
    
    if !isnothing(output_dir)
        savefig(p, joinpath(output_dir, "free_energy_stacked.png"))
    end
    
    return p
end

"""
    plot_parameter_correlation(params_history, true_params; output_dir=nothing)

Plot correlation between parameters during optimization.
"""
function plot_parameter_correlation(params_history, true_params; output_dir=nothing)
    n_params = length(params_history[1])
    
    if n_params >= 2
        # For parameters pairs, show the trajectory in parameter space
        p = plot(title="Parameter Space Trajectory", 
                 size=(600, 600), dpi=300, margin=5mm)
        
        # Extract parameter trajectories
        param1_values = [p[1] for p in params_history]
        param2_values = [p[2] for p in params_history]
        
        # Plot the trajectory with color gradient to show progression
        for i in 1:length(params_history)-1
            plot!(p, [param1_values[i], param1_values[i+1]], 
                 [param2_values[i], param2_values[i+1]], 
                 linewidth=2, color=cgrad(:viridis)[i/length(params_history)],
                 label=nothing)
        end
        
        # Add points for start, end, and true parameters
        scatter!(p, [param1_values[1]], [param2_values[1]], 
                 color=:blue, markersize=6, label="Start")
        scatter!(p, [param1_values[end]], [param2_values[end]], 
                 color=:green, markersize=6, label="End")
        scatter!(p, [true_params[1]], [true_params[2]], 
                 color=:red, markersize=6, label="True")
        
        plot!(p, xlabel="Parameter 1", ylabel="Parameter 2")
        
        if !isnothing(output_dir)
            savefig(p, joinpath(output_dir, "parameter_trajectory.png"))
        end
        
        return p
    end
    
    return nothing
end

"""
    save_optimization_trace(trace, title; output_dir=nothing)

Save the optimization trace plot.
"""
function save_optimization_trace(trace, title; output_dir=nothing)
    p = plot(title=title, size=(800, 500), dpi=300, margin=5mm)
    plot!(p, 1:length(trace), trace, label="Free Energy", linewidth=2, 
          xlabel="Iterations", ylabel="Free Energy")
    
    # Calculate convergence rate
    if length(trace) > 10
        # Fit exponential decay model to show convergence rate
        normalized_trace = (trace .- minimum(trace)) ./ (maximum(trace) - minimum(trace))
        
        # Add convergence visualization on a separate axis
        p2 = twinx(p)
        plot!(p2, 1:length(trace), normalized_trace, 
              color=:red, linewidth=1, linestyle=:dash, 
              ylabel="Normalized Value", label="Normalized", ylim=(0, 1))
    end
    
    if !isnothing(output_dir)
        savefig(p, joinpath(output_dir, "optimization_trace.png"))
    end
    
    return p
end

"""
    create_output_dir()

Create a timestamped output directory within the current script directory.
"""
function create_output_dir()
    # Get script directory path
    script_dir = @__DIR__
    
    # Create results dir inside the script directory
    results_dir = joinpath(script_dir, "results")
    isdir(results_dir) || mkdir(results_dir)
    
    # Create timestamped directory
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    output_dir = joinpath(results_dir, timestamp)
    mkpath(output_dir)
    
    # Create log file
    log_file = joinpath(output_dir, "log.txt")
    open(log_file, "w") do io
        println(io, "Parameter Optimisation with Optim.jl")
        println(io, "Run at: ", Dates.now())
        println(io, "Output directory: ", output_dir)
    end
    
    return output_dir, log_file
end

"""
    log_message(message, log_file=nothing)

Log a message to both console and log file if provided.
"""
function log_message(message, log_file=nothing)
    println(message)
    if !isnothing(log_file)
        open(log_file, "a") do io
            println(io, message)
        end
    end
end 