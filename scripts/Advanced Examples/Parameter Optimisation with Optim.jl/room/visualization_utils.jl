using Plots, StatsPlots, LaTeXStrings, Measures, Dates, Statistics

"""
    create_output_dir()

Create a timestamped output directory to store all results.
"""
function create_output_dir()
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    output_dir = joinpath(@__DIR__, "outputs", timestamp)
    mkpath(output_dir)
    log_file = joinpath(output_dir, "log.txt")
    return output_dir, log_file
end

"""
    log_message(message, log_file)

Log a message to both the console and the log file.
"""
function log_message(message, log_file)
    println(message)
    open(log_file, "a") do io
        println(io, message)
    end
end

"""
    plot_room_data(time, temp_data, humid_data, output_dir)

Plot the generated temperature and humidity data.
"""
function plot_room_data(time, temp_data, humid_data; output_dir=nothing)
    plt1 = plot(time, temp_data, 
        label="Temperature", 
        color=:red, 
        lw=2, 
        xlabel="Time (hours)", 
        ylabel="Temperature (°C)",
        title="Room Temperature Measurements",
        size=(800, 400),
        margin=5mm)
    
    plt2 = plot(time, humid_data, 
        label="Humidity", 
        color=:blue, 
        lw=2, 
        xlabel="Time (hours)", 
        ylabel="Humidity (%)",
        title="Room Humidity Measurements",
        size=(800, 400),
        margin=5mm)
    
    plt3 = scatter(temp_data, humid_data,
        xlabel="Temperature (°C)",
        ylabel="Humidity (%)",
        title="Temperature vs Humidity",
        color=:purple,
        alpha=0.6,
        size=(800, 400),
        margin=5mm)
    
    combined_plot = plot(plt1, plt2, plt3, layout=(3,1), size=(900, 900))
    
    if !isnothing(output_dir)
        savefig(plt1, joinpath(output_dir, "temperature_data.png"))
        savefig(plt2, joinpath(output_dir, "humidity_data.png"))
        savefig(plt3, joinpath(output_dir, "temp_vs_humid.png"))
        savefig(combined_plot, joinpath(output_dir, "room_data.png"))
    end
    
    return plt1, plt2, plt3, combined_plot
end

"""
    save_optimization_trace(trace_values, title; output_dir=nothing)

Save a plot of the optimization trace (free energy values).
"""
function save_optimization_trace(trace_values, title; output_dir=nothing)
    p = plot(trace_values, 
        label="Free Energy", 
        title=title,
        xlabel="Iteration", 
        ylabel="Free Energy",
        lw=2, 
        marker=:circle,
        markersize=3,
        color=:blue,
        size=(800, 500))
    
    if !isnothing(output_dir)
        savefig(p, joinpath(output_dir, "optimization_trace.png"))
    end
    
    return p
end

"""
    animate_optimization(trace_values, params_history, true_params; output_dir=nothing)

Create an animation of the optimization process.
"""
function animate_optimization(trace_values, params_history, true_params; output_dir=nothing)
    if length(trace_values) < 3
        return nothing
    end
    
    param_names = ["α (coupling strength)", "β (temperature decay)", "γ (humidity decay)", "T₀ (initial temp)", "H₀ (initial humidity)"]
    param_colors = [:red, :blue, :green, :purple, :orange]
    
    n_params = length(params_history[1])
    if n_params != length(true_params)
        @warn "Parameter count mismatch between history ($(n_params)) and true values ($(length(true_params)))"
        return nothing
    end
    
    anim = @animate for i in 1:length(params_history)
        plots = []
        
        # Plot free energy
        p1 = plot(trace_values[1:min(i, length(trace_values))], 
            title="Free Energy",
            xlabel="Iteration", 
            ylabel="Free Energy",
            lw=2, 
            legend=false,
            size=(400, 300))
        push!(plots, p1)
        
        # Plot each parameter
        for j in 1:n_params
            param_values = [p[j] for p in params_history[1:i]]
            p = plot(param_values, 
                title=param_names[min(j, length(param_names))],
                xlabel="Iteration", 
                ylabel="Value",
                lw=2, 
                color=param_colors[min(j, length(param_colors))],
                legend=false,
                size=(400, 300))
            # Add horizontal line for true parameter value
            hline!(p, [true_params[j]], linestyle=:dash, color=:black, alpha=0.7)
            push!(plots, p)
        end
        
        # Layout arrangement based on number of parameters
        if n_params <= 2
            layout = @layout [a; b c]
        elseif n_params <= 4
            layout = @layout [a; b c; d e]
        else
            layout = @layout [a; b c; d e; f g]
        end
        
        plot(plots[1:min(length(plots), 7)]..., layout=layout, size=(900, 900), margin=5mm)
    end
    
    if !isnothing(output_dir)
        gif(anim, joinpath(output_dir, "optimization_animation.gif"), fps=5)
    end
    
    return anim
end

"""
    visualize_free_energy_landscape(f, param_ranges, true_params; output_dir=nothing, n_points=30)

Visualize the free energy landscape for the first two parameters.
"""
function visualize_free_energy_landscape(f, param_ranges, true_params; output_dir=nothing, n_points=30)
    if length(param_ranges) < 2 || length(true_params) < 2
        @warn "Need at least 2 parameters for landscape visualization"
        return nothing
    end
    
    # Prepare the parameter grid
    param1_range = range(param_ranges[1][1], param_ranges[1][2], length=n_points)
    param2_range = range(param_ranges[2][1], param_ranges[2][2], length=n_points)
    
    # Compute the free energy values
    z = zeros(n_points, n_points)
    
    # Create a progress bar
    p = Progress(n_points^2, desc="Computing free energy landscape...")
    
    # For each point in the grid
    for i in 1:n_points
        for j in 1:n_points
            params = [param1_range[i], param2_range[j]]
            z[j, i] = f(params)  # Note the j,i order to match plot coordinates
            next!(p)
        end
    end
    
    # Create contour plot
    p1 = contour(param1_range, param2_range, z,
        fill=true,
        xlabel=param_ranges[1][3],
        ylabel=param_ranges[2][3],
        title="Free Energy Landscape",
        c=:viridis,
        size=(800, 600))
    
    # Mark the true parameter values
    scatter!(p1, [true_params[1]], [true_params[2]], 
        color=:red, 
        markersize=8, 
        label="True parameters")
    
    # Create 3D surface plot
    p2 = surface(param1_range, param2_range, z,
        xlabel=param_ranges[1][3],
        ylabel=param_ranges[2][3],
        zlabel="Free Energy",
        title="Free Energy Surface",
        c=:viridis,
        size=(800, 600))
    
    # Combine the plots
    combined = plot(p1, p2, layout=(1,2), size=(1600, 600))
    
    if !isnothing(output_dir)
        savefig(p1, joinpath(output_dir, "free_energy_contour.png"))
        savefig(p2, joinpath(output_dir, "free_energy_surface.png"))
        savefig(combined, joinpath(output_dir, "free_energy_landscape.png"))
    end
    
    return p1, p2, combined
end

"""
    animate_free_energy_descent(f, param_ranges, params_history, true_params; output_dir=nothing, n_points=30)

Create an animation showing the optimization path on the free energy landscape.
"""
function animate_free_energy_descent(f, param_ranges, params_history, true_params; output_dir=nothing, n_points=30)
    if length(params_history) < 2
        @warn "Not enough optimization steps for animation"
        return nothing
    end
    
    if length(param_ranges) < 2 || length(true_params) < 2
        @warn "Need at least 2 parameters for landscape animation"
        return nothing
    end
    
    # Extract the first two parameters from history
    path = [[p[1], p[2]] for p in params_history]
    
    # Prepare the parameter grid
    param1_range = range(param_ranges[1][1], param_ranges[1][2], length=n_points)
    param2_range = range(param_ranges[2][1], param_ranges[2][2], length=n_points)
    
    # Compute the free energy values
    z = zeros(n_points, n_points)
    
    # Create a progress bar
    p = Progress(n_points^2, desc="Computing free energy landscape for animation...")
    
    # For each point in the grid
    for i in 1:n_points
        for j in 1:n_points
            params = [param1_range[i], param2_range[j]]
            z[j, i] = f(params)  # Note the j,i order to match plot coordinates
            next!(p)
        end
    end
    
    anim = @animate for i in 1:length(path)
        # Create contour plot
        p1 = contour(param1_range, param2_range, z,
            fill=true,
            xlabel=param_ranges[1][3],
            ylabel=param_ranges[2][3],
            title="Optimization Progress (Step $i / $(length(path)))",
            c=:viridis,
            size=(800, 600))
        
        # Plot the optimization path up to the current step
        plot!(p1, getindex.(path[1:i], 1), getindex.(path[1:i], 2), 
            color=:red, 
            lw=2, 
            label="Optimization path")
        
        # Mark the true parameter values
        scatter!(p1, [true_params[1]], [true_params[2]], 
            color=:green, 
            markersize=8, 
            label="True parameters")
        
        # Mark the current position
        scatter!(p1, [path[i][1]], [path[i][2]], 
            color=:red, 
            markersize=8, 
            label="Current position")
    end
    
    if !isnothing(output_dir)
        gif(anim, joinpath(output_dir, "free_energy_descent.gif"), fps=5)
    end
    
    return anim
end

"""
    plot_inference_results(time, true_temp, true_humid, temp_obs, humid_obs, 
                          inferred_temp, inferred_humid; output_dir=nothing)

Plot the inference results compared to the true values and observations.
"""
function plot_inference_results(time, true_temp, true_humid, temp_obs, humid_obs, 
                               inferred_temp, inferred_humid; output_dir=nothing)
    # Temperature plot
    plt1 = plot(time, true_temp, 
        label="True Temperature", 
        color=:red, 
        lw=2, 
        xlabel="Time (hours)", 
        ylabel="Temperature (°C)",
        title="Temperature: True vs Inferred",
        size=(800, 400),
        margin=5mm)
    
    scatter!(plt1, time, temp_obs, 
        label="Observations", 
        color=:red, 
        alpha=0.3, 
        markersize=3)
    
    plot!(plt1, time, mean.(inferred_temp), 
        label="Inferred Mean", 
        color=:darkred, 
        lw=2)
    
    # Add confidence intervals
    inferred_std = sqrt.(var.(inferred_temp))
    plot!(plt1, time, mean.(inferred_temp) + 2*inferred_std, 
        fillrange=mean.(inferred_temp) - 2*inferred_std, 
        fillalpha=0.2, 
        label="95% CI", 
        color=:red)
    
    # Humidity plot
    plt2 = plot(time, true_humid, 
        label="True Humidity", 
        color=:blue, 
        lw=2, 
        xlabel="Time (hours)", 
        ylabel="Humidity (%)",
        title="Humidity: True vs Inferred",
        size=(800, 400),
        margin=5mm)
    
    scatter!(plt2, time, humid_obs, 
        label="Observations", 
        color=:blue, 
        alpha=0.3, 
        markersize=3)
    
    plot!(plt2, time, mean.(inferred_humid), 
        label="Inferred Mean", 
        color=:darkblue, 
        lw=2)
    
    # Add confidence intervals
    inferred_std = sqrt.(var.(inferred_humid))
    plot!(plt2, time, mean.(inferred_humid) + 2*inferred_std, 
        fillrange=mean.(inferred_humid) - 2*inferred_std, 
        fillalpha=0.2, 
        label="95% CI", 
        color=:blue)
    
    # Phase space plot
    plt3 = scatter(true_temp, true_humid,
        label="True Trajectory",
        xlabel="Temperature (°C)",
        ylabel="Humidity (%)",
        title="Temperature vs Humidity Phase Space",
        color=:purple,
        alpha=0.6,
        size=(800, 400),
        margin=5mm)
    
    scatter!(plt3, temp_obs, humid_obs,
        label="Observations",
        color=:gray,
        alpha=0.3,
        markersize=3)
    
    scatter!(plt3, mean.(inferred_temp), mean.(inferred_humid),
        label="Inferred Trajectory",
        color=:orange,
        alpha=0.6)
    
    combined_plot = plot(plt1, plt2, plt3, layout=(3,1), size=(900, 900))
    
    if !isnothing(output_dir)
        savefig(plt1, joinpath(output_dir, "temperature_inference.png"))
        savefig(plt2, joinpath(output_dir, "humidity_inference.png"))
        savefig(plt3, joinpath(output_dir, "phase_space_inference.png"))
        savefig(combined_plot, joinpath(output_dir, "inference_results.png"))
    end
    
    return plt1, plt2, plt3, combined_plot
end

"""
    animate_state_evolution(time, true_temp, true_humid, temp_obs, humid_obs, 
                           inferred_temp, inferred_humid; output_dir=nothing)

Create an animation showing the evolution of the system state over time.
"""
function animate_state_evolution(time, true_temp, true_humid, temp_obs, humid_obs, 
                                inferred_temp, inferred_humid; output_dir=nothing)
    n = length(time)
    
    anim = @animate for i in 1:n
        # Phase space plot
        p = scatter(true_temp[1:i], true_humid[1:i],
            label="True Trajectory",
            xlabel="Temperature (°C)",
            ylabel="Humidity (%)",
            title="State Evolution (Frame $i/$n)",
            color=:purple,
            alpha=0.6,
            size=(800, 600),
            margin=5mm,
            xlim=(minimum(true_temp)-1, maximum(true_temp)+1),
            ylim=(minimum(true_humid)-1, maximum(true_humid)+1))
        
        # Add observations
        if i > 0
            scatter!(p, temp_obs[1:min(i,length(temp_obs))], humid_obs[1:min(i,length(humid_obs))],
                label="Observations",
                color=:gray,
                alpha=0.3,
                markersize=3)
        end
        
        # Add inferred trajectory if available
        if i <= length(inferred_temp)
            scatter!(p, mean.(inferred_temp[1:i]), mean.(inferred_humid[1:i]),
                label="Inferred Trajectory",
                color=:orange,
                alpha=0.6)
        end
        
        # Highlight current point
        if i > 0
            scatter!(p, [true_temp[i]], [true_humid[i]],
                label="Current State",
                color=:green,
                markersize=8)
        end
        
        # Add direction arrow if not at the beginning
        if i > 1
            plot!(p, [true_temp[i-1], true_temp[i]],
                  [true_humid[i-1], true_humid[i]],
                  arrow=true, linewidth=2, color=:black, label=nothing)
        end
    end
    
    if !isnothing(output_dir)
        gif(anim, joinpath(output_dir, "state_evolution.gif"), fps=10)
    end
    
    return anim
end 