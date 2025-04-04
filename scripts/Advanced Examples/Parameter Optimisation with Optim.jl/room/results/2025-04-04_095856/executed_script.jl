#!/usr/bin/env julia
# Room Temperature and Humidity Parameter Optimization with RxInfer
# This script uses a minimal state-space model for room temperature dynamics

using Distributions, LinearAlgebra, Plots, Random, StatsPlots, Dates
using Printf, Optim, LaTeXStrings, RxInfer

# Create timestamped output directory
function create_output_dir()
    # Create results dir inside the script directory
    results_dir = joinpath(@__DIR__, "results")
    isdir(results_dir) || mkdir(results_dir)
    
    # Create timestamped directory
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    output_dir = joinpath(results_dir, timestamp)
    mkpath(output_dir)
    
    # Create log file
    log_file = joinpath(output_dir, "log.txt")
    open(log_file, "w") do io
        println(io, "Room Temperature Parameter Optimisation")
        println(io, "Run at: ", Dates.now())
        println(io, "Output directory: ", output_dir)
    end
    
    return output_dir, log_file
end

output_dir, log_file_path = create_output_dir()
log_file = open(log_file_path, "a")

function log_message(message)
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    println("[$timestamp] $message")
    println(log_file, "[$timestamp] $message")
    flush(log_file)
end

log_message("Starting room temperature parameter optimization simulation")

# Define the model using a bare-bones approach similar to the working example
@model function room_model(temp_obs, α, noise_T)
    # Set prior for initial temperature
    T0 ~ Normal(mean = temp_obs[1], variance = 5.0)
    
    # Initial value
    T_prev = T0
    
    for i in eachindex(temp_obs)
        # Simple linear model (similar to univariate example)
        T[i] := T_prev + α
        
        # Observation with Gaussian noise
        temp_obs[i] ~ Normal(mean = T[i], variance = noise_T^2)
        
        # Update previous value for next iteration
        T_prev = T[i]
    end
end

# Function to generate synthetic room data
function generate_room_data(α_true, n_hours, dt; seed=42)
    log_message("Generating synthetic room data with true parameter: α=$α_true")
    
    # Set random seed for reproducibility
    rng = MersenneTwister(seed)
    
    # Time parameters
    n_steps = Int(n_hours * 3600 / dt)
    
    # Initial conditions
    T0 = 22.0  # Initial temperature (°C)
    
    # Noise parameters
    noise_T = 0.5  # Temperature measurement noise (°C)
    
    # Arrays to store data
    times = collect(0:dt:(n_steps-1)*dt)
    temp_true = zeros(n_steps)
    temp_obs = zeros(n_steps)
    
    # Set initial conditions
    temp_true[1] = T0
    temp_obs[1] = T0 + noise_T * randn(rng)
    
    # Simulate the room dynamics
    for i in 2:n_steps
        # Temperature changes by a constant rate α
        temp_true[i] = temp_true[i-1] + α_true
        
        # Add measurement noise
        temp_obs[i] = temp_true[i] + noise_T * randn(rng)
    end
    
    log_message("Generated $n_steps data points ($n_hours hours with dt=$dt seconds)")
    
    return (
        times = times,
        temp_true = temp_true,
        temp_obs = temp_obs,
        noise_T = noise_T,
        T0 = T0
    )
end

# Generate synthetic data
α_true = 0.1   # Temperature change rate per time step
n_hours = 4.0  # Simulation time in hours
dt = 30.0      # Time step in seconds

data = generate_room_data(α_true, n_hours, dt)

# Plot the generated data
function plot_room_data(data)
    p = plot(data.times / 3600, data.temp_true, label="True temperature", 
              xlabel="Time (hours)", ylabel="Temperature (°C)",
              title="Room Temperature Over Time", lw=2, legend=:topright)
    scatter!(p, data.times / 3600, data.temp_obs, label="Observations", 
             alpha=0.3, ms=2)
    
    savefig(p, joinpath(output_dir, "room_data.png"))
    log_message("Room data plot saved to $(joinpath(output_dir, "room_data.png"))")
    return p
end

# Create animation of the generative process
function animate_generative_process(data)
    log_message("Creating animation of generative process...")
    n_points = length(data.times)
    anim = @animate for i in 1:min(n_points, 100)
        # Map animation frame to actual data index for smoother animation
        idx = Int(ceil(i * n_points / 100))
        
        p = plot(title="Room Temperature Generative Process", 
                 xlabel="Time (hours)", ylabel="Temperature (°C)",
                 xlim=(0, data.times[end]/3600), 
                 ylim=(data.T0-1, data.temp_true[end]+1),
                 size=(800, 500), legend=:topleft)
        
        # Plot the true generative process
        if idx > 1
            plot!(p, data.times[1:idx] / 3600, data.temp_true[1:idx], 
                  label="True temperature (α=$α_true)", lw=2, color=:blue)
        end
        
        # Add observations with noise
        if idx > 1
            scatter!(p, data.times[1:idx] / 3600, data.temp_obs[1:idx], 
                     label="Observations (noise=$(data.noise_T)°C)", alpha=0.5, 
                     color=:red, ms=3)
        end
        
        # Add annotation explaining the model
        annotate!(p, 0.5, data.temp_true[end], 
                  text("T[i] = T[i-1] + α\nwhere α=$α_true", 10, :left))
        
        # Highlight the most recent point
        if idx > 1
            scatter!(p, [data.times[idx]/3600], [data.temp_obs[idx]], 
                     ms=5, color=:red, label="Current observation")
            scatter!(p, [data.times[idx]/3600], [data.temp_true[idx]], 
                     ms=5, color=:blue, label="Current true value")
        end
    end
    
    gif_path = joinpath(output_dir, "generative_process.gif")
    gif(anim, gif_path, fps=15)
    log_message("Generative process animation saved to $gif_path")
    return anim
end

# Plot and save the generated data
plot_room_data(data)
animate_generative_process(data)

# Define the optimization function
function f(params)
    # Extract parameter
    α = params[1]
    
    # Run inference with the simplified model
    result = infer(
        model = room_model(
            α = α,
            noise_T = data.noise_T
        ),
        data = (temp_obs = data.temp_obs,),
        free_energy = true
    )
    
    # Return free energy (for minimization)
    return result.free_energy[end]
end

# Explore the free energy landscape
function plot_free_energy_landscape()
    log_message("Exploring free energy landscape...")
    # Calculate free energy for a range of alpha values
    alphas = range(0.0, 0.2, length=50)
    energies = Float64[]
    
    for α in alphas
        try
            energy = f([α])
            push!(energies, energy)
        catch
            # If error occurs, add NaN to maintain index alignment
            push!(energies, NaN)
        end
    end
    
    # Plot the free energy landscape
    p = plot(alphas, energies, 
             title="Free Energy Landscape", 
             xlabel="α parameter", 
             ylabel="Free Energy",
             lw=2, color=:blue, legend=false)
    
    # Mark the true value
    vline!([α_true], linestyle=:dash, color=:red, 
           label="True α=$(α_true)")
    
    # Annotate the minimum
    min_idx = argmin(filter(!isnan, energies))
    α_min = alphas[min_idx]
    annotate!(p, α_min, energies[min_idx], 
              text("Minimum: α=$(round(α_min, digits=3))", 8, :bottom))
    
    # Save the plot
    savefig(p, joinpath(output_dir, "free_energy_landscape.png"))
    log_message("Free energy landscape saved to $(joinpath(output_dir, "free_energy_landscape.png"))")
    return p
end

# Function to collect parameters during optimization
params_history = Vector{Vector{Float64}}()
fe_history = Vector{Float64}()
function f_with_history(params)
    push!(params_history, copy(params))
    fe = f(params)
    push!(fe_history, fe)
    return fe
end

# Plot the free energy landscape before optimization
plot_free_energy_landscape()

# Initial parameter guess
initial_params = [0.05]
log_message("Starting parameter optimization")
log_message("Initial parameter guess: α=$(initial_params[1])")

# Optimize using LBFGS
log_message("Running optimization with LBFGS method...")
result = optimize(
    f_with_history,
    initial_params,
    LBFGS(),
    Optim.Options(g_tol = 1e-3, iterations = 100, store_trace=true, show_trace=true)
)

# Log optimization results
log_message("Optimization complete")
log_message("True parameter: α=$α_true")
log_message("Optimized parameter: α=$(result.minimizer[1])")
log_message("Converged: $(Optim.converged(result))")

# Animate the optimization process
function animate_optimization(params_history, fe_history, α_true)
    log_message("Creating animation of optimization process...")
    
    # Create a range of alpha values for the background landscape
    alphas = range(0.0, 0.2, length=50)
    energies = Float64[]
    
    for α in alphas
        try
            energy = f([α])
            push!(energies, energy)
        catch
            # If error occurs, add NaN to maintain index alignment
            push!(energies, NaN)
        end
    end
    
    # Create the animation
    n_frames = min(length(params_history), 40)
    indices = round.(Int, range(1, length(params_history), length=n_frames))
    
    anim = @animate for i in indices
        # Plot the free energy landscape
        p1 = plot(alphas, energies, 
                 title="Parameter Optimization Progress", 
                 xlabel="α parameter", 
                 ylabel="Free Energy",
                 lw=2, color=:blue, alpha=0.5, legend=:topright,
                 size=(800, 400))
        
        # Mark the true value
        vline!(p1, [α_true], linestyle=:dash, color=:red, 
               label="True α=$(α_true)")
        
        # Show the optimization path
        alphas_path = [p[1] for p in params_history[1:i]]
        scatter!(p1, alphas_path, fe_history[1:i], 
                 label="Optimization path", color=:green, ms=3)
        
        # Show the current point
        scatter!(p1, [params_history[i][1]], [fe_history[i]], 
                 label="Current: α=$(round(params_history[i][1], digits=4))", 
                 color=:red, ms=6)
        
        # Second plot for parameter convergence
        p2 = plot(1:i, [p[1] for p in params_history[1:i]], 
                 title="Parameter Convergence", 
                 xlabel="Iteration", 
                 ylabel="α parameter",
                 lw=2, color=:green, legend=:topright,
                 size=(800, 300))
        
        hline!(p2, [α_true], linestyle=:dash, color=:red, 
               label="True α=$(α_true)")
        
        # Calculate error
        error_pct = abs(params_history[i][1] - α_true) / α_true * 100
        annotate!(p2, i, params_history[i][1], 
                  text("Error: $(round(error_pct, digits=2))%", 8, :bottom))
        
        # Combine plots
        plot(p1, p2, layout=(2,1), size=(800, 700))
    end
    
    gif_path = joinpath(output_dir, "optimization_process.gif")
    gif(anim, gif_path, fps=5)
    log_message("Optimization process animation saved to $gif_path")
    return anim
end

# Save results to file
open(joinpath(output_dir, "optimization_results.txt"), "w") do io
    println(io, "Room Temperature Parameter Optimization Results")
    println(io, "=============================================")
    println(io, "True parameter:")
    println(io, "α = $α_true")
    println(io, "")
    println(io, "Optimized parameter:")
    println(io, "α = $(result.minimizer[1])")
    println(io, "")
    println(io, "Relative error:")
    println(io, "α error: $(@sprintf("%.2f%%", 100 * abs(result.minimizer[1] - α_true) / α_true))")
    println(io, "")
    println(io, "Optimization details:")
    println(io, "Converged: $(Optim.converged(result))")
    println(io, "Iterations: $(result.iterations)")
    println(io, "Final free energy: $(result.minimum)")
end

# Animate the optimization process
animate_optimization(params_history, fe_history, α_true)

# Run inference with optimized parameter
log_message("Running inference with optimized parameter")
final_result = infer(
    model = room_model(
        α = result.minimizer[1],
        noise_T = data.noise_T
    ),
    data = (temp_obs = data.temp_obs,)
)

# Plot the final inference results
function plot_inference_results(data, result)
    T_marginals = result.posteriors[:T]
    
    T_means = mean.(T_marginals)
    T_stds = sqrt.(var.(T_marginals))
    
    p = plot(data.times / 3600, data.temp_true, label="True temperature", 
              xlabel="Time (hours)", ylabel="Temperature (°C)",
              title="Inferred Room Temperature", lw=2, legend=:topright)
    plot!(p, data.times / 3600, T_means, ribbon=T_stds, label="Inferred temperature", 
          alpha=0.6, lw=2)
    scatter!(p, data.times / 3600, data.temp_obs, label="Observations", 
             alpha=0.3, ms=2)
    
    savefig(p, joinpath(output_dir, "inference_results.png"))
    log_message("Inference results plot saved to $(joinpath(output_dir, "inference_results.png"))")
    return p
end

# Create animation showing the inference results evolving over time
function animate_inference_results(data, result, optimized_param)
    log_message("Creating animation of inference results...")
    T_marginals = result.posteriors[:T]
    T_means = mean.(T_marginals)
    T_stds = sqrt.(var.(T_marginals))
    
    n_points = length(data.times)
    step = max(1, div(n_points, 50))  # Show ~50 frames for smoother animation
    
    anim = @animate for i in 1:step:n_points
        p = plot(title="Room Temperature Model Fitting", 
                 xlabel="Time (hours)", ylabel="Temperature (°C)",
                 size=(800, 500), legend=:topleft)
        
        # Plot observations up to current point
        scatter!(p, data.times[1:i] / 3600, data.temp_obs[1:i], 
                 label="Observations", alpha=0.3, ms=2, color=:red)
        
        # Plot true temperature
        plot!(p, data.times[1:i] / 3600, data.temp_true[1:i], 
              label="True temperature (α=$α_true)", lw=2, color=:blue)
        
        # Plot inferred temperature with uncertainty
        plot!(p, data.times[1:i] / 3600, T_means[1:i], 
              ribbon=T_stds[1:i], 
              label="Inferred (α=$(round(optimized_param, digits=4)))", 
              lw=2, color=:green, alpha=0.6)
        
        # Add annotation for the current time
        current_hour = data.times[i] / 3600
        annotate!(p, current_hour, minimum(data.temp_obs) - 1, 
                  text("Time: $(round(current_hour, digits=1)) hours", 10, :top))
    end
    
    gif_path = joinpath(output_dir, "inference_results.gif")
    gif(anim, gif_path, fps=10)
    log_message("Inference results animation saved to $gif_path")
    return anim
end

# Plot and save the inference results
plot_inference_results(data, final_result)
animate_inference_results(data, final_result, result.minimizer[1])

# Plot parameter convergence
function plot_parameter_convergence(params_history, true_param, fe_history)
    p1 = plot(
        title = "Parameter Convergence",
        xlabel = "Iteration",
        ylabel = "Parameter Value",
        size = (800, 300),
        legend = :topright
    )
    
    values = [p[1] for p in params_history]
    plot!(p1, values, label="α (optimized)", lw=2, color=:blue)
    hline!(p1, [true_param], label="α (true)", 
           linestyle=:dash, lw=2, color=:red)
    
    # Also plot free energy evolution
    p2 = plot(
        title = "Free Energy Evolution",
        xlabel = "Iteration",
        ylabel = "Free Energy",
        size = (800, 300),
        legend = :topright
    )
    
    plot!(p2, fe_history, lw=2, color=:purple, label="Free Energy")
    
    # Combine plots
    p = plot(p1, p2, layout=(2,1), size=(800, 600))
    
    savefig(p, joinpath(output_dir, "parameter_convergence.png"))
    log_message("Parameter convergence plot saved to $(joinpath(output_dir, "parameter_convergence.png"))")
    return p
end

# Comprehensive visualization of the optimization process
plot_parameter_convergence(params_history, α_true, fe_history)

# Make a copy of the script in the output directory for reproducibility
cp(@__FILE__, joinpath(output_dir, "executed_script.jl"), force=true)
log_message("✓ Saved copy of script for reproducibility")

log_message("Simulation completed successfully")
log_message("All results saved to: $output_dir")
close(log_file) 