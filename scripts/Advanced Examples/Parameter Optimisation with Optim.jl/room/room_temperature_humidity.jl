#!/usr/bin/env julia
# Room Temperature and Humidity Parameter Optimization with RxInfer
# This script uses a minimal state-space model for room temperature dynamics

using Distributions, LinearAlgebra, Plots, Random, StatsPlots, Dates
using Printf, Optim, LaTeXStrings, RxInfer

# Create outputs directory if it doesn't exist
output_dir = joinpath(@__DIR__, "outputs")
isdir(output_dir) || mkdir(output_dir)

# Create a log file
log_file = open(joinpath(output_dir, "optimization_log.txt"), "w")

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
        noise_T = noise_T
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
              title="Room Temperature", lw=2, legend=:topright)
    scatter!(p, data.times / 3600, data.temp_obs, label="Observations", 
             alpha=0.3, ms=2)
    
    savefig(p, joinpath(output_dir, "room_data.png"))
    log_message("Room data plot saved to outputs/room_data.png")
    return p
end

# Plot and save the generated data
plot_room_data(data)

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

# Function to collect parameters during optimization
params_history = Vector{Vector{Float64}}()
function f_with_history(params)
    push!(params_history, copy(params))
    return f(params)
end

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
    Optim.Options(g_tol = 1e-3, iterations = 100)
)

# Log optimization results
log_message("Optimization complete")
log_message("True parameter: α=$α_true")
log_message("Optimized parameter: α=$(result.minimizer[1])")
log_message("Converged: $(Optim.converged(result))")

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
    log_message("Inference results plot saved to outputs/inference_results.png")
    return p
end

# Plot and save the inference results
plot_inference_results(data, final_result)

# Plot parameter convergence
function plot_parameter_convergence(params_history, true_param)
    p = plot(
        title = "Parameter Convergence",
        xlabel = "Iteration",
        ylabel = "Parameter Value",
        size = (800, 600),
        legend = :topright
    )
    
    values = [p[1] for p in params_history]
    plot!(p, values, label="α (optimized)", lw=2, color=:red)
    hline!(p, [true_param], label="α (true)", 
           linestyle=:dash, lw=2, color=:red)
    
    savefig(p, joinpath(output_dir, "parameter_convergence.png"))
    log_message("Parameter convergence plot saved to outputs/parameter_convergence.png")
    return p
end

# Plot parameter convergence
plot_parameter_convergence(params_history, α_true)

log_message("Simulation completed successfully")
close(log_file) 