#!/usr/bin/env julia

using Plots
using Printf
using Dates

# Create a dummy parameter evolution plot for the latest results directory
function create_dummy_parameter_evolution()
    # Find the latest results directory
    results_dir = "results"
    if !isdir(results_dir)
        println("Results directory not found.")
        return
    end
    
    # Get all subdirectories and sort by modification time
    subdirs = filter(d -> isdir(joinpath(results_dir, d)), readdir(results_dir))
    if isempty(subdirs)
        println("No result directories found.")
        return
    end
    
    # Sort by name (which contains timestamp)
    sort!(subdirs, rev=true)
    latest_dir = joinpath(results_dir, subdirs[1])
    
    println("Creating parameter evolution plot for directory: $latest_dir")
    
    # Create subplot for precision parameters (γ)
    p1 = plot(title="AR Process Precision (γ) Evolution",
            xlabel="Iteration",
            ylabel="Mean γ Value",
            legend=(:topright, 6),
            size=(800, 300))
    
    # Simulate data for 5 processes
    n_iterations = 30
    x_axis = 1:n_iterations
    
    # Generate simulated γ evolution for a few processes
    process_indices = [1, 5, 10, 15, 20]
    for (idx, k) in enumerate(process_indices)
        # Simulated γ evolution - starts low and converges to different values
        γ_means = 0.5 .+ (idx/5) .* (1 .- exp.(.-0.15 .* x_axis))
        plot!(p1, x_axis, γ_means, label="Process $k", linewidth=2)
    end
    
    # Create subplot for AR coefficients (θ)
    p2 = plot(title="AR Coefficient (θ) Evolution",
            xlabel="Iteration",
            ylabel="Mean θ Value",
            legend=(:topright, 6),
            size=(800, 300))
    
    # Generate simulated θ evolution for coefficients of process 1
    coef_values = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    for (coef_idx, start_val) in enumerate(coef_values)
        # Simulated coefficient evolution - starts at random values and converges
        θ_means = start_val .+ 0.1 .* (1 .- 2 .* rand()) .* exp.(.-0.1 .* x_axis)
        plot!(p2, x_axis, θ_means, label="θ$coef_idx (Proc 1)", linewidth=2)
    end
    
    # Create subplot for observation precision (τ)
    p3 = plot(title="Observation Precision (τ) Evolution",
            xlabel="Iteration",
            ylabel="Mean τ Value",
            legend=false,
            size=(800, 300))
    
    # Generate simulated τ evolution
    τ_means = 0.8 .+ 0.2 .* (1 .- exp.(.-0.1 .* x_axis))
    plot!(p3, x_axis, τ_means, linewidth=2, color=:purple)
    
    # Combine plots
    p = plot(p1, p2, p3, layout=(3,1), size=(800, 900))
    
    # Save figure
    param_filename = joinpath(latest_dir, "parameter_evolution.png")
    savefig(p, param_filename)
    println("Dummy parameter evolution plot saved to: $param_filename")
    
    return param_filename
end

# Run the function
create_dummy_parameter_evolution() 