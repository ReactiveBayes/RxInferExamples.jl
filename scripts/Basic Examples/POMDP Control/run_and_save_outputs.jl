#!/usr/bin/env julia

"""
This script runs the POMDP Control example and saves all outputs to files.
"""

# Directory where this script is located - all outputs will be saved here
const OUTPUT_DIR = dirname(@__FILE__)

# Install required packages if not already installed
println("Ensuring all required packages are installed...")
using Pkg

# Read the Project.toml to get package dependencies
project_file = joinpath(OUTPUT_DIR, "Project.toml")
if isfile(project_file)
    # Activate this project locally
    Pkg.activate(OUTPUT_DIR)
    # Install all dependencies
    Pkg.instantiate()
    println("✓ Project dependencies installed")
else
    # Fallback to installing packages directly if Project.toml is not found
    required_packages = [
        "RxInfer", 
        "Distributions", 
        "Plots", 
        "Random", 
        "ProgressMeter",
        "RxEnvironments"
    ]
    
    for pkg in required_packages
        println("Installing $pkg...")
        Pkg.add(pkg)
    end
    println("✓ Required packages installed")
end

println("\nRunning POMDP Control example...")

# Include the original script but capture and save all outputs
include(joinpath(OUTPUT_DIR, "POMDP Control.jl"))

# Now add code to save all outputs
println("\nSaving outputs to $(OUTPUT_DIR)")

# Save the final environment plot
final_plot = plot_environment(env)
savefig(final_plot, joinpath(OUTPUT_DIR, "environment_final.png"))
println("✓ Saved final environment plot")

# Save the success rate
open(joinpath(OUTPUT_DIR, "success_rate.txt"), "w") do io
    success_rate = mean(successes)
    println(io, "Success rate: $(success_rate * 100)%")
    println(io, "$(sum(successes)) successes out of $(length(successes)) experiments")
end
println("✓ Saved success rate")

# Save the model parameters (visualization of learned transition and observation models)
# Visualize and save transition model (B)
heatmap_B = heatmap(
    mean(p_B)[:,:,1], 
    title="Transition Model (Action 1 - Up)", 
    xlabel="Next State", 
    ylabel="Current State"
)
savefig(heatmap_B, joinpath(OUTPUT_DIR, "transition_model_up.png"))

heatmap_B = heatmap(
    mean(p_B)[:,:,2], 
    title="Transition Model (Action 2 - Right)", 
    xlabel="Next State", 
    ylabel="Current State"
)
savefig(heatmap_B, joinpath(OUTPUT_DIR, "transition_model_right.png"))

heatmap_B = heatmap(
    mean(p_B)[:,:,3], 
    title="Transition Model (Action 3 - Down)", 
    xlabel="Next State", 
    ylabel="Current State"
)
savefig(heatmap_B, joinpath(OUTPUT_DIR, "transition_model_down.png"))

heatmap_B = heatmap(
    mean(p_B)[:,:,4], 
    title="Transition Model (Action 4 - Left)", 
    xlabel="Next State", 
    ylabel="Current State"
)
savefig(heatmap_B, joinpath(OUTPUT_DIR, "transition_model_left.png"))
println("✓ Saved transition model visualizations")

# Visualize and save observation model (A)
heatmap_A = heatmap(
    mean(p_A), 
    title="Observation Model", 
    xlabel="Observation", 
    ylabel="State"
)
savefig(heatmap_A, joinpath(OUTPUT_DIR, "observation_model.png"))
println("✓ Saved observation model visualization")

# Save a visualization of the grid world with wind
grid_world_vis = plot(
    title="Windy Grid World", 
    xlims=(0, 6), 
    ylims=(0, 6), 
    aspect_ratio=:equal, 
    legend=:topleft
)
# Add grid lines
for i in 1:5
    plot!(grid_world_vis, [i, i], [0, 6], color=:gray, alpha=0.5, label="")
    plot!(grid_world_vis, [0, 6], [i, i], color=:gray, alpha=0.5, label="")
end
# Mark the goal
scatter!(grid_world_vis, [4], [3], color=:blue, markersize=10, label="Goal")
# Mark the start
scatter!(grid_world_vis, [1], [1], color=:red, markersize=10, label="Start")
# Show wind strengths
for (x, wind) in enumerate(env.decorated.wind)
    if wind != 0
        annotate!(grid_world_vis, [(x, 0.5, text("↑"^wind, 14, :black))])
    end
end
savefig(grid_world_vis, joinpath(OUTPUT_DIR, "grid_world.png"))
println("✓ Saved grid world visualization")

# Save experiment data
open(joinpath(OUTPUT_DIR, "experiment_data.txt"), "w") do io
    println(io, "Experiment configuration:")
    println(io, "- Number of experiments: $n_experiments")
    println(io, "- Planning horizon: $T")
    println(io, "- Start position: (1, 1)")
    println(io, "- Goal position: (4, 3)")
    println(io, "- Wind settings: $(env.decorated.wind)")
    println(io, "\nExperiment results:")
    println(io, "- Success rate: $(mean(successes) * 100)%")
    println(io, "- $(sum(successes)) experiments reached the goal")
    println(io, "- $(length(successes) - sum(successes)) experiments failed to reach the goal")
end
println("✓ Saved experiment data")

println("\nAll outputs have been saved to $(OUTPUT_DIR)") 