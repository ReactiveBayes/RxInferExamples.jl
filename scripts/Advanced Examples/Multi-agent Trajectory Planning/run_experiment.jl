#!/usr/bin/env julia
# run_experiment.jl - Run a single multi-agent trajectory planning experiment

# Activate the local project environment
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Set GR backend for headless operation
ENV["GKSwstype"] = "100"

# Load the module
println("Loading TrajectoryPlanning module...")
include("TrajectoryPlanning.jl")
using .TrajectoryPlanning
using Plots
using Dates

function main()
    # Experiment parameters (defaults, can be overridden by command line arguments)
    environment_type = "combined"  # One of: "door", "wall", "combined"
    seed = 42
    save_intermediate_steps = true
    output_dir = ""  # Empty means auto-generate a timestamped directory

# Parse command line arguments
for arg in ARGS
    if startswith(arg, "--env=")
        environment_type = split(arg, "=")[2]
    elseif startswith(arg, "--seed=")
        seed = parse(Int, split(arg, "=")[2])
    elseif startswith(arg, "--output=")
        output_dir = split(arg, "=")[2]
    elseif arg == "--no-intermediates"
        save_intermediate_steps = false
    elseif arg == "--help"
        println("""
        Usage: julia run_experiment.jl [options]
        
        Options:
          --env=TYPE         Choose environment type: door, wall, combined (default: combined)
          --seed=NUMBER      Set random seed (default: 42)
          --output=DIR       Specify output directory (default: auto-generated timestamp)
          --no-intermediates Don't save intermediate steps
          --help             Show this help message
        """)
        exit(0)
    end
end

# Create output directory if not specified
if isempty(output_dir)
    output_dir = TrajectoryPlanning.create_timestamped_dir()
end
mkpath(output_dir)

# Setup logging
log_file = joinpath(output_dir, "experiment.log")
logger = TrajectoryPlanning.DualLogger(log_file)

# Select environment based on user choice
println("Running experiment with $environment_type environment, seed=$seed")
TrajectoryPlanning.log_message("Starting experiment with $environment_type environment, seed=$seed", logger)

# Create environment and agents
local environment
if environment_type == "door"
    environment = TrajectoryPlanning.create_door_environment()
    TrajectoryPlanning.log_message("Created door environment", logger)
elseif environment_type == "wall"
    environment = TrajectoryPlanning.create_wall_environment()
    TrajectoryPlanning.log_message("Created wall environment", logger)
elseif environment_type == "combined"
    environment = TrajectoryPlanning.create_combined_environment()
    TrajectoryPlanning.log_message("Created combined environment", logger)
else
    error("Unknown environment type: $environment_type")
end

agents = TrajectoryPlanning.create_standard_agents()
TrajectoryPlanning.log_message("Created agents", logger)

try
    # Create environment visualization
    TrajectoryPlanning.log_message("Generating environment visualization...", logger)
    TrajectoryPlanning.visualize_obstacle_distance(
        environment, 
        output_dir=output_dir, 
        filename="environment_heatmap.png", 
        logger=logger
    )
    
    # Run the experiment
    TrajectoryPlanning.log_message("Running path planning experiment...", logger)
    result = TrajectoryPlanning.execute_and_save_animation(
        environment, 
        agents, 
        output_dir=output_dir, 
        gifname="trajectories.gif", 
        seed=seed, 
        save_intermediates=save_intermediate_steps,
        logger=logger
    )
    
    # Generate a detailed summary
    TrajectoryPlanning.generate_experiment_summary(
        environment=environment,
        agents=agents,
        result=result,
        output_dir=output_dir,
        logger=logger
    )
    
    # Create README
    TrajectoryPlanning.create_readme(output_dir, logger)
    
    TrajectoryPlanning.log_message("Experiment completed successfully", logger)
    TrajectoryPlanning.log_message("Results saved to: $output_dir", logger)
    println("Experiment completed successfully. Results saved to: $output_dir")
catch e
    TrajectoryPlanning.log_message("ERROR: Experiment failed with error: $e", logger)
    for (exc, bt) in Base.catch_stack()
        showerror(logger.log_file, exc, bt)
        println(logger.log_file)
    end
    rethrow(e)
finally
    # Close the logger
    TrajectoryPlanning.close_logger(logger)
end
end

# Call the main function
main()