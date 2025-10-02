#!/usr/bin/env julia
# Main runner script for Generic Agent-Environment Framework

@doc """
Main entry point for running agent-environment simulations.

Usage:
    julia run.jl                    # Show help
    julia run.jl simulate          # Run config-driven simulation
    julia run.jl config            # Print current configuration
    julia run.jl init              # Initialize output directories
"""

# Activate project environment
using Pkg
Pkg.activate(@__DIR__)

using Dates  # Need this at top level for run_config_simulation
using CSV, DataFrames, JSON  # For saving outputs
using Plots  # Need for visualization

# Include framework in correct order (avoiding circular dependencies)
include("src/types.jl")
include("src/constants.jl")
include("src/agents/abstract_agent.jl")
include("src/environments/abstract_environment.jl")
include("src/config.jl")
include("src/diagnostics.jl")
include("src/logging.jl")
include("src/visualization.jl")
include("src/simulation.jl")

using .Main: StateVector, ActionVector, ObservationVector

# Parse command line arguments
function main(args::Vector{String})
    if isempty(args) || args[1] == "help" || args[1] == "--help"
        print_help()
        return
    end
    
    command = args[1]
    
    if command == "simulate"
        println("Running config-driven simulation...")
        config_path = length(args) > 1 ? args[2] : "config.toml"
        run_config_simulation(config_path)
        
    elseif command == "config"
        println("Loading configuration...")
        config_path = length(args) > 1 ? args[2] : "config.toml"
        config = load_config(config_path)
        print_config(config)
        
    elseif command == "init"
        println("Initializing output directories...")
        config_path = length(args) > 1 ? args[2] : "config.toml"
        config = load_config(config_path)
        dirs = ensure_output_directories(config["outputs"])
        println("Created/verified directories:")
        for dir in dirs
            println("  ✓ $dir")
        end
        println("\nOutput structure ready!")
        
    else
        println("Unknown command: $command")
        print_help()
        exit(1)
    end
end

function run_config_simulation(config_path::String)
    """Run simulation using configuration file"""
    
    # Load configuration
    config = load_config(config_path)
    
    # Create timestamped run directory
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    agent_type = replace(config["agent"]["type"], "Agent" => "")
    env_type = replace(config["environment"]["type"], "Env" => "")
    run_name = "$(lowercase(agent_type))_$(lowercase(env_type))_$(timestamp)"
    run_dir = joinpath(config["outputs"]["base_dir"], run_name)
    
    # Create run-specific subdirectories
    mkpath(joinpath(run_dir, "logs"))
    mkpath(joinpath(run_dir, "data"))
    mkpath(joinpath(run_dir, "plots"))
    mkpath(joinpath(run_dir, "animations"))
    mkpath(joinpath(run_dir, "diagnostics"))
    mkpath(joinpath(run_dir, "results"))
    
    println("Run directory: $run_dir")
    
    # Create environment
    println("Creating environment: $(config["environment"]["type"])")
    env = create_environment_from_config(config["environment"])
    
    # Create agent
    println("Creating agent: $(config["agent"]["type"])")
    agent = create_agent_from_config(config["agent"], config["environment"], env)
    
    # Create simulation config
    sim_config = create_simulation_config_from_toml(config["simulation"])
    
    # Run simulation
    println("\nRunning simulation...")
    result = run_simulation(agent, env, sim_config)
    
    # Print summary
    println("\n" * "="^70)
    println("SIMULATION COMPLETE")
    println("="^70)
    println("Steps taken: $(result.steps_taken)")
    println("Total time: $(round(result.total_time, digits=3))s")
    println("Avg time per step: $(round(result.total_time / result.steps_taken, digits=4))s")
    println("Output directory: $run_dir")
    println("="^70)
    
    # Extract goal state if available
    goal_state = nothing
    if haskey(config["environment"], "goal_position")
        state_dim = length(result.states[1])
        if state_dim == 1
            goal_state = StateVector{1}([config["environment"]["goal_position"]])
        elseif state_dim == 2
            goal_pos = config["environment"]["goal_position"]
            goal_vel = get(config["environment"], "goal_velocity", 0.0)
            goal_state = StateVector{2}([goal_pos, goal_vel])
        end
    end
    
    # Save all outputs with comprehensive visualizations and animations
    save_simulation_outputs(
        result,
        run_dir,
        goal_state,
        generate_visualizations=true,
        generate_animations=true
    )
end

function print_config(config::Dict)
    """Print configuration in readable format"""
    
    println("\n" * "="^70)
    println("CONFIGURATION")
    println("="^70)
    
    for (section, values) in config
        println("\n[$section]")
        if isa(values, Dict)
            for (key, val) in values
                println("  $key = $val")
            end
        end
    end
    
    println("\n" * "="^70)
end

function print_help()
    println("""
    Generic Agent-Environment Framework
    ====================================
    
    Usage:
        julia run.jl [command] [options]
    
    Commands:
        simulate [config.toml]   Run config-driven simulation (default: config.toml)
        config [config.toml]     Print configuration file
        init [config.toml]       Initialize output directory structure
        help                     Show this help message
    
    Examples:
        julia run.jl simulate
        julia run.jl simulate my_config.toml
        julia run.jl config
        julia run.jl init
    
    Configuration:
        Edit config.toml to select:
        - Agent type (MountainCarAgent, SimpleNavAgent)
        - Environment type (MountainCarEnv, SimpleNavEnv)
        - Simulation parameters
        - Output directories
    
    Output Structure:
        All outputs are organized under outputs/ directory:
        - outputs/logs/          Log files
        - outputs/data/          Data exports
        - outputs/plots/         Visualizations
        - outputs/animations/    Animated plots
        - outputs/diagnostics/   Diagnostic reports
        - outputs/results/       Simulation results
    
    For more information, see README.md
    """)
end

# Run main if executed directly
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end

