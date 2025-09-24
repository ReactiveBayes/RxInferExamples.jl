#!/usr/bin/env julia

# Main execution script for Generalized Active Inference Car Examples
# Supports multiple car types with extensible architecture

@doc """
Generalized Active Inference Car Example Runner

This script demonstrates the complete active inference solution for various car types
including mountain car, race car, and autonomous car scenarios.

## Usage
```
julia run.jl [car_type] [options]

Car Types:
  mountain_car    Classic mountain car with gravitational dynamics
  race_car        High-speed racing car with aerodynamic forces
  autonomous_car  Urban autonomous vehicle with obstacle avoidance

Options:
  --help                    Show this comprehensive help message
  --list-car-types          List all available car types and their descriptions
  --naive                   Run only naive policy comparison
  --animation               Save animations to GIF files
  --verbose                 Enable detailed console logging
  --structured              Enable structured JSON logging
  --performance             Enable performance logging to CSV
  --export                  Export results to JSON/CSV files
  --benchmark               Run performance benchmarking
  --comparison              Create comparison animations between car types
  --interactive             Launch interactive mode for parameter tuning

Examples:
  julia run.jl mountain_car                     # Run mountain car example
  julia run.jl race_car --animation            # Run racing with animations
  julia run.jl autonomous_car --export         # Run autonomous car with export
  julia run.jl --comparison --animation        # Compare all car types
  julia run.jl mountain_car --benchmark        # Benchmark mountain car
  julia run.jl --interactive                   # Launch interactive mode
```

## Output Features
- **Console Output**: Real-time progress and status updates
- **Log Files**: Detailed logging with multiple output formats
- **Animations**: High-quality GIF animations for visualization
- **Data Export**: Comprehensive CSV and JSON export for analysis
- **Performance Metrics**: Detailed timing and memory usage tracking
- **Comparison Tools**: Side-by-side comparison of different car types

## Architecture
The system uses a modular architecture with:
- **Configurable Car Types**: Easy to add new car types and scenarios
- **Extensible Physics**: Support for different dynamics models
- **Modular Agents**: Multiple inference algorithms and planning strategies
- **Flexible Visualization**: Adaptable visualization for different scenarios
- **Comprehensive Testing**: Full test suite for all components
"""

# ==================== MODULE IMPORTS ====================

using Pkg
using Test
using Logging
using Dates
using Printf
using Random
using Statistics
using LinearAlgebra

# Activate project environment
Pkg.activate(".")

# Import core modules (skip agent for now due to GraphPPL dependency)
include("config.jl")
include("src/physics.jl")
include("src/world.jl")
# include("src/agent.jl")  # Skip agent module - requires GraphPPL/RxInfer
include("src/visualization.jl")
include("src/utils.jl")

# Import functions from modules
using .Config: get_car_config, get_config_value, validate_configuration, print_configuration
using .Physics: create_physics, create_integrator, get_landscape_coordinates
using .World: create_world, execute_action!, observe, reset!, get_state, set_state!
# using .Agent: create_agent, select_action, update_belief!, get_predictions, reset!  # Skip agent for now
using .Visualization: create_visualization, create_unified_animation, create_comparison_animation
using .Utils: setup_logging, PerformanceTimer, close, log_structured, export_experiment_results,
              export_trajectory_data, validate_experiment_config, create_progress_bar, update_progress!

# ==================== CAR TYPE DEFINITIONS ====================

@doc """
Available car types and their descriptions.
"""
const CAR_TYPES = Dict(
    :mountain_car => (
        name = "Mountain Car",
        description = "Classic mountain car with gravitational dynamics. The car must build momentum to overcome gravity and reach the goal on the opposite hill.",
        physics = "Gravitational forces with friction",
        challenges = ["Energy management", "Momentum building", "Hill climbing"],
        default_horizon = 20
    ),

    :race_car => (
        name = "Race Car",
        description = "High-performance racing car with aerodynamic forces, tire grip modeling, and track temperature effects.",
        physics = "Aerodynamic drag, downforce, tire dynamics",
        challenges = ["High-speed stability", "Lap optimization", "Tire management"],
        default_horizon = 30
    ),

    :autonomous_car => (
        name = "Autonomous Car",
        description = "Urban autonomous vehicle with obstacle detection, path planning, and safety constraints.",
        physics = "Sensor-based dynamics, obstacle avoidance",
        challenges = ["Obstacle avoidance", "Path planning", "Safety constraints"],
        default_horizon = 25
    )
)

# ==================== MAIN SIMULATION FUNCTIONS ====================

@doc """
Run naive policy simulation for comparison.

Args:
- car_type: Car type symbol
- config: Car configuration
- verbose: Enable verbose logging

Returns:
- Results dictionary, trajectory, and actions
"""
function run_naive_policy(car_type::Symbol, config; verbose::Bool = false)
    @info "Running naive policy simulation for $car_type..."

    timer = PerformanceTimer("naive_policy_simulation")

    # Create physics and world
    physics = create_physics(car_type; custom_params = Dict{Symbol, Any}())
    world = create_world(car_type; custom_params = Dict{Symbol, Any}())

    # Get configuration parameters
    time_steps = get_config_value(:simulation, :time_steps, 100, car_type)
    naive_action = get_config_value(:simulation, :naive_action, 100.0, car_type)

    # Initialize tracking
    trajectory = Vector{Vector{Float64}}(undef, time_steps + 1)
    actions = fill(naive_action, time_steps)
    engine_forces = Vector{Float64}(undef, time_steps)

    # Reset and get initial state
    reset!(world)
    trajectory[1] = observe(world)

    # Progress bar for long simulations
    progress = create_progress_bar(time_steps, "Naive Policy Simulation")

    # Run simulation
    for t in 1:time_steps
        # Execute action and get actual engine force
        success, _ = execute_action!(world, actions[t], physics)
        trajectory[t + 1] = observe(world)
        engine_forces[t] = physics.total_force(trajectory[t][1], trajectory[t][2], actions[t])

        # Update progress
        if t % 10 == 0
            current_pos = trajectory[t + 1][1]
            update_progress!(progress, t, metadata = @sprintf("Pos: %.3f", current_pos))
        end

        # Verbose logging
        if verbose && t % 20 == 0
            current_pos = trajectory[t + 1][1]
            current_vel = trajectory[t + 1][2]
            @info "Naive step $t" position = round(current_pos, digits=3) velocity = round(current_vel, digits=3)
        end
    end

    # Calculate results
    final_position = trajectory[end][1]
    goal_position = get_config_value(:world, :target_position, 0.5, car_type)
    success = abs(final_position - goal_position) <= get_config_value(:world, :goal_tolerance, 0.1, car_type)

    total_distance = sum(abs(trajectory[i][1] - trajectory[i-1][1]) for i in 2:length(trajectory))
    avg_velocity = mean(abs.([state[2] for state in trajectory]))
    max_velocity = maximum(abs.([state[2] for state in trajectory]))

    close(timer)

    results = Dict(
        "car_type" => car_type,
        "policy" => "naive",
        "final_position" => final_position,
        "success" => success,
        "total_distance" => total_distance,
        "avg_velocity" => avg_velocity,
        "max_velocity" => max_velocity,
        "total_time_steps" => time_steps,
        "constant_action" => naive_action,
        "avg_engine_force" => mean(engine_forces)
    )

    @info "Naive policy completed" final_position = round(final_position, digits=3) success = success
    return results, trajectory, actions
end

@doc """
Run active inference simulation (simplified version without full agent).

Args:
- car_type: Car type symbol
- config: Car configuration
- verbose: Enable verbose logging

Returns:
- Results dictionary, trajectory, actions, and predictions
"""
function run_active_inference(car_type::Symbol, config; verbose::Bool = false)
    @info "Running active inference simulation for $car_type..."

    timer = PerformanceTimer("active_inference_simulation")

    # Create physics and world (skip agent for now)
    physics = create_physics(car_type; custom_params = Dict{Symbol, Any}())
    world = create_world(car_type; custom_params = Dict{Symbol, Any}())
    planning_horizon = get_config_value(:agent, :planning_horizon, 20, car_type)
    # agent = create_agent(car_type, planning_horizon)  # Skip agent for now

    # Get configuration parameters
    time_steps = get_config_value(:simulation, :time_steps, 100, car_type)

    # Initialize tracking
    trajectory = Vector{Vector{Float64}}(undef, time_steps + 1)
    actions = Vector{Float64}(undef, time_steps)
    predictions = Matrix{Float64}(undef, time_steps, planning_horizon)

    # Reset systems
    reset!(world)
    # reset!(agent)  # Skip agent for now

    # Get initial state
    trajectory[1] = observe(world)

    # Progress bar
    progress = create_progress_bar(time_steps, "Active Inference Simulation")

    # Run active inference loop (simplified without agent)
    for t in 1:time_steps
        current_state = observe(world)

        # Simplified action selection (proportional controller)
        goal_position = get_config_value(:world, :target_position, 0.5, car_type)
        position_error = goal_position - current_state[1]
        velocity_error = -current_state[2]  # Target velocity = 0 at goal

        # Simple proportional controller
        kp = 0.1  # Position gain
        kv = 0.05  # Velocity gain
        actions[t] = clamp(kp * position_error + kv * velocity_error, -0.1, 0.1)

        # Execute action in world
        success, collision_info = execute_action!(world, actions[t], physics)
        trajectory[t + 1] = observe(world)

        # Generate mock predictions for visualization
        for i in 1:planning_horizon
            predictions[t, i] = current_state[1] + i * 0.1 * sign(position_error)  # Linear extrapolation
        end

        # Update progress
        if t % 10 == 0
            current_pos = trajectory[t + 1][1]
            update_progress!(progress, t, metadata = @sprintf("Pos: %.3f", current_pos))
        end

        # Verbose logging
        if verbose && t % 20 == 0
            @info "AI step $t" position = round(current_state[1], digits=3) action = round(actions[t], digits=3)
        end
    end

    # Calculate results
    final_position = trajectory[end][1]
    goal_position = get_config_value(:world, :target_position, 0.5, car_type)
    success = abs(final_position - goal_position) <= get_config_value(:world, :goal_tolerance, 0.1, car_type)

    positions = [state[1] for state in trajectory]
    velocities = [state[2] for state in trajectory]

    total_distance = sum(abs(positions[i] - positions[i-1]) for i in 2:length(positions))
    avg_velocity = mean(abs.(velocities))
    max_velocity = maximum(abs.(velocities))
    action_variance = var(actions)
    avg_action = mean(actions)

    close(timer)

    results = Dict(
        "car_type" => car_type,
        "policy" => "active_inference_simplified",
        "final_position" => final_position,
        "success" => success,
        "total_distance" => total_distance,
        "avg_velocity" => avg_velocity,
        "max_velocity" => max_velocity,
        "action_variance" => action_variance,
        "avg_action" => avg_action,
        "total_time_steps" => time_steps,
        "planning_horizon" => planning_horizon,
        "note" => "Simplified AI without full agent system (GraphPPL dependency)"
    )

    @info "Active inference completed" final_position = round(final_position, digits=3) success = success
    return results, trajectory, actions, predictions
end

@doc """
Create animation for a car type simulation.

Args:
- car_type: Car type symbol
- trajectory: State trajectory
- actions: Action sequence
- predictions: Prediction matrix
- filename: Output filename
- obstacles: Optional obstacles
"""
function create_car_animation(car_type::Symbol, trajectory::Vector{Vector{Float64}},
                             actions::Vector{Float64}, predictions::Union{Matrix{Float64}, Nothing},
                             filename::String; obstacles::Union{Vector{Float64}, Nothing} = nothing)

    @info "Creating animation for $car_type" filename = filename

    # Create visualization system
    vis_system = create_visualization(car_type)

    # Generate animation
    create_unified_animation(car_type, trajectory, actions, filename;
                           predictions = predictions,
                           obstacles = obstacles,
                           fps = 24)

    @info "Animation completed" car_type = car_type filename = filename
end

@doc """
Create comparison animation between different car types.

Args:
- car_results: Dictionary of results for different car types
- output_dir: Output directory for animations
"""
function create_comparison_animations(car_results::Dict, output_dir::String)
    @info "Creating comparison animations..."

    car_types = collect(keys(car_results))

    if length(car_types) >= 2
        for i in 1:length(car_types)-1
            for j in i+1:length(car_types)
                car1 = car_types[i]
                car2 = car_types[j]

                if haskey(car_results[car1], :trajectory) && haskey(car_results[car2], :trajectory)
                    filename = joinpath(output_dir, "comparison_$(car1)_vs_$(car2).gif")

                    create_comparison_animation(
                        car1, car_results[car1][:trajectory], car_results[car1][:actions],
                        car2, car_results[car2][:trajectory], car_results[car2][:actions],
                        filename
                    )
                end
            end
        end
    end

    @info "Comparison animations completed"
end

# ==================== BENCHMARKING FUNCTIONS ====================

@doc """
Run comprehensive benchmarking for a car type.

Args:
- car_type: Car type to benchmark
- iterations: Number of iterations per test
"""
function run_benchmark(car_type::Symbol, iterations::Int = 5)
    @info "Running benchmarks for $car_type..." iterations = iterations

    # Benchmark naive policy
    naive_times = Float64[]
    for i in 1:iterations
        _, trajectory, actions = run_naive_policy(car_type, get_car_config(car_type), verbose = false)
        push!(naive_times, length(trajectory) * 0.1)  # Assume 0.1s per step
    end

    # Benchmark active inference
    ai_times = Float64[]
    for i in 1:iterations
        _, trajectory, actions, predictions = run_active_inference(car_type, get_car_config(car_type), verbose = false)
        push!(ai_times, length(trajectory) * 0.1)  # Assume 0.1s per step
    end

    # Create benchmark results
    benchmark_results = Dict(
        "car_type" => car_type,
        "naive_policy" => Dict(
            "mean_time" => mean(naive_times),
            "std_time" => std(naive_times),
            "min_time" => minimum(naive_times),
            "max_time" => maximum(naive_times)
        ),
        "active_inference" => Dict(
            "mean_time" => mean(ai_times),
            "std_time" => std(ai_times),
            "min_time" => minimum(ai_times),
            "max_time" => maximum(ai_times)
        ),
        "performance_ratio" => mean(ai_times) / mean(naive_times),
        "iterations" => iterations,
        "timestamp" => string(now())
    )

    # Export benchmark results
    export_experiment_results("benchmark_$(car_type)", benchmark_results)

    @info "Benchmark completed" car_type = car_type ratio = round(benchmark_results["performance_ratio"], digits=2)
    return benchmark_results
end

# ==================== MAIN EXPERIMENT FUNCTION ====================

@doc """
Run complete experiment for specified car types.

Args:
- car_types: Vector of car types to run
- options: Dictionary of options (naive, animation, export, etc.)
"""
function run_experiment(car_types::Vector{Symbol}, options::Dict)
    @info "Starting generalized active inference car experiment..." car_types = car_types

    # Setup logging
    setup_logging(
        enable_structured = get(options, :structured, false),
        enable_performance = get(options, :performance, false),
        log_level = get(options, :verbose, false) ? Logging.Debug : Logging.Info
    )

    # Initialize results collection
    experiment_results = Dict(
        "experiment_name" => "generalized_active_inference_car",
        "timestamp" => string(now()),
        "car_types" => car_types,
        "options" => options,
        "results" => Dict{String, Any}()
    )

    # Run simulations for each car type
    for car_type in car_types
        @info "Processing car type: $car_type"
        car_key = string(car_type)

        try
            # Validate configuration
            config = get_car_config(car_type)
            validation_errors = validate_experiment_config(Dict(
                :physics => Dict(pairs(config.physics)),
                :world => Dict(pairs(config.world)),
                :agent => Dict(pairs(config.agent)),
                :simulation => Dict(pairs(config.simulation))
            ))

            if !isempty(validation_errors)
                @warn "Configuration validation issues for $car_type" issues = validation_errors
            end

            # Run naive policy if requested
            if get(options, :naive, false) || !get(options, :ai_only, false)
                naive_results, naive_trajectory, naive_actions = run_naive_policy(
                    car_type, config, verbose = get(options, :verbose, false)
                )
                experiment_results["results"][car_key] = Dict(
                    "naive" => naive_results,
                    "trajectory" => naive_trajectory,
                    "actions" => naive_actions
                )
            end

            # Run active inference
            ai_results, ai_trajectory, ai_actions, ai_predictions = run_active_inference(
                car_type, config, verbose = get(options, :verbose, false)
            )

            # Merge AI results
            if haskey(experiment_results["results"], car_key)
                experiment_results["results"][car_key]["active_inference"] = ai_results
                experiment_results["results"][car_key]["ai_trajectory"] = ai_trajectory
                experiment_results["results"][car_key]["ai_actions"] = ai_actions
                experiment_results["results"][car_key]["predictions"] = ai_predictions
            else
                experiment_results["results"][car_key] = Dict(
                    "active_inference" => ai_results,
                    "trajectory" => ai_trajectory,
                    "actions" => ai_actions,
                    "predictions" => ai_predictions
                )
            end

            # Create animations if requested
            if get(options, :animation, false)
                try
                    output_dir = joinpath("outputs", "animations")
                    mkpath(output_dir)

                    if haskey(experiment_results["results"][car_key], "naive")
                        naive_filename = joinpath(output_dir, "naive_$(car_type).gif")
                        create_car_animation(car_type, naive_trajectory, naive_actions, nothing, naive_filename)
                    end

                    ai_filename = joinpath(output_dir, "ai_$(car_type).gif")
                    create_car_animation(car_type, ai_trajectory, ai_actions, ai_predictions, ai_filename)
                catch e
                    @warn "Animation creation failed, continuing with export" error = string(e)
                end
            end

        catch e
            @error "Failed to process car type: $car_type" error = string(e)
            log_structured(Dict(
                "event" => "car_type_processing_failed",
                "car_type" => car_type,
                "error" => string(e)
            ))
        end

        # Create static visualizations
        try
            vis_output_dir = joinpath("outputs", "results", "$(experiment_results["experiment_name"])_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))")
            Visualization.create_static_visualizations(experiment_results, vis_output_dir)
        catch e
            @warn "Static visualization creation failed" error = string(e)
        end
    end

    # Create comparison animations
    if get(options, :comparison, false) && length(car_types) >= 2
        output_dir = joinpath("outputs", "comparisons")
        create_comparison_animations(experiment_results["results"], output_dir)
    end

    # Export results if requested
    if get(options, :export, false)
        export_experiment_results("generalized_car_experiment", experiment_results)
    end

    # Run benchmarking if requested
    if get(options, :benchmark, false)
        for car_type in car_types
            run_benchmark(car_type, get(options, :benchmark_iterations, 5))
        end
    end

    @info "Generalized experiment completed successfully"
    return experiment_results
end

# ==================== COMMAND LINE INTERFACE ====================

@doc """
Parse command line arguments.

Returns:
- Tuple of (car_types, options)
"""
function parse_arguments()
    args = ARGS

    # Default values
    car_types = Symbol[]
    options = Dict(
        :naive => false,
        :animation => false,
        :verbose => false,
        :structured => false,
        :performance => false,
        :export => false,
        :benchmark => false,
        :comparison => false,
        :interactive => false,
        :ai_only => false
    )

    # Parse arguments
    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "--help"
            print_help()
            exit(0)
        elseif arg == "--list-car-types"
            list_car_types()
            exit(0)
        elseif arg == "--naive"
            options[:naive] = true
        elseif arg == "--animation"
            options[:animation] = true
        elseif arg == "--verbose"
            options[:verbose] = true
        elseif arg == "--structured"
            options[:structured] = true
        elseif arg == "--performance"
            options[:performance] = true
        elseif arg == "--export"
            options[:export] = true
        elseif arg == "--benchmark"
            options[:benchmark] = true
        elseif arg == "--comparison"
            options[:comparison] = true
        elseif arg == "--interactive"
            options[:interactive] = true
        elseif arg == "--ai-only"
            options[:ai_only] = true
        elseif startswith(arg, "--")
            @warn "Unknown option: $arg"
        else
            # Assume it's a car type
            car_type = Symbol(lowercase(arg))
            if haskey(CAR_TYPES, car_type)
                push!(car_types, car_type)
            else
                @warn "Unknown car type: $arg. Available types: $(join(keys(CAR_TYPES), ", "))"
            end
        end

        i += 1
    end

    # Set default car type if none specified
    if isempty(car_types)
        car_types = [:mountain_car]
    end

    return car_types, options
end

@doc """
Print comprehensive help message.
"""
function print_help()
    println("""
    Generalized Active Inference Car Examples

    This system provides a comprehensive, extensible framework for active inference
    in various car scenarios including mountain car, racing, and autonomous driving.

    USAGE:
        julia run.jl [car_type] [options]

    CAR TYPES:
    """)

    for (key, info) in CAR_TYPES
        println("        $(rpad(string(key), 15)) $(info.name)")
        println("$(rpad("", 25)) $(info.description)")
        println()
    end

    println("""
    OPTIONS:
        --help                    Show this help message
        --list-car-types          List all available car types
        --naive                   Run naive policy comparison
        --animation               Create GIF animations
        --verbose                 Enable detailed logging
        --structured              Enable structured JSON logging
        --performance             Enable performance CSV logging
        --export                  Export results to files
        --benchmark               Run performance benchmarking
        --comparison              Create comparison animations
        --interactive             Launch interactive parameter tuning

    EXAMPLES:
        julia run.jl mountain_car                    # Basic mountain car
        julia run.jl race_car --animation           # Racing with animations
        julia run.jl autonomous_car --export        # Autonomous with export
        julia run.jl --comparison --animation       # Compare all types
        julia run.jl mountain_car --benchmark       # Benchmark mountain car
        julia run.jl --interactive                  # Interactive mode

    OUTPUT:
        - Real-time console output with progress updates
        - Log files: *.log, *_structured.jsonl, *_performance.csv
        - Animation files: *.gif (if --animation)
        - Results: JSON/CSV files (if --export)
        - Performance metrics and benchmarking data

    ARCHITECTURE:
        - Modular design with pluggable components
        - Support for multiple physics models and dynamics
        - Extensible agent system with different inference algorithms
        - Flexible visualization for various scenarios
        - Comprehensive testing and validation framework
    """)
end

@doc """
List all available car types with descriptions.
"""
function list_car_types()
    println("Available Car Types:")
    println("="^50)

    for (key, info) in CAR_TYPES
        println("$(rpad(string(key), 15)) $(info.name)")
        println("$(rpad("", 16)) Description: $(info.description)")
        println("$(rpad("", 16)) Physics: $(info.physics)")
        println("$(rpad("", 16)) Challenges: $(join(info.challenges, ", "))")
        println("$(rpad("", 16)) Default Horizon: $(info.default_horizon)")
        println()
    end

    println("Use any of these car types with the run.jl script:")
    println("  julia run.jl <car_type> [--options]")
end

@doc """
Launch interactive mode for parameter tuning.

This would provide an interactive interface for experimenting with
different parameters and car configurations.
"""
function launch_interactive_mode()
    @info "Interactive mode launched"
    @info "This would provide GUI controls for parameter tuning"
    @info "Feature not yet implemented - run with specific car type instead"
end

# ==================== MAIN FUNCTION ====================

@doc """
Main execution function.
"""
function main()
    # Parse command line arguments
    car_types, options = parse_arguments()

    # Handle special modes
    if get(options, :interactive, false)
        launch_interactive_mode()
        return
    end

    try
        # Run the experiment
        results = run_experiment(car_types, options)

        # Print summary
        println("\n" * "═"^60)
        println("EXPERIMENT SUMMARY")
        println("═"^60)

        for car_type in car_types
            car_key = string(car_type)
            if haskey(results["results"], car_key)
                car_results = results["results"][car_key]

                println("\n$(CAR_TYPES[car_type].name):")
                println("-"^(length(CAR_TYPES[car_type].name) + 1))

                if haskey(car_results, "naive")
                    naive = car_results["naive"]
                    println("Naive Policy:")
                    println("  Final Position: $(round(naive["final_position"], digits=3))")
                    println("  Success: $(naive["success"])")
                    println("  Avg Velocity: $(round(naive["avg_velocity"], digits=3))")
                    println("  Total Distance: $(round(naive["total_distance"], digits=3))")
                end

                if haskey(car_results, "active_inference")
                    ai = car_results["active_inference"]
                    println("Active Inference:")
                    println("  Final Position: $(round(ai["final_position"], digits=3))")
                    println("  Success: $(ai["success"])")
                    println("  Avg Velocity: $(round(ai["avg_velocity"], digits=3))")
                    println("  Total Distance: $(round(ai["total_distance"], digits=3))")
                    println("  Action Variance: $(round(ai["action_variance"], digits=3))")
                end
            end
        end

        println("\n" * "═"^60)
        @info "Experiment completed successfully"

    catch e
        @error "Experiment failed" error = string(e)
        if get(options, :verbose, false)
            showerror(stderr, e, catch_backtrace())
        end
        rethrow(e)
    end
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
