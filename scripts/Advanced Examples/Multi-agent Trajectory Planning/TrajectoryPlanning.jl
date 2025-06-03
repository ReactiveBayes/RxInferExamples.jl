module TrajectoryPlanning

using LinearAlgebra
using RxInfer
using Plots
using LogExpFunctions
using StableRNGs
using Dates

export run_all_experiments

#
# Environment structures and functions
#

# A simple struct to represent a rectangle, which is defined by its center (x, y) and size (width, height)
Base.@kwdef struct Rectangle
    center::Tuple{Float64, Float64}
    size::Tuple{Float64, Float64}
end

# A simple struct to represent an environment, which is defined by a list of obstacles,
# and in this demo the obstacles are just rectangles
Base.@kwdef struct Environment
    obstacles::Vector{Rectangle}
end

# Agent plan, encodes start and goal states
Base.@kwdef struct Agent
    radius::Float64
    initial_position::Tuple{Float64, Float64}
    target_position::Tuple{Float64, Float64}
end

#
# Visualization functions
#

function plot_rectangle!(p, rect::Rectangle)
    # Calculate the x-coordinates of the four corners
    x_coords = rect.center[1] .+ rect.size[1]/2 * [-1, 1, 1, -1, -1]
    # Calculate the y-coordinates of the four corners
    y_coords = rect.center[2] .+ rect.size[2]/2 * [-1, -1, 1, 1, -1]
    
    # Plot the rectangle with a black fill
    plot!(p, Shape(x_coords, y_coords), 
          label = "", 
          color = :black, 
          alpha = 0.5,
          linewidth = 1.5,
          fillalpha = 0.3)
end

function plot_environment!(p, env::Environment)
    for obstacle in env.obstacles
        plot_rectangle!(p, obstacle)
    end
    return p
end

function plot_environment(env::Environment)
    p = plot(size = (800, 400), xlims = (-20, 20), ylims = (-20, 20), aspect_ratio = :equal)
    plot_environment!(p, env)
    return p
end

function plot_marker_at_position!(p, radius, position; color="red", markersize=10.0, alpha=1.0, label="")
    # Draw the agent as a circle with the given radius
    θ = range(0, 2π, 100)
    
    x_coords = position[1] .+ radius .* cos.(θ)
    y_coords = position[2] .+ radius .* sin.(θ)
    
    plot!(p, Shape(x_coords, y_coords); color=color, label=label, alpha=alpha)
    return p
end

function plot_agent_naive_plan!(p, agent; color = "blue")
    plot_marker_at_position!(p, agent.radius, agent.initial_position, color = color)
    plot_marker_at_position!(p, agent.radius, agent.target_position, color = color, alpha = 0.1)
    quiver!(p, [ agent.initial_position[1] ], [ agent.initial_position[2] ], 
            quiver = ([ agent.target_position[1] - agent.initial_position[1] ], 
                    [ agent.target_position[2] -  agent.initial_position[2] ]))
end

function plot_agent_plans!(p, agents)
    colors = Plots.palette(:tab10)
    
    for (k, agent) in enumerate(agents)
        plot_agent_naive_plan!(p, agent, color = colors[k])
    end
    
    return p
end

function animate_paths(environment, agents, paths; output_dir=".", filename = "result.gif", fps = 15, logger=nothing)
    nr_agents, nr_steps = size(paths)
    colors = Plots.palette(:tab10)

    log_message("Generating animation frames...", logger)
    animation = @animate for t in 1:nr_steps
        frame = plot_environment(environment)
    
        for k in 1:nr_agents
            position = paths[k, t]          
            path = paths[k, 1:t]
            
            plot_marker_at_position!(frame, agents[k].radius, position, color = colors[k])
            plot_marker_at_position!(frame, agents[k].radius, agents[k].target_position, color = colors[k], alpha = 0.2)
            plot!(frame, getindex.(path, 1), getindex.(path, 2); linestyle=:dash, label="", color=colors[k])
        end

        frame
    end

    # Save the animation
    output_path = joinpath(output_dir, filename)
    log_message("Saving animation to $output_path...", logger)
    gif(animation, output_path, fps=fps, show_msg = false)
    log_message("Animation saved successfully.", logger)
    
    return nothing
end

#
# Model functions
#

# Define the probabilistic model for obstacles using halfspace constraints
struct Halfspace end

@node Halfspace Stochastic [out, a, σ2, γ]

# rule specification
@rule Halfspace(:out, Marginalisation) (q_a::Any, q_σ2::Any, q_γ::Any) = begin
    return NormalMeanVariance(mean(q_a) + mean(q_γ) * mean(q_σ2), mean(q_σ2))
end

@rule Halfspace(:σ2, Marginalisation) (q_out::Any, q_a::Any, q_γ::Any, ) = begin
    # `BayesBase.TerminalProdArgument` is used to ensure that the result of the posterior computation is equal to this value
    return BayesBase.TerminalProdArgument(PointMass( 1 / mean(q_γ) * sqrt(abs2(mean(q_out) - mean(q_a)) + var(q_out))))
end

softmin(x; l=10) = -logsumexp(-l .* x) / l

# state here is a 4-dimensional vector [x, y, vx, vy]
function distance(r::Rectangle, state)
    if abs(state[1] - r.center[1]) > r.size[1] / 2 || abs(state[2] - r.center[2]) > r.size[2] / 2
        # outside of rectangle
        dx = max(abs(state[1] - r.center[1]) - r.size[1] / 2, 0)
        dy = max(abs(state[2] - r.center[2]) - r.size[2] / 2, 0)
        return sqrt(dx^2 + dy^2)
    else
        # inside rectangle
        return max(abs(state[1] - r.center[1]) - r.size[1] / 2, abs(state[2] - r.center[2]) - r.size[2] / 2)
    end
end

function distance(env::Environment, state)
    return softmin([distance(obstacle, state) for obstacle in env.obstacles])
end

# Helper function, distance with radius offset
function g(environment, radius, state)
    return distance(environment, state) - radius
end

# Helper function, finds minimum distances between agents pairwise
function h(environment, radiuses, states...)
    # Calculate pairwise distances between all agents
    distances = Real[]
    n = length(states)

    for i in 1:n
        for j in (i+1):n
            push!(distances, norm(states[i] - states[j]) - radiuses[i] - radiuses[j])
        end
    end

    return softmin(distances)
end

# For more details about the model, please refer to the original paper
@model function path_planning_model(environment, agents, goals, nr_steps)

    # Model's parameters are fixed, refer to the original 
    # paper's implementation for more details about these parameters
    local dt = 1
    local A  = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
    local B  = [0 0; dt 0; 0 0; 0 dt]
    local C  = [1 0 0 0; 0 0 1 0]
    local γ  = 1

    local control
    local state
    local path   
    
    # Extract radiuses of each agent in a separate collection
    local rs = map((a) -> a.radius, agents)

    # Model is fixed for 4 agents
    for k in 1:4

        # Prior on state, the state structure is 4 dimensional, where
        # [ x_position, x_velocity, y_position, y_velocity ]
        state[k, 1] ~ MvNormal(mean = zeros(4), covariance = 1e2I)

        for t in 1:nr_steps

            # Prior on controls
            control[k, t] ~ MvNormal(mean = zeros(2), covariance = 1e-1I)

            # State transition
            state[k, t+1] ~ A * state[k, t] + B * control[k, t]

            # Path model, the path structure is 2 dimensional, where 
            # [ x_position, y_position ]
            path[k, t] ~ C * state[k, t+1]

            # Environmental distance
            zσ2[k, t] ~ GammaShapeRate(3 / 2, γ^2 / 2)
            z[k, t]   ~ g(environment, rs[k], path[k, t])
            
            # Halfspase priors were defined previousle in this experiment
            z[k, t] ~ Halfspace(0, zσ2[k, t], γ)

        end

        # goal priors (indexing reverse due to definition)
        goals[1, k] ~ MvNormal(mean = state[k, 1], covariance = 1e-5I)
        goals[2, k] ~ MvNormal(mean = state[k, nr_steps+1], covariance = 1e-5I)

    end

    for t = 1:nr_steps

        # observation constraint
        dσ2[t] ~ GammaShapeRate(3 / 2, γ^2 / 2)
        d[t] ~ h(environment, rs, path[1, t], path[2, t], path[3, t], path[4, t])
        d[t] ~ Halfspace(0, dσ2[t], γ)

    end

end

@constraints function path_planning_constraints()
    # Mean-field variational constraints on the parameters
    q(d, dσ2) = q(d)q(dσ2)
    q(z, zσ2) = q(z)q(zσ2)
end

function path_planning(; environment, agents, nr_iterations = 350, nr_steps = 40, seed = 42, logger=nothing)
    log_message("Starting path planning inference...", logger)
    # Fixed number of agents
    nr_agents = 4

    # Form goals compatible with the model
    goals = hcat(
        map(agents) do agent
            return [
                [ agent.initial_position[1], 0, agent.initial_position[2], 0 ],
                [ agent.target_position[1], 0, agent.target_position[2], 0 ]
            ]
        end...
    )
    
    rng = StableRNG(seed)
    
    # Initialize variables, more details about initialization 
    # can be found in the original paper
    init = @initialization begin

        q(dσ2) = repeat([PointMass(1)], nr_steps)
        q(zσ2) = repeat([PointMass(1)], nr_agents, nr_steps)
        q(control) = repeat([PointMass(0)], nr_steps)

        μ(state) = MvNormalMeanCovariance(randn(rng, 4), 100I)
        μ(path) = MvNormalMeanCovariance(randn(rng, 2), 100I)

    end

    # Define approximation methods for the non-linear functions used in the model
    # `Linearization` is a simple and fast approximation method, but it is not
    # the most accurate one. For more details about the approximation methods,
    # please refer to the RxInfer documentation
    door_meta = @meta begin 
        h() -> Linearization()
        g() -> Linearization()
    end

    log_message("Running inference with $(nr_iterations) iterations...", logger)
    results = infer(
        model 			= path_planning_model(environment = environment, agents = agents, nr_steps = nr_steps),
        data  			= (goals = goals, ),
        initialization  = init,
        constraints 	= path_planning_constraints(),
        meta 			= door_meta,
        iterations 		= nr_iterations,
        returnvars 		= KeepLast(), 
        options         = (limit_stack_depth = 300, )
    )
    log_message("Inference completed.", logger)
    return results
end

#
# Logging and output management
#

# Utility function to create timestamp-based directory
function create_timestamped_dir(base_dir="results")
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    output_dir = joinpath(base_dir, timestamp)
    mkpath(output_dir)
    return output_dir
end

# Simple logger to both console and file
mutable struct DualLogger
    log_file::Union{IOStream, Nothing}
    
    function DualLogger(log_path)
        mkpath(dirname(log_path))
        log_file = open(log_path, "w")
        new(log_file)
    end
end

function close_logger(logger::DualLogger)
    if logger.log_file !== nothing
        close(logger.log_file)
        logger.log_file = nothing
    end
end

function log_message(message, logger::Union{DualLogger, Nothing}=nothing)
    # Always print to console
    println(message)
    
    # If logger is provided, also write to log file
    if logger !== nothing && logger.log_file !== nothing
        println(logger.log_file, message)
        flush(logger.log_file)
    end
end

#
# Experiment functions
#

# Create standard environments
function create_door_environment()
    return Environment(obstacles = [
        Rectangle(center = (-40, 0), size = (70, 5)),
        Rectangle(center = (40, 0), size = (70, 5))
    ])
end

function create_wall_environment()
    return Environment(obstacles = [
        Rectangle(center = (0, 0), size = (10, 5))
    ])
end

function create_combined_environment()
    return Environment(obstacles = [
        Rectangle(center = (-50, 0), size = (70, 2)),
        Rectangle(center = (50, -0), size = (70, 2)),
        Rectangle(center = (5, -1), size = (3, 10))
    ])
end

# Create standard agents
function create_standard_agents()
    return [
        Agent(radius = 2.5, initial_position = (-4, 10), target_position = (-10, -10)),
        Agent(radius = 1.5, initial_position = (-10, 5), target_position = (10, -15)),
        Agent(radius = 1.0, initial_position = (-15, -10), target_position = (10, 10)),
        Agent(radius = 2.5, initial_position = (0, -10), target_position = (-10, 15))
    ]
end

function execute_and_save_animation(environment, agents; output_dir=".", gifname = "result.gif", logger=nothing, kwargs...)
    log_message("Planning paths for environment with $(length(environment.obstacles)) obstacles...", logger)
    result = path_planning(environment = environment, agents = agents; logger=logger, kwargs...)
    paths = mean.(result.posteriors[:path])
    
    # Create animation and save it
    animate_paths(environment, agents, paths; output_dir=output_dir, filename=gifname, logger=logger)
    
    return paths
end

function run_all_experiments()
    # Create a timestamped directory for outputs
    output_dir = create_timestamped_dir()
    log_message("Created output directory: $output_dir")
    
    # Create a logger that writes to both console and file
    log_file = joinpath(output_dir, "experiment.log")
    logger = DualLogger(log_file)
    
    log_message("Starting Multi-agent Trajectory Planning experiments...", logger)
    
    # Create environments
    door_environment = create_door_environment()
    wall_environment = create_wall_environment()
    combined_environment = create_combined_environment()
    
    # Create agents
    agents = create_standard_agents()
    
    try
        log_message("Running experiments for door environment...", logger)
        execute_and_save_animation(door_environment, agents; 
                                  output_dir=output_dir, 
                                  gifname="door_42.gif", 
                                  seed=42, 
                                  logger=logger)

        log_message("Running experiments with different seed...", logger)
        execute_and_save_animation(door_environment, agents; 
                                  output_dir=output_dir, 
                                  gifname="door_123.gif", 
                                  seed=123, 
                                  logger=logger)

        log_message("Running experiments for wall environment...", logger)
        execute_and_save_animation(wall_environment, agents; 
                                  output_dir=output_dir, 
                                  gifname="wall_42.gif", 
                                  seed=42, 
                                  logger=logger)

        log_message("Running experiments with different seed...", logger)
        execute_and_save_animation(wall_environment, agents; 
                                  output_dir=output_dir, 
                                  gifname="wall_123.gif", 
                                  seed=123, 
                                  logger=logger)

        log_message("Running experiments for combined environment...", logger)
        execute_and_save_animation(combined_environment, agents; 
                                  output_dir=output_dir, 
                                  gifname="combined_42.gif", 
                                  seed=42, 
                                  logger=logger)

        log_message("Running final experiment...", logger)
        execute_and_save_animation(combined_environment, agents; 
                                  output_dir=output_dir, 
                                  gifname="combined_123.gif", 
                                  seed=123, 
                                  logger=logger)

        log_message("All experiments completed successfully.", logger)
        log_message("Results saved to: $output_dir", logger)
    finally
        # Close the logger to ensure the file is properly closed
        close_logger(logger)
    end
    
    return output_dir
end

end # module TrajectoryPlanning 