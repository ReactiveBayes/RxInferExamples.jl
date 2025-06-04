module ConfigLoader

using TOML
using LinearAlgebra

export load_config, apply_config

"""
    load_config(config_path="config.toml")

Load configuration from a TOML file.

# Arguments
- `config_path`: Path to the configuration file (default: "config.toml")

# Returns
- Dictionary containing all configuration parameters
"""
function load_config(config_path="config.toml")
    if !isfile(config_path)
        error("Configuration file not found: $config_path")
    end
    
    return TOML.parsefile(config_path)
end

"""
    get_agents_from_config(config)

Extract agent configurations from the loaded config.

# Arguments
- `config`: The loaded configuration dictionary

# Returns
- Array of Agent structs
"""
function get_agents_from_config(config)
    agent_configs = get(config, "agents", [])
    if isempty(agent_configs)
        error("No agent configurations found in config file")
    end
    
    # Import Agent type from Environment module
    using ..Environment: Agent
    
    agents = []
    for agent_config in agent_configs
        agent = Agent(
            radius = agent_config["radius"],
            initial_position = (agent_config["initial_position"][1], agent_config["initial_position"][2]),
            target_position = (agent_config["target_position"][1], agent_config["target_position"][2])
        )
        push!(agents, agent)
    end
    
    return agents
end

"""
    get_environment_from_config(config, env_name)

Create an environment from the configuration.

# Arguments
- `config`: The loaded configuration dictionary
- `env_name`: Name of the environment to load ("door", "wall", or "combined")

# Returns
- Environment struct with obstacles
"""
function get_environment_from_config(config, env_name)
    if !haskey(config, "environments") || !haskey(config["environments"], env_name)
        error("Environment '$env_name' not found in configuration")
    end
    
    env_config = config["environments"][env_name]
    obstacle_configs = get(env_config, "obstacles", [])
    
    if isempty(obstacle_configs)
        error("No obstacles found for environment '$env_name'")
    end
    
    # Import types from Environment module
    using ..Environment: Environment, Rectangle
    
    obstacles = []
    for obstacle_config in obstacle_configs
        center = (obstacle_config["center"][1], obstacle_config["center"][2])
        size = (obstacle_config["size"][1], obstacle_config["size"][2])
        obstacle = Rectangle(center = center, size = size)
        push!(obstacles, obstacle)
    end
    
    return Environment(obstacles = obstacles)
end

"""
    get_model_parameters(config)

Extract model parameters from the configuration.

# Arguments
- `config`: The loaded configuration dictionary

# Returns
- Named tuple containing model parameters
"""
function get_model_parameters(config)
    model_config = get(config, "model", Dict())
    
    # Get basic parameters
    dt = get(model_config, "dt", 1.0)
    gamma = get(model_config, "gamma", 1.0)
    nr_steps = get(model_config, "nr_steps", 40)
    nr_iterations = get(model_config, "nr_iterations", 350)
    softmin_temperature = get(model_config, "softmin_temperature", 10.0)
    save_intermediates = get(model_config, "save_intermediates", false)
    intermediate_steps = get(model_config, "intermediate_steps", 10)
    
    # Get matrices if defined, otherwise construct them
    matrices = get(model_config, "matrices", Dict())
    
    # Default matrices based on dt
    A_default = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
    B_default = [0 0; dt 0; 0 0; 0 dt]
    C_default = [1 0 0 0; 0 0 1 0]
    
    # Get matrices from config or use defaults
    A = haskey(matrices, "A") ? convert(Matrix{Float64}, reduce(hcat, matrices["A"])') : A_default
    B = haskey(matrices, "B") ? convert(Matrix{Float64}, reduce(hcat, matrices["B"])') : B_default
    C = haskey(matrices, "C") ? convert(Matrix{Float64}, reduce(hcat, matrices["C"])') : C_default
    
    # Get prior parameters
    priors_config = get(config, "priors", Dict())
    initial_state_variance = get(priors_config, "initial_state_variance", 100.0)
    control_variance = get(priors_config, "control_variance", 0.1)
    goal_constraint_variance = get(priors_config, "goal_constraint_variance", 1e-5)
    gamma_shape = get(priors_config, "gamma_shape", 1.5)
    gamma_scale_factor = get(priors_config, "gamma_scale_factor", 0.5)
    
    return (
        dt = dt,
        gamma = gamma,
        nr_steps = nr_steps,
        nr_iterations = nr_iterations,
        softmin_temperature = softmin_temperature,
        save_intermediates = save_intermediates,
        intermediate_steps = intermediate_steps,
        A = A,
        B = B,
        C = C,
        initial_state_variance = initial_state_variance,
        control_variance = control_variance,
        goal_constraint_variance = goal_constraint_variance,
        gamma_shape = gamma_shape,
        gamma_scale_factor = gamma_scale_factor
    )
end

"""
    get_visualization_parameters(config)

Extract visualization parameters from the configuration.

# Arguments
- `config`: The loaded configuration dictionary

# Returns
- Named tuple containing visualization parameters
"""
function get_visualization_parameters(config)
    vis_config = get(config, "visualization", Dict())
    
    # Get plot boundaries
    x_limits = get(vis_config, "x_limits", [-20, 20])
    y_limits = get(vis_config, "y_limits", [-20, 20])
    
    # Get other visualization parameters
    fps = get(vis_config, "fps", 15)
    heatmap_resolution = get(vis_config, "heatmap_resolution", 100)
    plot_width = get(vis_config, "plot_width", 800)
    plot_height = get(vis_config, "plot_height", 400)
    agent_alpha = get(vis_config, "agent_alpha", 1.0)
    target_alpha = get(vis_config, "target_alpha", 0.2)
    color_palette = get(vis_config, "color_palette", "tab10")
    
    return (
        x_limits = (x_limits[1], x_limits[2]),
        y_limits = (y_limits[1], y_limits[2]),
        fps = fps,
        heatmap_resolution = heatmap_resolution,
        plot_size = (plot_width, plot_height),
        agent_alpha = agent_alpha,
        target_alpha = target_alpha,
        color_palette = color_palette
    )
end

"""
    get_experiment_parameters(config)

Extract experiment parameters from the configuration.

# Arguments
- `config`: The loaded configuration dictionary

# Returns
- Named tuple containing experiment parameters
"""
function get_experiment_parameters(config)
    exp_config = get(config, "experiments", Dict())
    
    seeds = get(exp_config, "seeds", [42, 123])
    results_dir = get(exp_config, "results_dir", "results")
    animation_template = get(exp_config, "animation_template", "{environment}_{seed}.gif")
    control_vis_filename = get(exp_config, "control_vis_filename", "control_signals.gif")
    obstacle_distance_filename = get(exp_config, "obstacle_distance_filename", "obstacle_distance.png")
    path_uncertainty_filename = get(exp_config, "path_uncertainty_filename", "path_uncertainty.png")
    convergence_filename = get(exp_config, "convergence_filename", "convergence.png")
    
    return (
        seeds = seeds,
        results_dir = results_dir,
        animation_template = animation_template,
        control_vis_filename = control_vis_filename,
        obstacle_distance_filename = obstacle_distance_filename,
        path_uncertainty_filename = path_uncertainty_filename,
        convergence_filename = convergence_filename
    )
end

"""
    apply_config(config, softmin_fn, halfspace_node)

Apply the loaded configuration to the appropriate model components.

# Arguments
- `config`: The loaded configuration dictionary
- `softmin_fn`: Reference to the softmin function to modify
- `halfspace_node`: Reference to the Halfspace node to modify (if needed)

# Returns
- Named tuple containing all configuration components:
  - `:model_params`: Model parameters
  - `:agents`: Array of Agent structs
  - `:environments`: Dictionary of Environment structs
  - `:vis_params`: Visualization parameters
  - `:exp_params`: Experiment parameters
"""
function apply_config(config)
    # Extract components from config
    model_params = get_model_parameters(config)
    agents = get_agents_from_config(config)
    
    # Create environments
    environments = Dict(
        "door" => get_environment_from_config(config, "door"),
        "wall" => get_environment_from_config(config, "wall"),
        "combined" => get_environment_from_config(config, "combined")
    )
    
    # Get visualization and experiment parameters
    vis_params = get_visualization_parameters(config)
    exp_params = get_experiment_parameters(config)
    
    return (
        model_params = model_params,
        agents = agents,
        environments = environments,
        vis_params = vis_params,
        exp_params = exp_params
    )
end

end # module 