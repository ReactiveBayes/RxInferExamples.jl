# Configuration Loading and Factory Functions
# Parses TOML config and creates agents/environments

using TOML

# Include only what we need for factories (no circular dependencies)
include("types.jl")
include("agents/abstract_agent.jl")
include("environments/abstract_environment.jl")
include("agents/mountain_car_agent.jl")
include("agents/simple_nav_agent.jl")
include("environments/mountain_car_env.jl")
include("environments/simple_nav_env.jl")
# NOTE: simulation.jl is NOT included here to avoid circular dependency
# simulation.jl includes config.jl, so config.jl must NOT include simulation.jl

using .Main: StateVector, ActionVector, ObservationVector

"""
load_config(path::String)

Load configuration from TOML file.

Args:
- path: Path to TOML config file

Returns:
- Dictionary with parsed configuration
"""
function load_config(path::String)
    if !isfile(path)
        error("Config file not found: $path")
    end
    
    config = TOML.parsefile(path)
    return config
end

"""
create_environment_from_config(config::Dict)

Factory function to create an environment from configuration.

Args:
- config: Environment configuration dictionary

Returns:
- AbstractEnvironment instance
"""
function create_environment_from_config(config::Dict)
    env_type = config["type"]
    
    if env_type == "MountainCarEnv"
        return MountainCarEnv(
            initial_position = get(config, "initial_position", -0.5),
            initial_velocity = get(config, "initial_velocity", 0.0),
            engine_force_limit = get(config, "engine_force_limit", 0.04),
            friction_coefficient = get(config, "friction_coefficient", 0.1),
            observation_precision = get(config, "observation_precision", 1e4),
            observation_noise_std = get(config, "observation_noise_std", 0.01)
        )
    elseif env_type == "SimpleNavEnv"
        return SimpleNavEnv(
            initial_position = get(config, "initial_position", 0.0),
            goal_position = get(config, "goal_position", 1.0),
            dt = get(config, "dt", 0.1),
            velocity_limit = get(config, "velocity_limit", 0.5),
            observation_precision = get(config, "observation_precision", 1e4),
            observation_noise_std = get(config, "observation_noise_std", 0.01)
        )
    else
        error("Unknown environment type: $env_type")
    end
end

"""
create_agent_from_config(agent_config::Dict, env_config::Dict, env)

Factory function to create an agent from configuration.

Args:
- agent_config: Agent configuration dictionary
- env_config: Environment configuration dictionary (for goal, initial state)
- env: Environment instance (for observation model params)

Returns:
- AbstractActiveInferenceAgent instance
"""
function create_agent_from_config(agent_config::Dict, env_config::Dict, env)
    agent_type = agent_config["type"]
    horizon = get(agent_config, "horizon", 20)
    initial_state_precision = get(agent_config, "initial_state_precision", 1e6)
    
    if agent_type == "MountainCarAgent"
        goal_pos = get(env_config, "goal_position", 0.5)
        goal_vel = get(env_config, "goal_velocity", 0.0)
        goal_state = StateVector{2}([goal_pos, goal_vel])
        
        init_pos = get(env_config, "initial_position", -0.5)
        init_vel = get(env_config, "initial_velocity", 0.0)
        initial_state = StateVector{2}([init_pos, init_vel])
        
        env_params = get_observation_model_params(env)
        
        return MountainCarAgent(
            horizon,
            goal_state,
            initial_state,
            env_params,
            initial_state_precision = initial_state_precision
        )
        
    elseif agent_type == "SimpleNavAgent"
        goal_pos = get(env_config, "goal_position", 1.0)
        goal_state = StateVector{1}([goal_pos])
        
        init_pos = get(env_config, "initial_position", 0.0)
        initial_state = StateVector{1}([init_pos])
        
        env_params = get_observation_model_params(env)
        
        return SimpleNavAgent(
            horizon,
            goal_state,
            initial_state,
            env_params,
            initial_state_precision = initial_state_precision
        )
    else
        error("Unknown agent type: $agent_type")
    end
end

"""
create_simulation_config_from_toml(config::Dict)

Create SimulationConfig from TOML configuration.

Args:
- config: Simulation configuration dictionary

Returns:
- SimulationConfig instance
"""
function create_simulation_config_from_toml(config::Dict)
    return SimulationConfig(
        max_steps = get(config, "max_steps", 100),
        enable_diagnostics = get(config, "enable_diagnostics", true),
        enable_logging = get(config, "enable_logging", true),
        verbose = get(config, "verbose", false),
        log_interval = get(config, "log_interval", 10)
    )
end

"""
ensure_output_directories(config::Dict)

Create output directories if they don't exist.

Args:
- config: Outputs configuration dictionary
"""
function ensure_output_directories(config::Dict)
    directories = [
        get(config, "base_dir", "outputs"),
        get(config, "logs_dir", "outputs/logs"),
        get(config, "data_dir", "outputs/data"),
        get(config, "plots_dir", "outputs/plots"),
        get(config, "animations_dir", "outputs/animations"),
        get(config, "diagnostics_dir", "outputs/diagnostics"),
        get(config, "results_dir", "outputs/results")
    ]
    
    for dir in directories
        if !isdir(dir)
            mkpath(dir)
        end
    end
    
    return directories
end

# Export
export load_config, create_environment_from_config, create_agent_from_config
export create_simulation_config_from_toml, ensure_output_directories

