# Generalized Configuration System for Active Inference Car Examples
# Supports multiple car types, dynamics, and environments with enhanced modularity

@doc """
Generalized configuration system for active inference car examples.

This module provides a comprehensive, extensible configuration framework
that supports multiple car types, dynamics models, and environments.

## Supported Car Types
- :mountain_car - Classic mountain car with gravitational dynamics
- :race_car - High-speed racing car with aerodynamic forces
- :autonomous_car - Self-driving car with obstacle avoidance
- :custom - User-defined car type with custom physics

## Key Features
- **Modular Design**: Separate configuration sections for different components
- **Type Safety**: Named tuples with comprehensive validation
- **Extensibility**: Easy to add new car types and parameters
- **Validation**: Comprehensive parameter checking and error reporting
- **Documentation**: Extensive docstrings and examples
"""
module Config

using Logging
using Printf

# ==================== CORE CONFIGURATION SECTIONS ====================

@doc """
Physics configuration for different car dynamics models.

Each car type has specific physics parameters that define how the car
interacts with its environment.
"""
const PHYSICS_CONFIGS = Dict(
    :mountain_car => (
        # Basic physics parameters
        engine_force_limit = 0.04,
        friction_coefficient = 0.1,

        # Mountain-specific parameters
        gravity_factor = 0.0025,
        mass = 1.0,
        time_step = 0.1,

        # Numerical parameters
        integration_method = :euler,  # :euler, :rk4, :adaptive
        numerical_precision = 1e-6,
    ),

    :race_car => (
        # Aerodynamic parameters
        engine_force_limit = 0.15,
        friction_coefficient = 0.05,
        air_resistance = 0.01,
        downforce_coefficient = 0.02,

        # Racing-specific parameters
        mass = 1.0,
        tire_grip = 0.95,
        track_temperature = 25.0,

        # Numerical parameters
        integration_method = :rk4,
        numerical_precision = 1e-8,
    ),

    :autonomous_car => (
        # Sensor and control parameters
        engine_force_limit = 0.08,
        friction_coefficient = 0.08,
        sensor_range = 10.0,
        reaction_time = 0.1,

        # Autonomous-specific parameters
        safety_margin = 2.0,
        path_planning_horizon = 15,
        obstacle_detection_threshold = 5.0,

        # Numerical parameters
        integration_method = :adaptive,
        numerical_precision = 1e-7,
    )
)

@doc """
World/environment configuration for different scenarios.

Defines the environment properties, boundaries, and initial conditions
for each car type.
"""
const WORLD_CONFIGS = Dict(
    :mountain_car => (
        # Environment bounds
        x_min = -2.0,
        x_max = 2.0,
        v_min = -1.0,
        v_max = 1.0,

        # Initial conditions
        initial_position = -0.5,
        initial_velocity = 0.0,

        # Goal/target conditions
        target_position = 0.5,
        target_velocity = 0.0,
        goal_tolerance = 0.1,

        # Environment properties
        has_boundaries = true,
        wrap_around = false,
        stochastic = false,
    ),

    :race_car => (
        # Track bounds
        x_min = 0.0,
        x_max = 100.0,
        v_min = 0.0,
        v_max = 5.0,

        # Starting conditions
        initial_position = 0.0,
        initial_velocity = 1.0,

        # Race goals
        target_position = 100.0,
        target_velocity = 3.0,
        goal_tolerance = 1.0,

        # Track properties
        has_boundaries = true,
        wrap_around = true,
        stochastic = true,
    ),

    :autonomous_car => (
        # City environment
        x_min = -50.0,
        x_max = 50.0,
        v_min = -2.0,
        v_max = 3.0,

        # Urban driving conditions
        initial_position = 0.0,
        initial_velocity = 1.5,

        # Navigation goals
        target_position = 25.0,
        target_velocity = 1.8,
        goal_tolerance = 2.0,

        # Urban environment
        has_boundaries = true,
        wrap_around = false,
        stochastic = true,
    )
)

@doc """
Agent configuration for active inference parameters.

Defines the inference parameters, planning horizons, and precision
settings for the active inference agent.
"""
const AGENT_CONFIGS = Dict(
    :mountain_car => (
        # Planning parameters
        planning_horizon = 20,
        discount_factor = 0.95,

        # Inference precision parameters
        transition_precision = 1e4,
        observation_precision = 1e-4,
        control_prior_precision = 1e6,
        goal_prior_precision = 1e-4,
        initial_state_precision = 1e-6,

        # Learning parameters
        learning_rate = 0.01,
        adaptation_rate = 0.1,
    ),

    :race_car => (
        # Faster planning for racing
        planning_horizon = 30,
        discount_factor = 0.98,

        # High precision for racing
        transition_precision = 1e5,
        observation_precision = 1e-5,
        control_prior_precision = 1e7,
        goal_prior_precision = 1e-3,
        initial_state_precision = 1e-7,

        # Racing-specific parameters
        learning_rate = 0.05,
        adaptation_rate = 0.2,
    ),

    :autonomous_car => (
        # Longer horizon for navigation
        planning_horizon = 25,
        discount_factor = 0.96,

        # Balanced precision for urban driving
        transition_precision = 1e4,
        observation_precision = 1e-4,
        control_prior_precision = 1e6,
        goal_prior_precision = 1e-3,
        initial_state_precision = 1e-6,

        # Adaptive parameters
        learning_rate = 0.02,
        adaptation_rate = 0.15,
    )
)

@doc """
Simulation configuration for different experimental setups.

Defines simulation parameters, time steps, and experimental conditions.
"""
const SIMULATION_CONFIGS = Dict(
    :mountain_car => (
        # Time parameters
        time_steps = 100,
        max_episodes = 1,
        episode_length = 100,

        # Policy parameters
        naive_action = 100.0,  # Constant full power

        # Experiment settings
        enable_comparison = true,
        enable_visualization = true,
        enable_logging = true,
    ),

    :race_car => (
        # Racing simulation
        time_steps = 200,
        max_episodes = 3,
        episode_length = 200,

        # Racing policy
        naive_action = 150.0,  # Higher power for racing

        # Racing features
        enable_comparison = true,
        enable_visualization = true,
        enable_logging = true,
    ),

    :autonomous_car => (
        # Urban simulation
        time_steps = 150,
        max_episodes = 5,
        episode_length = 150,

        # Conservative policy
        naive_action = 80.0,  # Safer driving

        # Urban features
        enable_comparison = true,
        enable_visualization = true,
        enable_logging = true,
    )
)

@doc """
Visualization configuration for different display requirements.

Defines plotting parameters, animation settings, and visual themes
for each car type.
"""
const VISUALIZATION_CONFIGS = Dict(
    :mountain_car => (
        # Display parameters
        plot_size = (800, 400),
        animation_fps = 24,
        landscape_points = 400,
        landscape_range = (-2.0, 2.0),

        # Feature toggles
        show_predictions = true,
        show_uncertainty = true,
        show_energy = true,
        show_metrics = true,

        # Animation settings
        animation_themes = [:default, :dark, :colorblind_friendly],
    ),

    :race_car => (
        # Racing display
        plot_size = (1000, 500),
        animation_fps = 30,
        landscape_points = 600,
        landscape_range = (0.0, 100.0),

        # Racing features
        show_predictions = true,
        show_uncertainty = true,
        show_energy = true,
        show_metrics = true,

        # High-speed features
        show_speed_zones = true,
        show_lap_times = true,
    ),

    :autonomous_car => (
        # Urban display
        plot_size = (900, 450),
        animation_fps = 20,
        landscape_points = 500,
        landscape_range = (-50.0, 50.0),

        # Urban features
        show_predictions = true,
        show_uncertainty = true,
        show_energy = false,
        show_metrics = true,

        # Navigation features
        show_obstacles = true,
        show_path_planning = true,
        show_sensor_range = true,
    )
)

# ==================== OUTPUT CONFIGURATION ====================

@doc """
Output configuration for file paths and naming conventions.

Centralized output file management with consistent naming schemes.
"""
const OUTPUT_CONFIGS = Dict(
    :mountain_car => (
        output_dir = "outputs",
        log_filename = "active_inference_car.log",
        animation_prefix = "active-inference-car",
        results_prefix = "car_experiment",
    ),

    :race_car => (
        output_dir = "outputs",
        log_filename = "race_car.log",
        animation_prefix = "race-car",
        results_prefix = "racing_experiment",
    ),

    :autonomous_car => (
        output_dir = "outputs",
        log_filename = "autonomous_car.log",
        animation_prefix = "autonomous-car",
        results_prefix = "navigation_experiment",
    )
)

# ==================== DEFAULT CONFIGURATION SELECTION ====================

@doc """
Default car type selection.

Change this to switch between different car configurations.
"""
const DEFAULT_CAR_TYPE = :mountain_car

@doc """
Available car types dictionary.
"""
const CAR_TYPES = Dict(
    :mountain_car => (
        name = "Mountain Car",
        description = "Classic mountain car with gravitational dynamics",
        physics = "Gravitational forces with friction",
        challenges = ["Energy management", "Momentum building", "Hill climbing"]
    ),

    :race_car => (
        name = "Race Car",
        description = "High-performance racing car with aerodynamic forces",
        physics = "Aerodynamic drag, downforce, tire dynamics",
        challenges = ["High-speed stability", "Lap optimization", "Tire management"]
    ),

    :autonomous_car => (
        name = "Autonomous Car",
        description = "Urban autonomous vehicle with navigation and safety",
        physics = "Sensor-based dynamics, obstacle avoidance",
        challenges = ["Obstacle avoidance", "Path planning", "Safety constraints"]
    )
)

# ==================== CONFIGURATION ACCESS FUNCTIONS ====================

@doc """
Get complete configuration for a specific car type.

Args:
- car_type: Symbol specifying the car type (:mountain_car, :race_car, :autonomous_car)

Returns:
- Named tuple containing all configuration sections
"""
function get_car_config(car_type::Symbol = DEFAULT_CAR_TYPE)
    if !haskey(PHYSICS_CONFIGS, car_type)
        @warn "Unknown car type: $car_type. Using default: $DEFAULT_CAR_TYPE"
        car_type = DEFAULT_CAR_TYPE
    end

    return (
        car_type = car_type,
        physics = PHYSICS_CONFIGS[car_type],
        world = WORLD_CONFIGS[car_type],
        agent = AGENT_CONFIGS[car_type],
        simulation = SIMULATION_CONFIGS[car_type],
        visualization = VISUALIZATION_CONFIGS[car_type],
        outputs = OUTPUT_CONFIGS[car_type],
    )
end

@doc """
Get specific configuration section for a car type.

Args:
- section: Symbol specifying configuration section (:physics, :world, :agent, etc.)
- car_type: Symbol specifying the car type

Returns:
- Named tuple for the requested section
"""
function get_config_section(section::Symbol, car_type::Symbol = DEFAULT_CAR_TYPE)
    config = get_car_config(car_type)
    return getfield(config, section)
end

@doc """
Validate configuration consistency and report issues.

Returns:
- Vector of validation error messages
"""
function validate_configuration(car_type::Symbol = DEFAULT_CAR_TYPE)
    errors = String[]

    try
        config = get_car_config(car_type)

        # Validate physics parameters
        physics = config.physics
        if physics.engine_force_limit <= 0
            push!(errors, "Engine force limit must be positive")
        end
        if physics.friction_coefficient < 0
            push!(errors, "Friction coefficient cannot be negative")
        end

        # Validate world parameters
        world = config.world
        if world.x_max <= world.x_min
            push!(errors, "World x_max must be greater than x_min")
        end
        if world.goal_tolerance <= 0
            push!(errors, "Goal tolerance must be positive")
        end

        # Validate agent parameters
        agent = config.agent
        if agent.planning_horizon <= 0
            push!(errors, "Planning horizon must be positive")
        end
        if agent.transition_precision <= 0
            push!(errors, "Transition precision must be positive")
        end

        # Validate simulation parameters
        sim = config.simulation
        if sim.time_steps <= 0
            push!(errors, "Time steps must be positive")
        end

    catch e
        push!(errors, "Configuration validation failed: $e")
    end

    return errors
end

@doc """
Print detailed configuration information.

Args:
- car_type: Symbol specifying the car type to display
"""
function print_configuration(car_type::Symbol = DEFAULT_CAR_TYPE)
    config = get_car_config(car_type)

    println("=== Active Inference Car Configuration ===")
    println("Car Type: $car_type")
    println()

    println("Physics Parameters:")
    for (key, value) in pairs(config.physics)
        println("  $key = $value")
    end
    println()

    println("World Parameters:")
    for (key, value) in pairs(config.world)
        println("  $key = $value")
    end
    println()

    println("Agent Parameters:")
    for (key, value) in pairs(config.agent)
        println("  $key = $value")
    end
    println()

    println("Simulation Parameters:")
    for (key, value) in pairs(config.simulation)
        println("  $key = $value")
    end
    println()

    # Check for validation errors
    errors = validate_configuration(car_type)
    if isempty(errors)
        println("✓ Configuration validation: PASSED")
    else
        println("⚠ Configuration validation issues:")
        for error in errors
            println("  - $error")
        end
    end
end

@doc """
Get configuration value with fallback.

Args:
- section: Configuration section symbol
- key: Parameter key symbol
- default: Default value if not found
- car_type: Car type symbol

Returns:
- Configuration value or default
"""
function get_config_value(section::Symbol, key::Symbol, default = nothing, car_type::Symbol = DEFAULT_CAR_TYPE)
    try
        config_section = get_config_section(section, car_type)
        # Use get() to safely access named tuple fields
        return get(config_section, key, default)
    catch
        return default
    end
end

# ==================== CONFIGURATION EXPORT ====================

@doc """
Export current configuration to JSON format.

Args:
- filename: Output filename
- car_type: Car type to export

Returns:
- Success status
"""
function export_configuration(filename::String, car_type::Symbol = DEFAULT_CAR_TYPE)
    config = get_car_config(car_type)

    try
        open(filename, "w") do f
            JSON.print(f, Dict(
                "car_type" => string(car_type),
                "physics" => Dict(pairs(config.physics)),
                "world" => Dict(pairs(config.world)),
                "agent" => Dict(pairs(config.agent)),
                "simulation" => Dict(pairs(config.simulation)),
                "visualization" => Dict(pairs(config.visualization)),
                "outputs" => Dict(pairs(config.outputs)),
                "validation_errors" => validate_configuration(car_type)
            ), 2)
        end
        return true
    catch e
        @error "Failed to export configuration: $e"
        return false
    end
end

# ==================== DYNAMIC CONFIGURATION ====================

@doc """
Create a custom car configuration by merging with base configuration.

Args:
- base_car_type: Base car type to start from
- overrides: Dictionary of parameter overrides

Returns:
- Custom configuration named tuple
"""
function create_custom_config(base_car_type::Symbol, overrides::Dict{Symbol, Any})
    base_config = get_car_config(base_car_type)

    # Deep merge overrides into base configuration
    function merge_configs(base, overrides)
        merged = Dict(pairs(base))
        for (key, value) in overrides
            if haskey(merged, key) && typeof(merged[key]) <: NamedTuple && typeof(value) <: Dict
                # Recursively merge nested configurations
                merged[key] = merge_configs(merged[key], value)
            else
                merged[key] = value
            end
        end
        return NamedTuple{Tuple(keys(merged))}(values(merged))
    end

    custom_config = merge_configs(base_config, overrides)
    return custom_config
end

# ==================== CONFIGURATION CONSTANTS ====================

@doc """
Global configuration constants derived from the selected car type.
These are computed once at module load time.
"""
const ACTIVE_CONFIG = get_car_config(DEFAULT_CAR_TYPE)
const PHYSICS = ACTIVE_CONFIG.physics
const WORLD = ACTIVE_CONFIG.world
const AGENT = ACTIVE_CONFIG.agent
const SIMULATION = ACTIVE_CONFIG.simulation
const VISUALIZATION = ACTIVE_CONFIG.visualization
const OUTPUTS = ACTIVE_CONFIG.outputs

# ==================== MODULE EXPORTS ====================

export
    # Configuration access
    get_car_config,
    get_config_section,
    get_config_value,
    print_configuration,

    # Validation and export
    validate_configuration,
    export_configuration,

    # Custom configuration
    create_custom_config,

    # Configuration constants
    PHYSICS_CONFIGS,
    WORLD_CONFIGS,
    AGENT_CONFIGS,
    SIMULATION_CONFIGS,
    VISUALIZATION_CONFIGS,
    OUTPUT_CONFIGS,

    # Active configuration
    ACTIVE_CONFIG,
    PHYSICS,
    WORLD,
    AGENT,
    SIMULATION,
    VISUALIZATION,
    OUTPUTS,

    # Car types
    DEFAULT_CAR_TYPE

end # module Config
