# Configuration file for Active Inference Mountain Car example
# All variables are centralized here for easy modification and experimentation

@doc """
Configuration parameters for the Active Inference Mountain Car example.

This module contains all configurable parameters used throughout the example,
including physics parameters, simulation settings, agent parameters, and visualization settings.
"""
module Config

# Physics parameters
const PHYSICS = (
    engine_force_limit = 0.04,      # Maximum engine force limit
    friction_coefficient = 0.1,     # Friction coefficient
)

# World/environment parameters
const WORLD = (
    initial_position = -0.5,        # Initial car position
    initial_velocity = 0.0,         # Initial car velocity
    target_position = 0.5,          # Target position (camping site)
    target_velocity = 0.0,          # Target velocity
)

# Target configuration
const TARGET = (
    position = WORLD.target_position,
    velocity = WORLD.target_velocity,
)

# Simulation parameters
const SIMULATION = (
    time_steps_naive = 50,         # Number of time steps for naive policy
    time_steps_ai = 50,            # Number of time steps for AI policy
    planning_horizon = 20,          # Agent's planning horizon (T)
    naive_action = 100.0,           # Fixed action for naive policy (full power)
)

# Agent parameters
const AGENT = (
    transition_precision = 1e4,     # Precision for state transitions (Gamma)
    observation_variance = 1e-4,    # Observation noise variance (Theta)
    control_prior_variance = 1e6,   # Control prior variance (huge)
    goal_prior_variance = 1e-4,     # Goal prior variance (Sigma)
    initial_state_variance = 1e-6,  # Initial state variance (tiny)
)

# Visualization parameters
const VISUALIZATION = (
    landscape_points = 400,         # Number of points for landscape plot
    landscape_range = (-2.0, 2.0),  # Range for landscape x-axis
    animation_fps = 24,             # Frames per second for animations
    plot_size = (800, 400),         # Size of plots
    engine_force_limits = (-0.05, 0.05), # Y-axis limits for engine force plot
)

# File paths for outputs
const OUTPUTS = (
    output_dir = "outputs",
    naive_animation = "outputs/ai-mountain-car-naive.gif",
    ai_animation = "outputs/ai-mountain-car-ai.gif",
    log_file = "outputs/mountain_car_log.txt",
    structured_log_file = "outputs/mountain_car_log_structured.jsonl",
    performance_log_file = "outputs/mountain_car_log_performance.csv",
    results_dir = "outputs/results",
)

# Numerical constants
const NUMERICAL = (
    epsilon = 1e-3,                 # Small value for numerical stability
    tolerance = 1e-6,               # Tolerance for convergence checks
)

# Export all configuration sections
export PHYSICS, WORLD, TARGET, SIMULATION, AGENT, VISUALIZATION, OUTPUTS, NUMERICAL

end # module Config
