# Configuration module for Generic Active Inference Agent Framework
# Provides flexible, extensible configuration system

@doc """
Generic configuration system for Active Inference agents.

This module provides a hierarchical configuration system that can be easily
customized for different problem domains while maintaining consistency.
"""
module Config

using Logging

# ==================== CORE AGENT PARAMETERS ====================

const AGENT = (
    # Planning parameters
    planning_horizon = 20,              # Number of steps to plan ahead
    
    # Precision parameters (inverse variances)
    transition_precision = 1e4,         # Precision of state transitions (Gamma)
    observation_precision = 1e4,        # Precision of observations
    control_prior_precision = 1e-6,     # Initial control prior precision (huge variance)
    goal_prior_precision = 1e4,         # Goal prior precision (Sigma)
    initial_state_precision = 1e6,      # Initial state belief precision (tiny variance)
    
    # Inference parameters
    inference_iterations = 10,          # Number of inference iterations
    free_energy_tracking = true,        # Track free energy during inference
    convergence_tolerance = 1e-6,       # Convergence tolerance for inference
    
    # Memory and caching
    cache_results = true,               # Cache inference results
    max_memory_mb = 1000,               # Maximum memory usage in MB
    
    # Advanced features
    enable_diagnostics = true,          # Enable diagnostic tracking
    enable_memory_trace = true,         # Enable memory tracing
    enable_performance_trace = true,    # Enable performance profiling
)

# ==================== SIMULATION PARAMETERS ====================

const SIMULATION = (
    max_timesteps = 100,                # Maximum simulation timesteps
    verbose = false,                    # Verbose output during simulation
    seed = 42,                          # Random seed for reproducibility
    
    # Early stopping
    enable_early_stopping = true,       # Stop when goal is reached
    goal_tolerance = 0.01,              # Tolerance for goal achievement
)

# ==================== OUTPUT PARAMETERS ====================
# All outputs go into organized subdirectories under outputs/

const OUTPUTS = (
    # Main output directory
    base_dir = "outputs",                           # Root directory for all outputs
    
    # Subdirectories for organized output
    logs_dir = "outputs/logs",                      # All log files
    data_dir = "outputs/data",                      # Data exports (CSV, JSON)
    plots_dir = "outputs/plots",                    # Static plots and figures
    animations_dir = "outputs/animations",          # Animated visualizations
    diagnostics_dir = "outputs/diagnostics",        # Diagnostic reports and traces
    results_dir = "outputs/results",                # Simulation results
    
    # Export format flags
    export_json = true,                             # Export to JSON format
    export_csv = true,                              # Export to CSV format
    export_plots = true,                            # Export plots and figures
    
    # Output options
    compress_outputs = false,                       # Compress output files
    timestamp_outputs = true,                       # Add timestamps to output filenames
)

# ==================== LOGGING PARAMETERS ====================

const LOGGING = (
    enable_logging = true,                          # Enable logging
    log_level = Logging.Info,                       # Default log level
    log_to_console = true,                          # Log to console
    log_to_file = true,                             # Log to file
    
    # Log files (all in outputs/logs/)
    log_file = "outputs/logs/agent.log",            # Main log file
    structured_file = "outputs/logs/structured.jsonl",  # Structured JSON logs
    performance_file = "outputs/logs/performance.csv",   # Performance metrics
    memory_trace_file = "outputs/logs/memory.csv",       # Memory traces
    
    # Structured logging
    enable_structured = true,                       # Enable structured JSON logging
    enable_performance = true,                      # Enable performance logging
    enable_memory_trace = true,                     # Enable memory tracing
    memory_trace_interval = 10,                     # Trace memory every N steps
)

# ==================== DIAGNOSTICS PARAMETERS ====================

const DIAGNOSTICS = (
    # Tracking flags
    track_beliefs = true,                           # Track belief evolution
    track_actions = true,                           # Track action history
    track_predictions = true,                       # Track prediction accuracy
    track_free_energy = true,                       # Track free energy
    track_inference_time = true,                    # Track inference timing
    track_memory_usage = true,                      # Track memory usage
    
    # Storage (all in outputs/diagnostics/)
    save_diagnostics = true,                        # Save diagnostics to disk
    diagnostics_dir = "outputs/diagnostics",        # Diagnostics output directory
    beliefs_file = "outputs/diagnostics/beliefs.csv",      # Belief evolution
    actions_file = "outputs/diagnostics/actions.csv",      # Action history
    predictions_file = "outputs/diagnostics/predictions.csv",  # Predictions
    free_energy_file = "outputs/diagnostics/free_energy.csv",  # Free energy
)

# ==================== VISUALIZATION PARAMETERS ====================

const VISUALIZATION = (
    enable_plots = true,                            # Enable visualization
    plot_realtime = false,                          # Real-time plotting during simulation
    plot_final = true,                              # Plot final results
    
    # Output directories (all in outputs/)
    plots_dir = "outputs/plots",                    # Static plots directory
    animations_dir = "outputs/animations",          # Animations directory
    
    # Animation settings
    create_animations = true,                       # Create GIF animations
    animation_fps = 24,                             # Animation frames per second
    
    # Plot settings
    plot_size = (800, 600),                         # Default plot size
    plot_theme = :default,                          # Plot theme
    save_format = :png,                             # Save format for plots
)

# ==================== NUMERICAL PARAMETERS ====================

const NUMERICAL = (
    epsilon = 1e-10,                    # Small value for numerical stability
    tolerance = 1e-6,                   # General numerical tolerance
    max_iterations = 1000,              # Maximum iterations for numerical methods
    clip_values = true,                 # Clip extreme values
    clip_range = (-1e6, 1e6),          # Clipping range
)

@doc """
Ensure all output directories exist.

Creates the full directory structure for outputs if it doesn't already exist.
"""
function ensure_output_directories()
    directories = [
        OUTPUTS.base_dir,
        OUTPUTS.logs_dir,
        OUTPUTS.data_dir,
        OUTPUTS.plots_dir,
        OUTPUTS.animations_dir,
        OUTPUTS.diagnostics_dir,
        OUTPUTS.results_dir
    ]
    
    for dir in directories
        if !isdir(dir)
            mkpath(dir)
        end
    end
    
    return directories
end

@doc """
Validate configuration parameters.

Returns:
- Vector of validation issues (empty if valid)
"""
function validate_config()
    issues = String[]
    
    # Validate agent parameters
    if AGENT.planning_horizon <= 0
        push!(issues, "Planning horizon must be positive")
    end
    
    if AGENT.transition_precision <= 0
        push!(issues, "Transition precision must be positive")
    end
    
    if AGENT.observation_precision <= 0
        push!(issues, "Observation precision must be positive")
    end
    
    # Validate simulation parameters
    if SIMULATION.max_timesteps <= 0
        push!(issues, "Max timesteps must be positive")
    end
    
    # Validate numerical parameters
    if NUMERICAL.epsilon <= 0
        push!(issues, "Epsilon must be positive")
    end
    
    if NUMERICAL.tolerance <= 0
        push!(issues, "Tolerance must be positive")
    end
    
    return issues
end

@doc """
Create custom configuration by merging with defaults.

Args:
- custom_params: Dictionary of custom parameter overrides

Returns:
- Merged configuration
"""
function create_custom_config(custom_params::Dict{Symbol, Any})
    # This would create a custom configuration
    # For now, just validate and return defaults
    issues = validate_config()
    if !isempty(issues)
        @warn "Configuration validation issues" issues
    end
    
    return (
        AGENT = AGENT,
        SIMULATION = SIMULATION,
        LOGGING = LOGGING,
        DIAGNOSTICS = DIAGNOSTICS,
        VISUALIZATION = VISUALIZATION,
        OUTPUTS = OUTPUTS,
        NUMERICAL = NUMERICAL
    )
end

@doc """
Print current configuration.
"""
function print_configuration()
    println("=== Active Inference Agent Configuration ===")
    println("\n[Agent Parameters]")
    for (k, v) in pairs(AGENT)
        println("  $k = $v")
    end
    
    println("\n[Simulation Parameters]")
    for (k, v) in pairs(SIMULATION)
        println("  $k = $v")
    end
    
    println("\n[Logging Parameters]")
    for (k, v) in pairs(LOGGING)
        println("  $k = $v")
    end
    
    println("\n[Diagnostics Parameters]")
    for (k, v) in pairs(DIAGNOSTICS)
        println("  $k = $v")
    end
    
    println("\n[Visualization Parameters]")
    for (k, v) in pairs(VISUALIZATION)
        println("  $k = $v")
    end
    
    println("\n[Output Parameters]")
    for (k, v) in pairs(OUTPUTS)
        println("  $k = $v")
    end
    
    println("\n[Numerical Parameters]")
    for (k, v) in pairs(NUMERICAL)
        println("  $k = $v")
    end
    
    println("\n============================================")
end

# Export configuration sections
export AGENT, SIMULATION, LOGGING, DIAGNOSTICS, VISUALIZATION, OUTPUTS, NUMERICAL
export validate_config, create_custom_config, print_configuration, ensure_output_directories

end # module Config

