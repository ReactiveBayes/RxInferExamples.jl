# Configuration Constants
# This file contains ONLY constants and has NO includes to avoid circular dependencies

module AgentConstants

# Diagnostics configuration
const DIAGNOSTICS = (
    track_beliefs = true,
    track_predictions = true,
    track_free_energy = true,
    track_inference_time = true,
)

# Logging configuration  
const LOGGING = (
    enable_logging = true,
    enable_memory_trace = true,
    enable_performance = true,
    enable_structured = false,
    log_to_console = true,
    log_to_file = true,
    log_file = "outputs/logs/agent.log",
    structured_file = "outputs/logs/structured.jsonl",
    performance_file = "outputs/logs/performance.csv",
    memory_trace_file = "outputs/logs/memory.csv",
    memory_trace_interval = 10,
)

# Outputs configuration
const OUTPUTS = (
    base_dir = "outputs",
    logs_dir = "outputs/logs",
    data_dir = "outputs/data",
    plots_dir = "outputs/plots",
    animations_dir = "outputs/animations",
    diagnostics_dir = "outputs/diagnostics",
    results_dir = "outputs/results",
)

export DIAGNOSTICS, LOGGING, OUTPUTS

end # module AgentConstants

