#!/usr/bin/env julia
# Simple 1D Navigation Example - Explicit agent-environment creation

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Dates
using Plots  # Required for visualizations

# Load framework in correct order (avoiding circular dependencies)
include("../src/types.jl")
include("../src/environments/abstract_environment.jl")
include("../src/environments/simple_nav_env.jl")
include("../src/agents/abstract_agent.jl")
include("../src/agents/simple_nav_agent.jl")
include("../src/constants.jl")
include("../src/diagnostics.jl")
include("../src/logging.jl")
include("../src/visualization.jl")
include("../src/simulation.jl")

using .Main: StateVector, ActionVector, ObservationVector

println("="^70)
println("SIMPLE 1D NAVIGATION - ACTIVE INFERENCE EXAMPLE")
println("="^70)
println()

# Create timestamped output directory
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
run_name = "simplenav_explicit_$(timestamp)"
run_dir = joinpath(@__DIR__, "..", "outputs", run_name)

# Create run-specific subdirectories
mkpath(joinpath(run_dir, "logs"))
mkpath(joinpath(run_dir, "data"))
mkpath(joinpath(run_dir, "plots"))
mkpath(joinpath(run_dir, "animations"))
mkpath(joinpath(run_dir, "diagnostics"))
mkpath(joinpath(run_dir, "results"))

println("Output directory: $run_dir")
println()

# Create environment
println("Creating Simple Navigation environment...")
env = SimpleNavEnv(
    initial_position = 0.0,
    goal_position = 1.0,
    dt = 0.1,
    velocity_limit = 0.5,
    observation_precision = 1e4,
    observation_noise_std = 0.01
)

# Get environment parameters for agent
env_params = get_observation_model_params(env)

# Create agent
println("Creating Simple Navigation Active Inference agent...")
goal_state = StateVector{1}([1.0])  # Goal: reach position 1.0
initial_state = StateVector{1}([0.0])

agent = SimpleNavAgent(
    10,  # horizon
    goal_state,
    initial_state,
    env_params,
    initial_state_precision = 1e6
)

# Create simulation configuration
println("Configuring simulation...")
config = SimulationConfig(
    max_steps = 30,
    enable_diagnostics = true,
    enable_logging = true,
    verbose = true,
    log_interval = 5
)

# Run simulation
println("\nRunning simulation...\n")
result = run_simulation(agent, env, config)

# Print results
println("\n" * "="^70)
println("SIMULATION RESULTS")
println("="^70)
println("Steps taken: $(result.steps_taken)")
println("Total time: $(round(result.total_time, digits=3))s")
println("Final position: $(result.states[end][1])")
println("Goal position: $(goal_state[1])")
println("Distance to goal: $(abs(result.states[end][1] - goal_state[1]))")
println("="^70)

# Summary
if abs(result.states[end][1] - goal_state[1]) < 0.1
    println("\n✅ SUCCESS: Agent reached the goal!")
else
    println("\n⚠️  Agent did not fully reach the goal.")
end

# Save all outputs with visualizations and animations
save_simulation_outputs(
    result,
    run_dir,
    goal_state,
    generate_visualizations=true,
    generate_animations=true
)

