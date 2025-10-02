#!/usr/bin/env julia
# Quick test to verify visualizations work

using Pkg
Pkg.activate(@__DIR__)

println("="^70)
println("QUICK VISUALIZATION TEST")
println("="^70)
println()

# Load Plots first to ensure it's available
println("[1/3] Loading Plots.jl...")
using Plots
using Dates
println("  ✓ Plots.jl loaded")

# Load framework
println("\n[2/3] Loading framework...")
include("src/types.jl")
include("src/environments/abstract_environment.jl")
include("src/environments/simple_nav_env.jl")
include("src/agents/abstract_agent.jl")
include("src/agents/simple_nav_agent.jl")
include("src/constants.jl")
include("src/diagnostics.jl")
include("src/logging.jl")
include("src/visualization.jl")
include("src/simulation.jl")

using .Main: StateVector, ActionVector, ObservationVector
using .Visualization
println("  ✓ Framework loaded")

# Run quick test
println("\n[3/3] Running quick 5-step simulation with visualization...")

# Create test directory
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
test_dir = joinpath(@__DIR__, "outputs", "quick_test_$(timestamp)")
mkpath(test_dir)

# Setup
env = SimpleNavEnv(initial_position = 0.0, goal_position = 1.0)
env_params = get_observation_model_params(env)
goal_state = StateVector{1}([1.0])
initial_state = StateVector{1}([0.0])
agent = SimpleNavAgent(5, goal_state, initial_state, env_params)

config = SimulationConfig(
    max_steps = 5,
    enable_diagnostics = true,
    enable_logging = false,
    verbose = false
)

# Run
result = run_simulation(agent, env, config)
println("  ✓ Simulation completed ($(result.steps_taken) steps)")

# Save with visualizations
println("\n  Saving outputs with visualizations...")
save_simulation_outputs(
    result,
    test_dir,
    goal_state,
    generate_visualizations=true,
    generate_animations=true
)

# Verify outputs
println("\n" * "="^70)
println("VERIFICATION")
println("="^70)

required_files = [
    "REPORT.md",
    "metadata.json",
    "data/trajectory.csv",
    "data/observations.csv",
    "results/summary.csv",
    "diagnostics/diagnostics.json",
    "plots/trajectory_1d.png",
    "animations/trajectory_1d.gif"
]

all_present = true
for file in required_files
    full_path = joinpath(test_dir, file)
    if isfile(full_path)
        size_kb = round(filesize(full_path) / 1024, digits=2)
        println("  ✓ $file ($(size_kb) KB)")
    else
        println("  ✗ MISSING: $file")
        all_present = false
    end
end

if all_present
    println("\n" * "="^70)
    println("✅ SUCCESS: All visualizations generated!")
    println("="^70)
    println("\nOutputs saved to: $test_dir")
    println()
    println("Check:")
    println("  • plots/trajectory_1d.png - Static plot")
    println("  • animations/trajectory_1d.gif - Animated GIF")
    println("  • REPORT.md - Comprehensive report")
    exit(0)
else
    println("\n" * "="^70)
    println("❌ FAILURE: Some outputs missing")
    println("="^70)
    exit(1)
end

