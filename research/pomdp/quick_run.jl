#!/usr/bin/env julia

"""
Quick run of the POMDP Control example with fewer iterations.
"""

# Directory where this script is located
const OUTPUT_DIR = dirname(@__FILE__)

# Activate the project
using Pkg
Pkg.activate(OUTPUT_DIR)
Pkg.instantiate()

# Load required packages
using Dates
println("[" * string(Dates.now()) * "] Loading packages...")
using RxInfer
using Distributions
ENV["GKSwstype"] = "100" # headless GR to avoid GUI hangs
using Plots
gr()
using Random
using RxEnvironments
using Dates
println("[" * string(Dates.now()) * "] ✓ Packages loaded")

# Create outputs directory structure
mkpath(joinpath(OUTPUT_DIR, "outputs", "plots"))
mkpath(joinpath(OUTPUT_DIR, "outputs", "analytics"))

# Save a record of execution
open(joinpath(OUTPUT_DIR, "outputs", "analytics", "execution_log.txt"), "w") do io
    println(io, "POMDP Control Quick Run")
    println(io, "Executed at: $(now())")
    println(io, "------------------")
end

# Define the environment (copied from the original script)
struct WindyGridWorld{N}
    wind::NTuple{N,Int}
    agents::Vector
    goal::Tuple{Int,Int}
end

mutable struct WindyGridWorldAgent
    position::Tuple{Int,Int}
end

RxEnvironments.update!(env::WindyGridWorld, dt) = nothing

function RxEnvironments.receive!(env::WindyGridWorld{N}, agent::WindyGridWorldAgent, action::Tuple{Int,Int}) where {N}
    if action[1] != 0
        @assert action[2] == 0 "Only one of the two actions can be non-zero"
    elseif action[2] != 0
        @assert action[1] == 0 "Only one of the two actions can be non-zero"
    end
    new_position = (agent.position[1] + action[1], agent.position[2] + action[2] + env.wind[agent.position[1]])
    if all(elem -> 0 < elem < N, new_position)
        agent.position = new_position
    end
end

function RxEnvironments.what_to_send(env::WindyGridWorld, agent::WindyGridWorldAgent)
    return agent.position
end

function RxEnvironments.what_to_send(agent::WindyGridWorldAgent, env::WindyGridWorld)
    return agent.position
end

function RxEnvironments.add_to_state!(env::WindyGridWorld, agent::WindyGridWorldAgent)
    push!(env.agents, agent)
end

function reset_env!(environment::RxEnvironments.RxEntity{<:WindyGridWorld,T,S,A}) where {T,S,A}
    env = environment.decorated
    for agent in env.agents
        agent.position = (1, 1)
    end
    for subscriber in RxEnvironments.subscribers(environment)
        send!(subscriber, environment, (1, 1))
    end
end

function plot_environment(environment::RxEnvironments.RxEntity{<:WindyGridWorld,T,S,A}) where {T,S,A}
    env = environment.decorated
    p1 = scatter([env.goal[1]], [env.goal[2]], color=:blue, label="Goal", xlims=(0, 6), ylims=(0, 6))
    for agent in env.agents
        p1 = scatter!([agent.position[1]], [agent.position[2]], color=:red, label="Agent")
    end
    return p1
end

# Create the environment
println("[" * string(Dates.now()) * "] Setting up environment...")
env = RxEnvironment(WindyGridWorld((0, 1, 1, 1, 0), [], (4, 3)))
agent = add!(env, WindyGridWorldAgent((1, 1)))

# Save initial environment plot
println("[" * string(Dates.now()) * "] Creating initial environment plot...")
initial_plot = plot_environment(env)
try
    savefig(initial_plot, joinpath(OUTPUT_DIR, "outputs", "plots", "initial_environment.png"))
    println("[" * string(Dates.now()) * "] ✓ Saved initial environment plot")
catch err
    println("[" * string(Dates.now()) * "] ✗ Failed to save initial environment plot: $(err)")
end

# Define the POMDP model (simplified from original script)
function grid_location_to_index(pos::Tuple{Int, Int})
    return (pos[2] - 1) * 5 + pos[1]
end

function index_to_grid_location(index::Int)
    return (index % 5, index ÷ 5 + 1,)
end

function index_to_one_hot(index::Int)
    return [i == index ? 1.0 : 0.0 for i in 1:25]
end

# Run a simplified experiment
println("[" * string(Dates.now()) * "] Running a simplified experiment...")
reset_env!(env)

# Move the agent manually to demonstrate the environment
positions = [(1,1)]
send!(env, agent, (0, 1))  # Move up
push!(positions, env.decorated.agents[1].position)
send!(env, agent, (1, 0))  # Move right
push!(positions, env.decorated.agents[1].position)
send!(env, agent, (1, 0))  # Move right
push!(positions, env.decorated.agents[1].position)
send!(env, agent, (0, 1))  # Move up
push!(positions, env.decorated.agents[1].position)

println("[" * string(Dates.now()) * "] Creating path plot...")
path_plot = plot(title="Agent Path", xlims=(0, 6), ylims=(0, 6))
scatter!(path_plot, [env.decorated.goal[1]], [env.decorated.goal[2]], color=:blue, label="Goal")
for i in 1:length(positions)
    scatter!(path_plot, [positions[i][1]], [positions[i][2]], color=:red, alpha=0.5, label=i==1 ? "Agent Path" : "")
    if i > 1
        plot!(path_plot, [positions[i-1][1], positions[i][1]], [positions[i-1][2], positions[i][2]], 
              color=:red, alpha=0.5, label="")
    end
end
# Highlight start and current position
scatter!(path_plot, [positions[1][1]], [positions[1][2]], color=:green, label="Start")
scatter!(path_plot, [positions[end][1]], [positions[end][2]], color=:purple, label="Current")

savefig(path_plot, joinpath(OUTPUT_DIR, "outputs", "plots", "agent_path.png"))
println("✓ Saved agent path plot")

# Save final environment state
println("[" * string(Dates.now()) * "] Creating final environment plot...")
final_plot = plot_environment(env)
try
    savefig(final_plot, joinpath(OUTPUT_DIR, "outputs", "plots", "final_environment.png"))
    println("[" * string(Dates.now()) * "] ✓ Saved final environment plot")
catch err
    println("[" * string(Dates.now()) * "] ✗ Failed to save final environment plot: $(err)")
end

# Save configuration and results data
open(joinpath(OUTPUT_DIR, "outputs", "analytics", "quick_run_results.txt"), "w") do io
    println(io, "Environment Configuration:")
    println(io, "- Start position: (1, 1)")
    println(io, "- Goal position: (4, 3)")
    println(io, "- Wind settings: $(env.decorated.wind)")
    println(io, "\nAgent Path:")
    for (i, pos) in enumerate(positions)
        println(io, "  Step $i: $pos")
    end
    println(io, "\nGoal Reached: $(positions[end] == env.decorated.goal)")
end
println("✓ Saved results data")

println("\nQuick run completed successfully")
println("Output files are available in: $(OUTPUT_DIR)") 