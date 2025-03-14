using RxInfer
using Distributions
using Plots
gr() # Use GR backend for better animation support
using Random
using ProgressMeter
using RxEnvironments
using Dates
using FileIO
using LinearAlgebra

# Define output directories first
const OUTPUT_DIR = joinpath(dirname(@__FILE__), "outputs")
const ANALYTICS_DIR = joinpath(OUTPUT_DIR, "analytics")
const PLOTS_DIR = joinpath(OUTPUT_DIR, "plots")
const ANIMATION_DIR = joinpath(OUTPUT_DIR, "animations")
const MATRIX_DIR = joinpath(OUTPUT_DIR, "matrices")

# Create output directories
for dir in [OUTPUT_DIR, ANALYTICS_DIR, PLOTS_DIR, ANIMATION_DIR, MATRIX_DIR]
    if !isdir(dir)
        mkpath(dir)
        println("Created directory: $dir")
    end
end

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

env = RxEnvironment(WindyGridWorld((0, 1, 1, 1, 0), [], (4, 3)))
agent = add!(env, WindyGridWorldAgent((1, 1)))
plot_environment(env)

@model function pomdp_model(p_A, p_B, p_goal, p_control, previous_control, p_previous_state, current_y, future_y, T, m_A, m_B)
    # Instantiate all model parameters with priors
    A ~ p_A
    B ~ p_B
    previous_state ~ p_previous_state
    
    # Parameter inference
    current_state ~ DiscreteTransition(previous_state, B, previous_control)
    current_y ~ DiscreteTransition(current_state, A)

    prev_state = current_state
    # Inference-as-planning
    for t in 1:T
        controls[t] ~ p_control
        s[t] ~ DiscreteTransition(prev_state, m_B, controls[t])
        future_y[t] ~ DiscreteTransition(s[t], m_A)
        prev_state = s[t]
    end
    # Goal prior initialization
    s[end] ~ p_goal
end

init = @initialization begin
    q(A) = DirichletCollection(diageye(25) .+ 0.1)
    q(B) = DirichletCollection(ones(25, 25, 4))
end

constraints = @constraints begin
    q(previous_state, previous_control, current_state, B) = q(previous_state, previous_control, current_state)q(B)
    q(current_state, current_y, A) = q(current_state, current_y)q(A)
    q(current_state, s, controls, B) = q(current_state, s, controls), q(B)
    q(s, future_y, A) = q(s, future_y), q(A)
end

p_A = DirichletCollection(diageye(25) .+ 0.1)
p_B = DirichletCollection(ones(25, 25, 4))

function grid_location_to_index(pos::Tuple{Int, Int})
    return (pos[2] - 1) * 5 + pos[1]
end

function index_to_grid_location(index::Int)
    return (index % 5, index ÷ 5 + 1,)
end

function index_to_one_hot(index::Int)
    return [i == index ? 1.0 : 0.0 for i in 1:25]
end

goal = Categorical(index_to_one_hot(grid_location_to_index((4, 3))))

# Number of times to run the experiment
n_experiments = 100
# Number of steps in each experiment
T = 30  # Increased to 30 steps
# Number of grid simulations to show (5x5=25)
const GRID_SIMS = 25

# Setup observations
observations = keep(Any)
# Subscribe the agent to receive observations
RxEnvironments.subscribe_to_observations!(agent, observations)

# Animation frames storage
env_frames = []
belief_frames = []
policy_frames = []
transition_frames = []  # For B matrix evolution
observation_frames = []  # For A matrix evolution
grid_sim_frames = []    # For grid simulation

successes = []  # Track successful experiments

# Function to create a grid of simulations
function create_grid_simulation_frame(sims_data, step)
    plots = []
    for sim_idx in 1:min(GRID_SIMS, length(sims_data))
        data = sims_data[sim_idx]
        p = plot_environment(env)
        if step > 0 && step <= length(data[:position_history])
            path = data[:position_history][1:step]
            path_x = [pos[1] for pos in path]
            path_y = [pos[2] for pos in path]
            plot!(p, path_x, path_y, color=:red, label="", linewidth=2)
            # Mark current position
            scatter!([path_x[end]], [path_y[end]], color=:green, label="Current", markersize=6)
        end
        title!(p, "Sim $(sim_idx)")
        push!(plots, p)
    end
    
    # Arrange in 5x5 grid
    grid_plot = plot(plots..., layout=(5,5), size=(1500,1500), legend=false)
    title!(grid_plot, "Step $step")
    return grid_plot
end

# Function to visualize POMDP matrices
function visualize_pomdp_matrices(A_matrix, B_matrix, step)
    # Observation matrix (A) visualization
    A_plot = heatmap(A_matrix, 
                     title="Observation Matrix (A) - Step $(step)",
                     xlabel="Current State",
                     ylabel="Observation",
                     color=:viridis,
                     aspect_ratio=:equal)
    push!(observation_frames, A_plot)
    savefig(A_plot, joinpath(MATRIX_DIR, "A_matrix_step_$(step).png"))
    
    # Transition matrix (B) visualization - one for each action
    B_plots = []
    for action in 1:4
        B_slice = B_matrix[:, :, action]
        B_plot = heatmap(B_slice,
                        title="Transition Matrix (B) - Action $(action) - Step $(step)",
                        xlabel="Previous State",
                        ylabel="Current State",
                        color=:viridis,
                        aspect_ratio=:equal)
        push!(B_plots, B_plot)
    end
    
    # Combine B matrix plots
    B_combined = plot(B_plots..., layout=(2,2), size=(800,800))
    push!(transition_frames, B_combined)
    savefig(B_combined, joinpath(MATRIX_DIR, "B_matrices_step_$(step).png"))
end

function save_experiment_state(env, step, position_history, policy_probs, belief_state)
    # Environment state and path animation
    p = plot_environment(env)
    if !isempty(position_history)
        path_x = [pos[1] for pos in position_history]
        path_y = [pos[2] for pos in position_history]
        plot!(p, path_x, path_y, color=:red, label="Path", linewidth=2)
    end
    push!(env_frames, p)
    savefig(p, joinpath(PLOTS_DIR, "env_state_step_$(step).png"))
    
    # Policy probabilities animation
    if !isnothing(policy_probs)
        policy_plot = bar(["Up", "Right", "Down", "Left"], 
                         policy_probs, 
                         title="Action Probabilities - Step $(step)",
                         ylabel="Probability",
                         ylim=(0,1),
                         legend=false)
        push!(policy_frames, policy_plot)
        savefig(policy_plot, joinpath(PLOTS_DIR, "policy_step_$(step).png"))
    end
    
    # Belief state visualization animation
    if !isnothing(belief_state)
        belief_matrix = zeros(5, 5)
        belief_probs = pdf.(belief_state, 1:25)
        for i in 1:25
            x, y = index_to_grid_location(i)
            if 1 <= x <= 5 && 1 <= y <= 5
                belief_matrix[y, x] = belief_probs[i]
            end
        end
        belief_plot = heatmap(belief_matrix, 
                             title="Belief State - Step $(step)",
                             color=:viridis,
                             aspect_ratio=:equal,
                             clim=(0,1))
        push!(belief_frames, belief_plot)
        savefig(belief_plot, joinpath(PLOTS_DIR, "belief_state_step_$(step).png"))
    end
    
    # Add POMDP matrix visualization
    if !isnothing(p_A) && !isnothing(p_B)
        A_mean = mean(p_A)
        B_mean = mean(p_B)
        visualize_pomdp_matrices(A_mean, B_mean, step)
    end
end

# Add this after creating the environment
println("Saving initial environment state...")
initial_plot = plot_environment(env)
savefig(initial_plot, joinpath(OUTPUT_DIR, "initial_environment.png"))

# Modify the experiment loop to track more data
println("\nRunning experiments...")
all_paths = Vector{Vector{Tuple{Int,Int}}}()
all_step_counts = Int[]
experiment_data = Dict{Int,Dict{Symbol,Any}}()

# Modify experiment loop to collect grid simulation data
grid_sim_data = []

@showprogress for i in 1:n_experiments
    # Reset environment and initialize
    reset_env!(env)
    p_s = Categorical(index_to_one_hot(grid_location_to_index((1, 1))))
    policy = [Categorical([0.0, 0.0, 1.0, 0.0])]
    prev_u = [0.0, 0.0, 1.0, 0.0]
    
    # Track path and data for this experiment
    position_history = [(1, 1)]
    step_count = 0
    exp_data = Dict{Symbol,Any}()
    
    for t in 1:T
        step_count += 1
        current_action = mode(first(policy))
        
        # Record pre-action state
        if i == 1  # Only save visualizations for first experiment
            policy_probs = pdf.(first(policy), 1:4)
            save_experiment_state(env, t, position_history, policy_probs, p_s)
        end
        
        # Execute action
        if current_action == 1
            send!(env, agent, (0, 1))
            prev_u = [1.0, 0.0, 0.0, 0.0]
        elseif current_action == 2
            send!(env, agent, (1, 0))
            prev_u = [0.0, 1.0, 0.0, 0.0]
        elseif current_action == 3
            send!(env, agent, (0, -1))
            prev_u = [0.0, 0.0, 1.0, 0.0]
        elseif current_action == 4
            send!(env, agent, (-1, 0))
            prev_u = [0.0, 0.0, 0.0, 1.0]
        end
        
        # Record new position
        push!(position_history, RxEnvironments.data(last(observations)))
        
        # Rest of the inference code...
        last_observation = index_to_one_hot(grid_location_to_index(RxEnvironments.data(last(observations))))
        
        inference_result = infer(
            model = pomdp_model(
                p_A = p_A, p_B = p_B,
                T = max(T - t, 1),
                p_previous_state = p_s,
                p_goal = goal,
                p_control = vague(Categorical, 4),
                m_A = mean(p_A),
                m_B = mean(p_B)
            ),
            data = (
                previous_control = UnfactorizedData(prev_u),
                current_y = UnfactorizedData(last_observation),
                future_y = UnfactorizedData(fill(missing, max(T - t, 1)))
            ),
            constraints = constraints,
            initialization = init,
            iterations = 10
        )
        
        p_s = last(inference_result.posteriors[:current_state])
        policy = last(inference_result.posteriors[:controls])
        
        global p_A = last(inference_result.posteriors[:A])
        global p_B = last(inference_result.posteriors[:B])
        
        if RxEnvironments.data(last(observations)) == (4, 3)
            break
        end
    end
    
    # Store experiment results
    push!(all_paths, position_history)
    push!(all_step_counts, step_count)
    exp_data[:path] = position_history
    exp_data[:steps] = step_count
    exp_data[:success] = RxEnvironments.data(last(observations)) == (4, 3)
    exp_data[:position_history] = position_history
    experiment_data[i] = exp_data
    
    # Store data for grid simulation
    if i <= GRID_SIMS
        push!(grid_sim_data, exp_data)
    end
    
    if exp_data[:success]
        push!(successes, true)
    else
        push!(successes, false)
    end
end

# Save experiment results
success_rate = mean(successes)
println("\nExperiment Results:")
println("Success rate: $(round(success_rate * 100, digits=1))%")

# Convert successes to Vector{Bool} for proper indexing
successes_bool = Bool.(successes)
successful_steps = all_step_counts[successes_bool]

if !isempty(successful_steps)
    println("Average steps (successful): $(mean(successful_steps))")
    println("Min steps (successful): $(minimum(successful_steps))")
    println("Max steps (successful): $(maximum(successful_steps))")
else
    println("No successful experiments to analyze")
end

# Save detailed results to file
open(joinpath(OUTPUT_DIR, "experiment_results.txt"), "w") do io
    println(io, "POMDP Control Experiment Results")
    println(io, "============================")
    println(io, "Run at: $(now())")
    println(io, "\nParameters:")
    println(io, "- Number of experiments: $n_experiments")
    println(io, "- Maximum steps per experiment: $T")
    println(io, "- Planning horizon: $T")
    println(io, "\nResults:")
    println(io, "- Success rate: $(round(success_rate * 100, digits=1))%")
    if !isempty(successful_steps)
        println(io, "- Average steps (successful): $(mean(successful_steps))")
        println(io, "- Min steps (successful): $(minimum(successful_steps))")
        println(io, "- Max steps (successful): $(maximum(successful_steps))")
    else
        println(io, "No successful experiments to analyze")
    end
    
    println(io, "\nDetailed Results:")
    for (i, data) in experiment_data
        println(io, "\nExperiment $i:")
        println(io, "  Success: $(data[:success])")
        println(io, "  Steps: $(data[:steps])")
        println(io, "  Path: $(data[:path])")
    end
end

# Create and save summary visualizations
println("\nGenerating summary visualizations...")

# Success rate over time
if !isempty(successful_steps)
    success_by_step = [count(x -> x <= s, successful_steps) / count(successes_bool) for s in 1:maximum(all_step_counts)]
    success_rate_plot = plot(1:length(success_by_step), success_by_step,
                            title="Cumulative Success Rate by Step Count",
                            xlabel="Steps",
                            ylabel="Success Rate",
                            legend=false)
    savefig(success_rate_plot, joinpath(PLOTS_DIR, "success_rate_by_steps.png"))
end

# Step count distribution
step_dist_plot = histogram(all_step_counts,
                          title="Distribution of Steps to Goal",
                          xlabel="Steps",
                          ylabel="Count",
                          legend=false)
savefig(step_dist_plot, joinpath(PLOTS_DIR, "step_distribution.png"))

# Final environment state with example path
final_plot = plot_environment(env)
if !isempty(all_paths)
    # Plot the shortest successful path
    successful_paths = all_paths[successes_bool]
    if !isempty(successful_paths)
        shortest_path = successful_paths[argmin(length.(successful_paths))]
        path_x = [pos[1] for pos in shortest_path]
        path_y = [pos[2] for pos in shortest_path]
        plot!(final_plot, path_x, path_y, color=:red, label="Shortest Path", linewidth=2)
    end
end
savefig(final_plot, joinpath(OUTPUT_DIR, "final_environment.png"))

# Generate grid simulation frames
println("\nGenerating grid simulation frames...")
max_steps = maximum(length(data[:position_history]) for data in grid_sim_data)
for step in 0:max_steps
    frame = create_grid_simulation_frame(grid_sim_data, step)
    push!(grid_sim_frames, frame)
end

# After the experiment loop, add animation generation
println("\nGenerating animations...")

# Create environment animation with 2 fps
if !isempty(env_frames)
    anim = @animate for p in env_frames
        plot(p)
    end
    gif(anim, joinpath(ANIMATION_DIR, "environment_evolution.gif"), fps=2)
end

# Create belief state animation
if !isempty(belief_frames)
    anim = @animate for p in belief_frames
        plot(p)
    end
    gif(anim, joinpath(ANIMATION_DIR, "belief_state_evolution.gif"), fps=2)
end

# Create policy animation
if !isempty(policy_frames)
    anim = @animate for p in policy_frames
        plot(p)
    end
    gif(anim, joinpath(ANIMATION_DIR, "policy_evolution.gif"), fps=2)
end

# Create POMDP matrix animations
if !isempty(observation_frames)
    anim = @animate for p in observation_frames
        plot(p)
    end
    gif(anim, joinpath(ANIMATION_DIR, "observation_matrix_evolution.gif"), fps=2)
end

if !isempty(transition_frames)
    anim = @animate for p in transition_frames
        plot(p)
    end
    gif(anim, joinpath(ANIMATION_DIR, "transition_matrix_evolution.gif"), fps=2)
end

# After other animation generation code, add:
println("\nGenerating grid simulation animation...")
if !isempty(grid_sim_frames)
    anim = @animate for p in grid_sim_frames
        plot(p)
    end
    gif(anim, joinpath(ANIMATION_DIR, "grid_simulation.gif"), fps=2)
end

# Create combined matrix animation
println("\nGenerating combined matrix animation...")
if !isempty(observation_frames) && !isempty(transition_frames)
    combined_frames = []
    for (i, (obs_frame, trans_frame)) in enumerate(zip(observation_frames, transition_frames))
        combined = plot(
            obs_frame, trans_frame,
            layout=@layout([a{0.4h}; b]),
            size=(800, 1200),
            title=["Observation Matrix (A) - Step $i" "Transition Matrices (B) - Step $i"]
        )
        push!(combined_frames, combined)
    end
    
    anim = @animate for p in combined_frames
        plot(p)
    end
    gif(anim, joinpath(ANIMATION_DIR, "matrix_evolution_combined.gif"), fps=2)
end

println("\nAnimations saved to: $(ANIMATION_DIR)")
println("Matrix visualizations saved to: $(MATRIX_DIR)")
println("\nResults and visualizations saved to: $(OUTPUT_DIR)") 