println("Ensuring all required packages are installed...")
using Pkg
const OUTPUT_DIR_ROOT = dirname(@__FILE__)
if isfile(joinpath(OUTPUT_DIR_ROOT, "Project.toml"))
    Pkg.activate(OUTPUT_DIR_ROOT)
    Pkg.instantiate()
    println("✓ Project dependencies installed")
else
    Pkg.activate(OUTPUT_DIR_ROOT)
end

ENV["GKSwstype"] = "100"
using RxInfer
using Distributions
using Plots
gr() # Use GR backend for better animation support
using Random
using ProgressMeter
using RxEnvironments
using Dates
using LinearAlgebra
using Statistics
using DelimitedFiles

include(joinpath(@__DIR__, "src", "POMDPControl.jl"))
using .POMDPControl

# Define output directories first
const OUTPUT_DIR_ROOT = dirname(@__FILE__)
const OUTPUT_DIR = joinpath(OUTPUT_DIR_ROOT, "outputs")
const ANALYTICS_DIR = joinpath(OUTPUT_DIR, "analytics")
const PLOTS_DIR = joinpath(OUTPUT_DIR, "plots")
const ANIMATION_DIR = joinpath(OUTPUT_DIR, "animations")
const MATRIX_DIR = joinpath(OUTPUT_DIR, "matrices")

println("$(now()) | Creating output directories...")
for dir in [OUTPUT_DIR, ANALYTICS_DIR, PLOTS_DIR, ANIMATION_DIR, MATRIX_DIR]
    if !isdir(dir)
        mkpath(dir)
        println("Created directory: $dir")
    end
end

env, agent = create_environment()
println("$(now()) | Environment initialized")

p_A, p_B, init, constraints = build_pomdp()
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
println("$(now()) | Saving initial environment state...")
initial_plot = plot_environment(env)
savefig(initial_plot, joinpath(OUTPUT_DIR, "initial_environment.png"))

# Modify the experiment loop to track more data
println("$(now()) | Running experiments...")
all_paths = Vector{Vector{Tuple{Int,Int}}}()
all_step_counts = Int[]
experiment_data = Dict{Int,Dict{Symbol,Any}}()

# Modify experiment loop to collect grid simulation data
grid_sim_data = []

# Analytics tracing containers
free_energy_series = Float64[]
entropy_series = Float64[]
policy_timeseries = Vector{Vector{Float64}}()
action_series = Int[]
efe_timeseries = Vector{Vector{Float64}}()

# VFE model (no planning horizon)
@model function vfe_model(p_A, p_B, p_previous_state)
    A ~ p_A
    B ~ p_B
    previous_state ~ p_previous_state
    current_state ~ DiscreteTransition(previous_state, B, previous_control)
    current_y ~ DiscreteTransition(current_state, A)
end

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
            model = POMDPControl.pomdp_model(
                p_A = p_A, p_B = p_B,
                T = max(T - t, 1),
                p_previous_state = p_s,
                p_goal = goal,
                p_control = vague(Categorical, 4),
                m_A = mean(p_A),
                m_B = mean(p_B)
            ),
            data = (
                previous_control = prev_u,
                current_y = last_observation,
                future_y = UnfactorizedData(fill(missing, max(T - t, 1)))
            ),
            constraints = constraints,
            initialization = init,
            iterations = 10,
            free_energy = false
        )
        
        p_s = last(inference_result.posteriors[:current_state])
        policy = last(inference_result.posteriors[:controls])
        
        global p_A = last(inference_result.posteriors[:A])
        global p_B = last(inference_result.posteriors[:B])

        # Track analytics for the first experiment
        if i == 1
            # VFE computation using simplified single-step model
            try
                vfe_result = infer(
                    model = vfe_model(
                        p_A = p_A,
                        p_B = p_B,
                        p_previous_state = p_s,
                        previous_control = prev_u,
                        current_y = last_observation
                    ),
                    iterations = 10,
                    free_energy = true
                )
                vfe_val = try
                    fe = vfe_result.free_energy
                    if isnothing(fe) || (isa(fe, AbstractVector) && isempty(fe))
                        NaN
                    else
                        isa(fe, AbstractVector) ? Float64(fe[end]) : Float64(fe)
                    end
                catch e
                    println("VFE computation failed: ", e)
                    NaN
                end
                push!(free_energy_series, vfe_val)
            catch e
                println("VFE model failed: ", e)
                push!(free_energy_series, NaN)
            end

            # Belief entropy of current state posterior
            probs = pdf.(p_s, 1:25)
            h = -sum(probs .* log.(probs .+ 1e-12))
            push!(entropy_series, h)

            # Policy probabilities and chosen action
            pol_probs = pdf.(first(policy), 1:4)
            push!(policy_timeseries, pol_probs)
            push!(action_series, current_action)

            # EFE computation for each action
            A_mean = mean(p_A)
            B_mean = mean(p_B)
            goal_probs = pdf(goal, 1:25)

            efe_actions = Float64[]
            for a in 1:4
                # Predict next state belief: q(s'|s,a) = B[:,:,a]^T * q(s)
                q_s_next = B_mean[:,:,a]' * pdf.(p_s, 1:25)

                # Predict observation belief: q(o'|s') = A^T * q(s')
                q_o_next = A_mean' * q_s_next

                # Compute EFE: E[KL(q(o'|s') || p(o'|goal))] - H[q(o'|a)]
                kl_term = sum(q_o_next .* (log.(q_o_next .+ 1e-12) .- log.(goal_probs .+ 1e-12)))
                entropy_term = -sum(q_o_next .* log.(q_o_next .+ 1e-12))
                efe = kl_term - entropy_term
                push!(efe_actions, efe)
            end
            push!(efe_timeseries, efe_actions)
        end
        
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
println("$(now()) | Experiment Results:")
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
println("$(now()) | Generating summary visualizations...")

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
println("$(now()) | Generating grid simulation frames...")
max_steps = maximum(length(data[:position_history]) for data in grid_sim_data)
for step in 0:max_steps
    frame = create_grid_simulation_frame(grid_sim_data, step)
    push!(grid_sim_frames, frame)
end

# After the experiment loop, add animation generation
println("$(now()) | Generating animations...")

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
println("$(now()) | Generating grid simulation animation...")
if !isempty(grid_sim_frames)
    anim = @animate for p in grid_sim_frames
        plot(p)
    end
    gif(anim, joinpath(ANIMATION_DIR, "grid_simulation.gif"), fps=2)
end

# Create combined matrix animation
println("$(now()) | Generating combined matrix animation...")
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

# Write an outputs index for quick review
open(joinpath(OUTPUT_DIR, "outputs_index.txt"), "w") do io
    println(io, "Run completed at: $(now())")
    println(io, "\nDirectories:")
    println(io, "- PLOTS_DIR: $(PLOTS_DIR)")
    println(io, "- ANIMATION_DIR: $(ANIMATION_DIR)")
    println(io, "- MATRIX_DIR: $(MATRIX_DIR)")
    println(io, "\nFile counts:")
    plots_count = isdir(PLOTS_DIR) ? length(readdir(PLOTS_DIR)) : 0
    anims_count = isdir(ANIMATION_DIR) ? length(readdir(ANIMATION_DIR)) : 0
    mats_count = isdir(MATRIX_DIR) ? length(readdir(MATRIX_DIR)) : 0
    println(io, "plots: $plots_count, animations: $anims_count, matrices: $mats_count")
end
println("$(now()) | Wrote outputs_index.txt")

# Save analytics CSVs and plots (first experiment tracing)
if !isempty(free_energy_series)
    steps = collect(1:length(free_energy_series))
    # VFE CSV
    open(joinpath(ANALYTICS_DIR, "vfe_timeseries.csv"), "w") do io
        write(io, "step,vfe_last_iter\n")
        for k in eachindex(steps)
            write(io, string(steps[k], ",", free_energy_series[k], "\n"))
        end
    end
    # VFE plot
    vfe_plot = plot(steps, free_energy_series, label="VFE", xlabel="Step", ylabel="Value", title="Variational Free Energy", legend=false)
    savefig(vfe_plot, joinpath(PLOTS_DIR, "vfe_timeseries.png"))

    # FE and entropy CSV
    open(joinpath(ANALYTICS_DIR, "free_energy_entropy.csv"), "w") do io
        write(io, "step,free_energy,entropy\n")
        for k in eachindex(steps)
            write(io, string(steps[k], ",", free_energy_series[k], ",", entropy_series[k], "\n"))
        end
    end

    # Enhanced entropy analysis plot
    entropy_plot = plot(steps, entropy_series,
                       label="Belief Entropy",
                       xlabel="Step", ylabel="Entropy (bits)",
                       title="Belief Entropy Evolution - Perception/Sensemaking Quality",
                       linewidth=2, color=:blue)
    # Add interpretation zones
    hline!([3.0], label="High Uncertainty", line=:dash, color=:red, alpha=0.7)
    hline!([1.0], label="Medium Uncertainty", line=:dash, color=:orange, alpha=0.7)
    hline!([0.1], label="Low Uncertainty", line=:dash, color=:green, alpha=0.7)
    savefig(entropy_plot, joinpath(PLOTS_DIR, "belief_entropy_perception.png"))

    # Combined FE and entropy analysis
    fe_plot = plot(layout=(2,1), size=(800,600))
    plot!(fe_plot[1], steps, free_energy_series, label="VFE (NaN - architectural limitation)",
          xlabel="", ylabel="VFE", title="Free Energy & Entropy Analysis", linewidth=2)
    plot!(fe_plot[2], steps, entropy_series, label="Belief Entropy",
          xlabel="Step", ylabel="Entropy", linewidth=2)
    savefig(fe_plot, joinpath(PLOTS_DIR, "free_energy_entropy_analysis.png"))
    savefig(fe_plot, joinpath(PLOTS_DIR, "free_energy_entropy.png"))  # Keep original for compatibility
end

if !isempty(policy_timeseries)
    steps = collect(1:length(policy_timeseries))
    # CSV
    open(joinpath(ANALYTICS_DIR, "policy_timeseries.csv"), "w") do io
        write(io, "step,up,right,down,left,action\n")
        for (k, probs) in enumerate(policy_timeseries)
            write(io, string(steps[k], ",", probs[1], ",", probs[2], ",", probs[3], ",", probs[4], ",", action_series[k], "\n"))
        end
    end
    # Plot
    pol_mat = reduce(vcat, (probs' for probs in policy_timeseries))
    pol_plot = plot(steps, pol_mat[:,1], label="Up", xlabel="Step", ylabel="Probability", ylim=(0,1), title="Policy Probabilities Over Time")
    plot!(pol_plot, steps, pol_mat[:,2], label="Right")
    plot!(pol_plot, steps, pol_mat[:,3], label="Down")
    plot!(pol_plot, steps, pol_mat[:,4], label="Left")
    savefig(pol_plot, joinpath(PLOTS_DIR, "policy_timeseries.png"))
end

if !isempty(efe_timeseries)
    steps = collect(1:length(efe_timeseries))

    # Enhanced EFE time series analysis
    efe_mat = reduce(vcat, (efe' for efe in efe_timeseries))

    # 1. EFE evolution plot
    efe_plot = plot(steps, efe_mat[:,1], label="G(up)", xlabel="Step", ylabel="Expected Free Energy",
                   title="Expected Free Energy per Action", linewidth=2)
    plot!(efe_plot, steps, efe_mat[:,2], label="G(right)", linewidth=2)
    plot!(efe_plot, steps, efe_mat[:,3], label="G(down)", linewidth=2)
    plot!(efe_plot, steps, efe_mat[:,4], label="G(left)", linewidth=2)
    # Highlight chosen actions
    chosen_efe = [efe_mat[k, action_series[k]] for k in 1:length(action_series)]
    scatter!(efe_plot, steps, chosen_efe, label="Chosen Action EFE", color=:red, markersize=4)
    savefig(efe_plot, joinpath(PLOTS_DIR, "efe_timeseries.png"))

    # 2. EFE vs Policy overlay (enhanced)
    if !isempty(policy_timeseries)
        pol_mat = reduce(vcat, (probs' for probs in policy_timeseries))
        combined_plot = plot(layout=(2,1), size=(1000,800))

        # Top subplot: EFE values
        plot!(combined_plot[1], steps, efe_mat[:,1], label="G(up)", xlabel="", ylabel="EFE",
              title="Expected Free Energy vs Policy Probabilities", line=:solid, linewidth=2)
        plot!(combined_plot[1], steps, efe_mat[:,2], label="G(right)", line=:solid, linewidth=2)
        plot!(combined_plot[1], steps, efe_mat[:,3], label="G(down)", line=:solid, linewidth=2)
        plot!(combined_plot[1], steps, efe_mat[:,4], label="G(left)", line=:solid, linewidth=2)

        # Bottom subplot: Policy probabilities
        plot!(combined_plot[2], steps, pol_mat[:,1], label="P(up)", xlabel="Step", ylabel="Policy Probability",
              line=:dash, linewidth=2)
        plot!(combined_plot[2], steps, pol_mat[:,2], label="P(right)", line=:dash, linewidth=2)
        plot!(combined_plot[2], steps, pol_mat[:,3], label="P(down)", line=:dash, linewidth=2)
        plot!(combined_plot[2], steps, pol_mat[:,4], label="P(left)", line=:dash, linewidth=2)

        savefig(combined_plot, joinpath(PLOTS_DIR, "efe_vs_policy_detailed.png"))
        savefig(combined_plot, joinpath(PLOTS_DIR, "efe_vs_policy.png"))  # Keep original for compatibility
    end

    # 3. Action selection efficiency plot
    if !isempty(action_series)
        action_efficiency = Float64[]
        for (k, chosen_action) in enumerate(action_series)
            other_efes = [efe_mat[k,j] for j in 1:4 if j != chosen_action]
            efficiency = chosen_efe[k] - maximum(other_efes)  # How much better than next best
            push!(action_efficiency, efficiency)
        end

        efficiency_plot = plot(steps, action_efficiency,
                              label="Action Selection Efficiency",
                              xlabel="Step", ylabel="EFE Advantage",
                              title="Action Selection Efficiency Over Time")
        hline!([0], label="No Advantage", line=:dash, color=:gray)
        savefig(efficiency_plot, joinpath(PLOTS_DIR, "action_selection_efficiency.png"))
    end
end

# Graphical abstract of POMDP matrices (means)
try
    A_mean = mean(p_A)
    B_mean = mean(p_B)

    # Enhanced graphical abstract with perception-action insights
    abs_plot = plot(layout=(3,2), size=(1200,1400))
    pA = heatmap(A_mean, title="Observation Model A\n(State → Observation Likelihood)", xlabel="State", ylabel="Observation")
    pB1 = heatmap(B_mean[:,:,1], title="Transition B - Up Action\n(State → Next State)")
    pB2 = heatmap(B_mean[:,:,2], title="Transition B - Right Action")
    pB3 = heatmap(B_mean[:,:,3], title="Transition B - Down Action")
    pB4 = heatmap(B_mean[:,:,4], title="Transition B - Left Action")
    plot!(abs_plot, pA, pB1, pB2, pB3, pB4)
    savefig(abs_plot, joinpath(PLOTS_DIR, "graphical_abstract_matrices.png"))

    # Perception summary statistics
    if !isempty(entropy_series)
        final_entropy = entropy_series[end]
        avg_entropy = mean(entropy_series)
        entropy_reduction = entropy_series[1] - entropy_series[end]

        # Create perception quality summary plot
        summary_stats = [final_entropy, avg_entropy, entropy_reduction, length(entropy_series)]
        summary_labels = ["Final Entropy", "Average Entropy", "Entropy Reduction", "Time Steps"]

        summary_plot = bar(summary_labels, summary_stats,
                          title="Perception/Sensemaking Summary",
                          ylabel="Value", legend=false,
                          color=[:blue, :green, :red, :purple])
        annotate!(1, summary_stats[1]/2, text("$(round(final_entropy, digits=3))", :white, :center, 10))
        annotate!(2, summary_stats[2]/2, text("$(round(avg_entropy, digits=3))", :white, :center, 10))
        annotate!(3, summary_stats[3]/2, text("$(round(entropy_reduction, digits=3))", :white, :center, 10))
        annotate!(4, summary_stats[4]/2, text("$(summary_stats[4])", :white, :center, 10))
        savefig(summary_plot, joinpath(PLOTS_DIR, "perception_summary.png"))
    end

    # Complete system overview
    if !isempty(efe_timeseries) && !isempty(policy_timeseries) && !isempty(entropy_series)
        local overview_plot = plot(layout=(3,1), size=(1000,1200))

        # Top: EFE landscape evolution
        local efe_mat = reduce(vcat, (efe' for efe in efe_timeseries))
        local steps = collect(1:length(efe_timeseries))
        plot!(overview_plot[1], steps, efe_mat[:,1], label="Up", title="Decision-Making Landscape Evolution",
              ylabel="Expected Free Energy", xlabel="", linewidth=2)
        plot!(overview_plot[1], steps, efe_mat[:,2], label="Right", linewidth=2)
        plot!(overview_plot[1], steps, efe_mat[:,3], label="Down", linewidth=2)
        plot!(overview_plot[1], steps, efe_mat[:,4], label="Left", linewidth=2)

        # Middle: Policy evolution
        local pol_mat = reduce(vcat, (probs' for probs in policy_timeseries))
        plot!(overview_plot[2], steps, pol_mat[:,1], label="P(Up)", ylabel="Policy Probability", xlabel="", linewidth=2)
        plot!(overview_plot[2], steps, pol_mat[:,2], label="P(Right)", linewidth=2)
        plot!(overview_plot[2], steps, pol_mat[:,3], label="P(Down)", linewidth=2)
        plot!(overview_plot[2], steps, pol_mat[:,4], label="P(Left)", linewidth=2)

        # Bottom: Perception quality
        plot!(overview_plot[3], steps, entropy_series, label="Belief Entropy",
              ylabel="Entropy (bits)", xlabel="Step", linewidth=2, color=:blue)
        hline!([2.0], label="Reference", line=:dash, color=:gray, alpha=0.5)

        savefig(overview_plot, joinpath(PLOTS_DIR, "system_overview_perception_action.png"))
    end

catch err
    @warn "Failed to generate enhanced visualizations" err
end