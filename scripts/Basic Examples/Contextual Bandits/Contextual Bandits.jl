# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Basic Examples/Contextual Bandits/Contextual Bandits.ipynb
# by notebooks_to_scripts.jl at 2025-06-03T10:14:28.896
#
# Source notebook: Contextual Bandits.ipynb

using RxInfer, Distributions, LinearAlgebra, Plots, StatsPlots, ProgressMeter, StableRNGs

# Random number generator 
rng = StableRNG(42)
# Number of data points to generate
n_samples = 200
# Number of arms (actions) in the bandit problem
n_arms    = 3
# Number of different contexts available
n_contexts  = 10
# Dimensionality of context feature vectors
context_dim = 6

# Generate true arm parameters (θ_k in the model description)
arms     = [ randn(rng, context_dim) for _ in 1:n_arms] # True arm parameters
# Generate (normalized) context feature vectors (x_t in the model description)
contexts = [ randn(rng, context_dim) for _ in 1:n_contexts] # Context feature vectors

# Standard deviation of the observation noise (1/sqrt(τ) in the model)
noise_sd = 0.1

# Arrays to store the generated data
arm_choices     = [] # Stores a_t - which arm was chosen
context_choices = [] # Stores indices of contexts used
rewards         = [] # Stores r_t - the observed rewards

# Generate synthetic data according to the model
for i in 1:n_samples
    # Randomly select a context and an arm
    push!(context_choices, rand(rng, 1:n_contexts))
    push!(arm_choices,     rand(rng, 1:n_arms))
    
    # Calculate the deterministic part of the reward (μ_t = x_t^T β_t)
    # Here we're simplifying by using arms directly as β_t instead of sampling from N(θ_k, Λ_k^-1)
    mean_reward = dot(arms[arm_choices[end]], contexts[context_choices[end]])
    
    # Add Gaussian noise (r_t ~ N(μ_t, τ^-1))
    noisy_reward = mean_reward + randn(rng) * noise_sd
    push!(rewards, noisy_reward)
end

# Create more informative and visually appealing plots
p1 = scatter(1:n_samples, context_choices, 
             label="Context Choices", 
             title="Context Selection Over Time",
             xlabel="Sample", ylabel="Context ID",
             marker=:circle, markersize=6, 
             color=:blue, alpha=0.7,
             legend=:topright)

p2 = scatter(1:n_samples, arm_choices, 
             label="Arm Choices", 
             title="Arm Selection Over Time",
             xlabel="Sample", ylabel="Arm ID",
             marker=:diamond, markersize=6, 
             color=:red, alpha=0.7,
             legend=:topright)

p3 = plot(1:n_samples, rewards, 
          label="Rewards", 
          title="Rewards Over Time",
          xlabel="Sample", ylabel="Reward Value",
          linewidth=2, marker=:star, markersize=4, 
          color=:green, alpha=0.8,
          legend=:topright)

# Add a horizontal line at mean reward
hline!(p3, [mean(rewards)], label="Mean Reward", linestyle=:dash, linewidth=2, color=:black)

# Combine plots with a title
plot(p1, p2, p3, 
     layout=(3, 1), 
     size=(800, 600), 
     margin=5Plots.mm,
     plot_title="Contextual Bandit Experiment Data")

@model function contextual_bandit_simplified(n_arms, priors, past_rewards, past_choices, past_contexts)
    local θ
    local γ
    local τ
    
    # Prior for each arm's parameters
    for k in 1:n_arms
        θ[k] ~ priors[:θ][k]
        γ[k] ~ priors[:γ][k]
    end

    # Prior for the noise precision
    τ ~ priors[:τ]

    # Model for past observations
    for n in eachindex(past_rewards)
        arm_vals[n] ~ NormalMixture(switch = past_choices[n], m = θ, p = γ)
        past_rewards[n] ~ Normal(μ=dot(arm_vals[n], past_contexts[n]), γ=τ)
    end
end

priors_rng = StableRNG(42)
priors = Dict(
    :θ => [MvNormalMeanPrecision(randn(priors_rng, context_dim), diagm(ones(context_dim))) for _ in 1:n_arms], 
    :γ => [Wishart(context_dim + 1, diagm(ones(context_dim))) for _ in 1:n_arms], 
    :τ => GammaShapeRate(1.0, 1.0)
)

function run_inference(; n_arms, priors, past_rewards, past_choices, past_contexts, iterations = 50, free_energy = true)
    init = @initialization begin
        q(θ) = priors[:θ]
        q(γ) = priors[:γ]
        q(τ) = priors[:τ]
    end

    return infer(
        model = contextual_bandit_simplified(
            n_arms = n_arms, 
            priors = priors,
        ), 
        data  = (
            past_rewards  = past_rewards, 
            past_choices  = past_choices, 
            past_contexts = past_contexts
        ), 
        constraints   = MeanField(),
        initialization = init,
        showprogress = true, 
        iterations  = iterations, 
        free_energy = free_energy
    )

end

# Convert to the required types for the model
rewards_data = Float64.(rewards)
contexts_data = Vector{Float64}[contexts[idx] for idx in context_choices]
arm_choices_data = [[Float64(k == chosen_arm) for k in 1:n_arms] for chosen_arm in arm_choices];

result = run_inference(
    n_arms = n_arms, 
    priors = priors, 
    past_rewards = rewards_data, 
    past_choices = arm_choices_data, 
    past_contexts = contexts_data, 
    iterations = 50, 
    free_energy = false
)

# Diagnostics of inferred arms

# MSE of inferred arms
mse_arms = mean(mean((mean.(result.posteriors[:θ][end])[i] .- arms[i]).^2) for i in eachindex(arms))
println("MSE of inferred arms: $mse_arms")

println("True mean rewards:")
println(arms[1]'*contexts[1])
println(arms[2]'*contexts[2])
println(arms[1]'*contexts[2])
println(arms[2]'*contexts[1])
println("Inferred mean rewards:")
println(mean.(result.posteriors[:θ][end])[1]'*contexts[1])
println(mean.(result.posteriors[:θ][end])[2]'*contexts[2])
println(mean.(result.posteriors[:θ][end])[1]'*contexts[2])
println(mean.(result.posteriors[:θ][end])[2]'*contexts[1])

println("Precisions of mixtures:")
println(repr(MIME"text/plain"(), mean(result.posteriors[:γ][end][1])))
println(repr(MIME"text/plain"(), mean(result.posteriors[:γ][end][2])))


function random_strategy(; rng, n_arms)
    chosen_arm = rand(rng, 1:n_arms)
    return chosen_arm
end

function thompson_strategy(; rng, n_arms, current_context, posteriors)
    # Thompson Sampling: Sample parameter vectors and choose best arm
    expected_rewards = zeros(n_arms)
    for k in 1:n_arms
        # Sample parameters from posterior
        theta_sample = rand(rng, posteriors[:θ][k])
        expected_rewards[k] = dot(theta_sample, current_context)
    end
    
    # Choose best arm based on sampled parameters
    chosen_arm = argmax(expected_rewards)

    # Randomly choose an arm with 20% probability to explore
    if rand(rng) < 0.20
        chosen_arm = rand(rng, 1:n_arms)
    end

    return chosen_arm
end

@model function contextual_bandit_predictive(reward, priors, current_context)
    local θ
    local γ
    local τ

    # Prior for each arm's parameters
    for k in 1:n_arms
        θ[k] ~ priors[:θ][k]
        γ[k] ~ priors[:γ][k]
    end

    τ ~ priors[:τ]

    chosen_arm ~ Categorical(ones(n_arms) ./ n_arms)
    arm_vals ~ NormalMixture(switch=chosen_arm, m=θ, p=γ)
    reward ~ Normal(μ=dot(arm_vals, current_context), γ=τ)
end

function predictive_strategy(; rng, n_arms, current_context, posteriors)

    priors = Dict(
        :θ => posteriors[:θ],
        :γ => posteriors[:γ],
        :τ => posteriors[:τ]
    )

    init = @initialization begin
        q(θ) = priors[:θ]
        q(τ) = priors[:τ]
        q(γ) = priors[:γ]
        q(chosen_arm) = Categorical(ones(n_arms) ./ n_arms)
    end

    result = infer(
        model=contextual_bandit_predictive(
            priors=priors,
            current_context=current_context
        ),
        data=(reward=maximum(rewards),),
        constraints=MeanField(),
        initialization=init,
        showprogress=true,
        iterations=50,
    )

    chosen_arm = argmax(probvec(result.posteriors[:chosen_arm][end]))

    return chosen_arm
end

# Helper functions
function select_context(rng, n_contexts)
    idx = rand(rng, 1:n_contexts)
    return (index = idx, value = contexts[idx])
end

function plan(rng, n_arms, context, posteriors)
    # Generate actions from different strategies
    return Dict(
        :random => random_strategy(rng = rng, n_arms = n_arms),
        :thompson => thompson_strategy(rng = rng, n_arms = n_arms, current_context = context, posteriors = posteriors),
        :predictive => predictive_strategy(rng = rng, n_arms = n_arms, current_context = context, posteriors = posteriors)
    )
end

function act(rng, strategies)
    # Here one would choose which strategy to actually follow
    # For this simulation, we're evaluating all in parallel
    # In a real scenario, one might return just one: return strategies[:thompson]
    return strategies
end

function observe(rng, strategies, context, arms, noise_sd)
    rewards = Dict()
    for (strategy, arm_idx) in strategies
        rewards[strategy] = dot(arms[arm_idx], context) + randn(rng) * noise_sd
    end
    return rewards
end

function learn(rng, n_arms, posteriors, past_rewards, past_choices, past_contexts)
    # Note that we don't do any forgetting here which might be useful for long-term learning
    # Prepare priors from current posteriors
    priors = Dict(:θ => posteriors[:θ], :τ => posteriors[:τ], :γ => posteriors[:γ])
    
    # Default initialization
    init = @initialization begin
        q(θ) = priors[:θ]
        q(τ) = priors[:τ]
        q(γ) = priors[:γ]
    end
    
    # Run inference
    results = infer(
        model = contextual_bandit_simplified(
            n_arms = n_arms, 
            priors = priors,
        ), 
        data = (
            past_rewards = past_rewards, 
            past_choices = past_choices,
            past_contexts = past_contexts, 
        ), 
        returnvars = KeepLast(),
        constraints = MeanField(),
        initialization = init,
        iterations = 50, 
        free_energy = false
    )
    
    return results.posteriors
end

function keep_history!(n_arms, history, strategies, rewards, context, posteriors)
    # Update choices
    for (strategy, arm_idx) in strategies
        push!(history[:choices][strategy], [Float64(k == arm_idx) for k in 1:n_arms])
    end
    
    # Update rewards
    for (strategy, reward) in rewards
        push!(history[:rewards][strategy], reward)
    end
    
    # Update real history - using predictive strategy as the actual choice
    push!(history[:real][:rewards], last(history[:rewards][:predictive]))
    push!(history[:real][:choices], last(history[:choices][:predictive]))
    
    # Update contexts
    push!(history[:contexts][:values], context.value)
    push!(history[:contexts][:indices], context.index)
    
    # Update posteriors
    push!(history[:posteriors], deepcopy(posteriors))
end

function run_bandit_simulation(n_epochs, window_length, n_arms, n_contexts, context_dim)
    rng = StableRNG(42)

    # Initialize histories with empty arrays, removing the references to undefined variables
    history = Dict(
        :rewards => Dict(:random => [], :thompson => [], :predictive => []),
        :choices => Dict(:random => [], :thompson => [], :predictive => []),
        :real => Dict(:rewards => [], :choices => []),
        :contexts => Dict(:values => [], :indices => []),
        :posteriors => []
    )

    # Initialize prior posterior as uninformative 
    posteriors = Dict(:θ => [MvNormalMeanPrecision(randn(rng, context_dim), diagm(ones(context_dim))) for _ in 1:n_arms], 
                      :γ => [Wishart(context_dim + 1, diagm(ones(context_dim))) for _ in 1:n_arms], 
                      :τ => GammaShapeRate(1.0, 1.0))

    @showprogress for epoch in 1:n_epochs
        # 1. PLAN - Run different strategies
        current_context = select_context(rng, n_contexts)

        strategies = plan(rng, n_arms, current_context.value, posteriors)
        
        # 2. ACT - In this simulation, we're evaluating all strategies in parallel
        # In a real scenario, you might choose one strategy here
        chosen_arm = act(rng, strategies)
        
        # 3. OBSERVE - Get rewards for all strategies
        rewards = observe(rng, strategies, current_context.value, arms, noise_sd)
        
        # 4. LEARN - Update posteriors based on history
        # Only try to learn if we have collected data
        if length(history[:real][:rewards]) > 0
            data_idx = max(1, length(history[:real][:rewards]) - window_length + 1):length(history[:real][:rewards])
            
            posteriors = learn(
                rng,
                n_arms,
                posteriors,
                history[:real][:rewards][data_idx],
                history[:real][:choices][data_idx],
                history[:contexts][:values][data_idx]
            )

        end
        
        # 5. KEEP HISTORY - Record all results
        keep_history!(n_arms, history, strategies, rewards, current_context, posteriors)
    end
    
    return history
end

# Run the simulation
n_epochs = 500
window_length = 10

history = run_bandit_simulation(n_epochs, window_length, n_arms, n_contexts, context_dim)

function print_summary_statistics(history, n_epochs)
    # Additional summary statistics
    println("Random strategy cumulative reward: $(sum(history[:rewards][:random]))")
    println("Thompson strategy cumulative reward: $(sum(history[:rewards][:thompson]))")
    println("Predictive strategy cumulative reward: $(sum(history[:rewards][:predictive]))")

    println("Results after $n_epochs epochs:")
    println("Random strategy average reward: $(mean(history[:rewards][:random]))")
    println("Thompson strategy average reward: $(mean(history[:rewards][:thompson]))")
    println("Predictive strategy average reward: $(mean(history[:rewards][:predictive]))")
end

# Print the summary statistics
print_summary_statistics(history, n_epochs)

function plot_arm_distribution(history, n_arms)
    # Count frequency of each arm selection
    arm_counts_random = [count(==(i), argmax.(history[:choices][:random])) for i in 1:n_arms]
    arm_counts_thompson = [count(==(i), argmax.(history[:choices][:thompson])) for i in 1:n_arms]
    arm_counts_predictive = [count(==(i), argmax.(history[:choices][:predictive])) for i in 1:n_arms]

    # Create grouped bar plot
    bar_plot = groupedbar(
        ["Random", "Thompson", "Predictive"],
        [arm_counts_random arm_counts_thompson arm_counts_predictive]',
        title="Arm Selection Distribution",
        xlabel="Strategy",
        ylabel="Selection Count",
        bar_position=:dodge,
        bar_width=0.8,
        alpha=0.7
    )
    
    return bar_plot
end

# Plot arm distribution
arm_distribution_plot = plot_arm_distribution(history, n_arms)
display(arm_distribution_plot)

function calculate_improvements(history)
    # Get final average rewards
    final_random_avg = mean(history[:rewards][:random])
    final_thompson_avg = mean(history[:rewards][:thompson])
    final_predictive_avg = mean(history[:rewards][:predictive])

    # Improvements over random baseline
    thompson_improvement = (final_thompson_avg - final_random_avg) / abs(final_random_avg) * 100
    predictive_improvement = (final_predictive_avg - final_random_avg) / abs(final_random_avg) * 100

    println("Thompson sampling improves over random by $(round(thompson_improvement, digits=2))%")
    println("Predictive strategy improves over random by $(round(predictive_improvement, digits=2))%")
    
    return Dict(
        :thompson => thompson_improvement,
        :predictive => predictive_improvement
    )
end

# Calculate and display improvements
improvements = calculate_improvements(history)

function plot_moving_averages(history, n_epochs, ma_window=20)
    # Calculate moving average rewards
    ma_rewards_random = [mean(history[:rewards][:random][max(1,i-ma_window+1):i]) for i in 1:n_epochs]
    ma_rewards_thompson = [mean(history[:rewards][:thompson][max(1,i-ma_window+1):i]) for i in 1:n_epochs]
    ma_rewards_predictive = [mean(history[:rewards][:predictive][max(1,i-ma_window+1):i]) for i in 1:n_epochs]

    # Plot moving average
    plot(1:n_epochs, [ma_rewards_random, ma_rewards_thompson, ma_rewards_predictive], 
         label=["Random" "Thompson" "Predictive"],
         title="Moving Average Reward", 
         xlabel="Epoch", ylabel="Average Reward",
         lw=2)
end

# Plot moving averages
plot_moving_averages(history, n_epochs)

function create_comprehensive_plots(history, window=100, k=10)
    # Create a better color palette
    colors = palette(:tab10)

    # Plot 1: Arm choices comparison (every k-th point)
    p1 = plot(title="Arm Choices Over Time", xlabel="Epoch", ylabel="Arm Index", 
              legend=:outertopright, dpi=300)
    plot!(p1, argmax.(history[:choices][:random][1:k:end]), label="Random", color=colors[1], 
          markershape=:circle, markersize=3, alpha=0.5, linewidth=0)
    plot!(p1, argmax.(history[:choices][:thompson][1:k:end]), label="Thompson", color=colors[2], 
          markershape=:circle, markersize=3, alpha=0.5, linewidth=0)
    plot!(p1, argmax.(history[:choices][:predictive][1:k:end]), label="Predictive", color=colors[3], 
          markershape=:circle, markersize=3, alpha=0.5, linewidth=0)

    # Plot 2: Context values (every k-th point)
    p2 = plot(title="Context Changes", xlabel="Epoch", ylabel="Context Index", 
              legend=false, dpi=300)
    plot!(p2, history[:contexts][:indices][1:k:end], color=colors[4], linewidth=1.5)

    # Plot 3: Reward comparison (every k-th point)
    p3 = plot(title="Rewards by Strategy", xlabel="Epoch", ylabel="Reward Value", 
              legend=:outertopright, dpi=300)
    plot!(p3, history[:rewards][:random][1:k:end], label="Random", color=colors[1], linewidth=1.5, alpha=0.7)
    plot!(p3, history[:rewards][:thompson][1:k:end], label="Thompson", color=colors[2], linewidth=1.5, alpha=0.7)
    plot!(p3, history[:rewards][:predictive][1:k:end], label="Predictive", color=colors[3], linewidth=1.5, alpha=0.7)

    # Plot 4: Cumulative rewards (every k-th point)
    cumul_random = cumsum(history[:rewards][:random])[1:k:end]
    cumul_thompson = cumsum(history[:rewards][:thompson])[1:k:end]
    cumul_predictive = cumsum(history[:rewards][:predictive])[1:k:end]

    p4 = plot(title="Cumulative Rewards", xlabel="Epoch", ylabel="Cumulative Reward", 
              legend=:outertopright, dpi=300)
    plot!(p4, cumul_random, label="Random", color=colors[1], linewidth=2)
    plot!(p4, cumul_thompson, label="Thompson", color=colors[2], linewidth=2)
    plot!(p4, cumul_predictive, label="Predictive", color=colors[3], linewidth=2)

    # Plot 5: Moving average rewards (every k-th point)
    ma_random = [mean(history[:rewards][:random][max(1,i-window+1):i]) for i in 1:length(history[:rewards][:random])][1:k:end]
    ma_thompson = [mean(history[:rewards][:thompson][max(1,i-window+1):i]) for i in 1:length(history[:rewards][:thompson])][1:k:end]
    ma_predictive = [mean(history[:rewards][:predictive][max(1,i-window+1):i]) for i in 1:length(history[:rewards][:predictive])][1:k:end]

    p5 = plot(title="$window-Epoch Moving Average Rewards", xlabel="Epoch", ylabel="Avg Reward", 
              legend=:outertopright, dpi=300)
    plot!(p5, ma_random, label="Random", color=colors[1], linewidth=2)
    plot!(p5, ma_thompson, label="Thompson", color=colors[2], linewidth=2)
    plot!(p5, ma_predictive, label="Predictive", color=colors[3], linewidth=2)

    # Combine all plots with a title
    combined_plot = plot(p1, p2, p3, p4, p5, 
         layout=(5, 1), 
         size=(900, 900), 
         plot_title="Bandit Strategies Comparison (shows every $k th point)", 
         plot_titlefontsize=14,
         left_margin=10Plots.mm,
         bottom_margin=10Plots.mm)
         
    return combined_plot
end

create_comprehensive_plots(history, window_length, 10)  # Using k=10 for prettier plots