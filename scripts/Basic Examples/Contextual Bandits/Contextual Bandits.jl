# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Basic Examples/Contextual Bandits/Contextual Bandits.ipynb
# by notebooks_to_scripts.jl at 2025-08-07T12:32:28.592
#
# Source notebook: Contextual Bandits.ipynb

using RxInfer, Distributions, LinearAlgebra, Plots, StatsPlots, ProgressMeter, StableRNGs, Random

using StableRNGs

# Random number generator 
rng = StableRNG(42)

# Data generation parameters
n_train_samples = 300
n_test_samples = 100
n_total_samples = n_train_samples + n_test_samples
n_arms = 10
n_contexts = 50
context_dim = 20
noise_sd = 0.1

# Generate true arm parameters (θ_k in the model description)
arms = [randn(rng, context_dim) for _ in 1:n_arms]

# Generate context feature vectors with missing values
contexts = []
for i in 1:n_contexts
    # Create a vector that can hold both Float64 and Missing values
    context = Vector{Union{Float64,Missing}}(undef, context_dim)

    # Fill with random values initially
    for j in 1:context_dim
        context[j] = randn(rng)
    end

    # Randomly introduce missing values in some contexts
    if rand(rng) < 0.4  # 40% of contexts will have some missing values
        n_missing = rand(rng, 1:2)  # 1-2 missing values per context

        # Simple approach: randomly select indices for missing values
        missing_indices = []
        while length(missing_indices) < n_missing
            idx = rand(rng, 1:context_dim)
            if !(idx in missing_indices)
                push!(missing_indices, idx)
            end
        end

        for idx in missing_indices
            context[idx] = missing
        end
    end

    push!(contexts, context)
end

# Synthetic reward function (reusable)
function compute_reward(arm_params, context_vec, noise_sd=0.0; rng=nothing)
    """
    Compute reward for given arm parameters and context.

    Args:
        arm_params: Vector of arm parameters
        context_vec: Context vector (may contain missing values)
        noise_sd: Standard deviation of Gaussian noise to add
        rng: Random number generator (optional, for reproducible noise)

    Returns:
        Scalar reward value
    """
    # Calculate the deterministic part of the reward, handling missing values
    mean_reward = 0.0
    valid_dims = 0

    for j in 1:length(context_vec)
        if !ismissing(context_vec[j])
            mean_reward += arm_params[j] * context_vec[j]
            valid_dims += 1
        end
    end

    # Normalize by the number of valid dimensions to maintain similar scale
    if valid_dims > 0
        mean_reward = mean_reward * (length(context_vec) / valid_dims)
    end

    # Add Gaussian noise if requested
    if noise_sd > 0
        if rng !== nothing
            mean_reward += randn(rng) * noise_sd
        else
            mean_reward += randn() * noise_sd
        end
    end

    return mean_reward
end

# Generate training and test data
function generate_bandit_data(n_samples, arms, contexts, noise_sd; rng, start_idx=1)
    """
    Generate bandit data for given number of samples.

    Returns:
        (arm_choices, context_choices, rewards)
    """
    arm_choices = []
    context_choices = []
    rewards = []

    for i in 1:n_samples
        # Randomly select a context and an arm
        push!(context_choices, rand(rng, 1:length(contexts)))
        push!(arm_choices, rand(rng, 1:length(arms)))

        # Get the selected context and arm
        selected_context = contexts[context_choices[end]]
        selected_arm = arms[arm_choices[end]]

        # Compute reward using the synthetic reward function
        reward = compute_reward(selected_arm, selected_context, noise_sd; rng=rng)
        push!(rewards, reward)
    end

    return arm_choices, context_choices, rewards
end

# Generate training data
println("Generating training data...")
train_arm_choices, train_context_choices, train_rewards = generate_bandit_data(
    n_train_samples, arms, contexts, noise_sd; rng=rng
)

# Generate test data
println("Generating test data...")
test_arm_choices, test_context_choices, test_rewards = generate_bandit_data(
    n_test_samples, arms, contexts, noise_sd; rng=rng
)

# Display information about the generated data
println("\nDataset Summary:")
println("Training samples: $n_train_samples")
println("Test samples: $n_test_samples")
println("Total contexts: $(length(contexts))")
println("Number of contexts with missing values: ", sum(any(ismissing, ctx) for ctx in contexts))
println("Arms: $n_arms")
println("Context dimension: $context_dim")
println("Noise standard deviation: $noise_sd")

# Show examples of contexts with missing values
println("\nExamples of contexts with missing values:")
count = 0
for (i, ctx) in enumerate(contexts)
    if any(ismissing, ctx) && count < 3  # Show first 3 examples
        println("Context $i: $ctx")
        global count += 1
    end
end

# Show sample data
println("\nTraining data samples (first 5):")
for i in 1:min(5, length(train_rewards))
    println("Sample $i: Arm=$(train_arm_choices[i]), Context=$(train_context_choices[i]), Reward=$(round(train_rewards[i], digits=4))")
end

println("\nTest data samples (first 5):")
for i in 1:min(5, length(test_rewards))
    println("Sample $i: Arm=$(test_arm_choices[i]), Context=$(test_context_choices[i]), Reward=$(round(test_rewards[i], digits=4))")
end

# Verify the reward function works correctly
println("\nTesting reward function:")
test_context = contexts[1]
test_arm = arms[1]
deterministic_reward = compute_reward(test_arm, test_context, 0.0)  # No noise
noisy_reward = compute_reward(test_arm, test_context, noise_sd; rng=rng)  # With noise
println("Deterministic reward: $(round(deterministic_reward, digits=4))")
println("Noisy reward: $(round(noisy_reward, digits=4))")

function create_bandit_plots(arm_choices, context_choices, rewards, title_prefix, color_scheme)
    p1 = scatter(1:length(context_choices), context_choices,
        label="Context Choices",
        title="$title_prefix: Context Selection",
        xlabel="Sample", ylabel="Context ID",
        marker=:circle, markersize=6,
        color=color_scheme[:context], alpha=0.7)

    p2 = scatter(1:length(arm_choices), arm_choices,
        label="Arm Choices",
        title="$title_prefix: Arm Selection",
        xlabel="Sample", ylabel="Arm ID",
        marker=:diamond, markersize=6,
        color=color_scheme[:arm], alpha=0.7)

    p3 = plot(1:length(rewards), rewards,
        label="Rewards",
        title="$title_prefix: Rewards",
        xlabel="Sample", ylabel="Reward Value",
        linewidth=2, marker=:star, markersize=4,
        color=color_scheme[:reward], alpha=0.8)

    hline!(p3, [mean(rewards)], label="Mean Reward",
        linestyle=:dash, linewidth=2, color=:black)

    return plot(p1, p2, p3, layout=(3, 1), size=(800, 600))
end

# Create training plots
train_colors = Dict(:context => :blue, :arm => :red, :reward => :green)
train_plot = create_bandit_plots(train_arm_choices, train_context_choices, train_rewards,
    "Training Data", train_colors)

# Create test plots  
test_colors = Dict(:context => :lightblue, :arm => :pink, :reward => :lightgreen)
test_plot = create_bandit_plots(test_arm_choices, test_context_choices, test_rewards,
    "Test Data", test_colors)

# Display both
plot(train_plot, test_plot, layout=(1, 2), size=(1600, 600),
    plot_title="Contextual Bandit Experiment: Training and Test Data")

@model function conditional_regression(n_arms, priors, past_rewards, past_choices, past_contexts)
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
        arm_vals[n] ~ NormalMixture(switch=past_choices[n], m=θ, p=γ)
        latent_context[n] ~ past_contexts[n]
        past_rewards[n] ~ softdot(arm_vals[n], latent_context[n], τ)
    end
end

priors_rng = StableRNG(42)
priors = Dict(
    :θ => [MvNormalMeanPrecision(randn(priors_rng, context_dim), diagm(ones(context_dim))) for _ in 1:n_arms],
    :γ => [Wishart(context_dim + 1, diagm(ones(context_dim))) for _ in 1:n_arms],
    :τ => GammaShapeRate(1.0, 1.0)
)

function run_inference(; n_arms, priors, past_rewards, past_choices, past_contexts, iterations=50, free_energy=true)
    init = @initialization begin
        q(θ) = priors[:θ]
        q(γ) = priors[:γ]
        q(τ) = priors[:τ]
        q(latent_context) = MvNormalMeanPrecision(zeros(context_dim), Diagonal(ones(context_dim)))
    end

    return infer(
        model=conditional_regression(
            n_arms=n_arms,
            priors=priors,
            past_contexts=past_contexts,
        ),
        data=(
            past_rewards=past_rewards,
            past_choices=past_choices,
        ),
        constraints=MeanField(),
        initialization=init,
        showprogress=true,
        iterations=iterations,
        free_energy=free_energy
    )

end

# Utility function to convert context with missing values to MvNormal distribution
function context_to_mvnormal(context_vec; tiny_precision=1e-6, huge_precision=1e6)
    """
    Convert a context vector (potentially with missing values) to MvNormal distribution.

    Args:
        context_vec: Vector that may contain missing values
        tiny_v: Small variance for known values (high precision)
        huge_var: Large variance for missing values (low precision)

    Returns:
        MvNormal distribution
    """
    context_mean = Vector{Float64}(undef, length(context_vec))
    context_precision = Vector{Float64}(undef, length(context_vec))

    for j in 1:length(context_vec)
        if ismissing(context_vec[j])
            context_mean[j] = 0.0
            context_precision[j] = tiny_precision
        else
            context_mean[j] = context_vec[j]
            context_precision[j] = huge_precision
        end
    end

    return MvNormalMeanPrecision(context_mean, Diagonal(context_precision))
end

# Convert to the required types for the model (TRAINING DATA ONLY)
rewards_data = Float64.(train_rewards)

# Parameters for the covariance matrix
tiny_precision = 1e-6   # Very high precision (small variance) for known values
huge_precision = 1e6  # Very low precision (large variance) for missing values

contexts_data = [
    let context = contexts[idx]
        context_to_mvnormal(context; tiny_precision=tiny_precision, huge_precision=huge_precision)
    end
    for idx in train_context_choices  # Use training context choices
]

arm_choices_data = [[Float64(k == chosen_arm) for k in 1:n_arms] for chosen_arm in train_arm_choices];  # Use training arm choices

result = run_inference(
    n_arms=n_arms,
    priors=priors,
    past_rewards=rewards_data,
    past_choices=arm_choices_data,
    past_contexts=contexts_data,
    iterations=20,
    free_energy=false
)

# Diagnostics of inferred arms

# 1. MSE of inferred arms coefficients
inferred_arms = mean.(result.posteriors[:θ][end])
mse_arms = mean(mean((inferred_arms[i] .- arms[i]) .^ 2) for i in eachindex(arms))
println("MSE of inferred arm coefficients: $mse_arms")

# Function to compute predicted rewards using softdot rules
function compute_predicted_rewards_with_variance(
    arm_posteriors,
    precision_posterior,
    eval_arm_choices,
    eval_context_choices,
    eval_rewards
)
    predicted_rewards = []
    reward_variances = []

    for i in 1:length(eval_rewards)  # Evaluate on all samples in the evaluation set
        arm_idx = eval_arm_choices[i]
        ctx_idx = eval_context_choices[i]

        # Get the posterior distributions
        q_arm = arm_posteriors[arm_idx]  # Posterior over arm parameters
        q_precision = precision_posterior  # Precision posterior

        # Get the actual context and convert to MvNormal
        actual_context = contexts[ctx_idx]
        q_context = context_to_mvnormal(actual_context)

        # Use softdot rule to compute predicted reward distribution
        predicted_reward_dist = NormalMeanPrecision(
            mean(q_arm)' * mean(q_context),
            mean(q_precision)
        )

        push!(predicted_rewards, mean(predicted_reward_dist))
        push!(reward_variances, var(predicted_reward_dist))
    end

    return predicted_rewards, reward_variances
end

# Function to display evaluation results
function display_evaluation_results(predicted_rewards, reward_variances, actual_rewards,
    arm_choices, context_choices, dataset_name)
    println("\n$dataset_name Evaluation Results:")
    println("Sample | Actual Reward | Predicted Mean | Predicted Std | Arm | Context")
    println("-------|---------------|----------------|---------------|-----|--------")

    for i in 1:min(10, length(predicted_rewards))
        actual = actual_rewards[i]
        pred_mean = predicted_rewards[i]
        pred_std = sqrt(reward_variances[i])
        arm_idx = arm_choices[i]
        ctx_idx = context_choices[i]

        println("$(lpad(i, 6)) | $(rpad(round(actual, digits=4), 13)) | $(rpad(round(pred_mean, digits=4), 14)) | $(rpad(round(pred_std, digits=4), 13)) | $(lpad(arm_idx, 3)) | $(lpad(ctx_idx, 7))")
    end

    # Compute prediction metrics
    prediction_mse = mean((predicted_rewards .- actual_rewards) .^ 2)
    println("\n$dataset_name Prediction MSE: $prediction_mse")

    # Compute log-likelihood of actual rewards under predicted distributions
    log_likelihood = sum(
        logpdf(Normal(predicted_rewards[i], sqrt(reward_variances[i])), actual_rewards[i])
        for i in 1:length(predicted_rewards)
    )
    println("$dataset_name Average log-likelihood: $(log_likelihood / length(predicted_rewards))")

    return prediction_mse, log_likelihood / length(predicted_rewards)
end

# Evaluate on TRAINING data
println("=== TRAINING DATA EVALUATION ===")
train_predicted_rewards, train_reward_variances = compute_predicted_rewards_with_variance(
    result.posteriors[:θ][end],  # Use full posteriors
    result.posteriors[:τ][end],  # Precision posterior
    train_arm_choices,
    train_context_choices,
    train_rewards
)

train_mse, train_ll = display_evaluation_results(
    train_predicted_rewards,
    train_reward_variances,
    train_rewards,
    train_arm_choices,
    train_context_choices,
    "Training"
)

# Evaluate on TEST data
println("\n=== TEST DATA EVALUATION ===")
test_predicted_rewards, test_reward_variances = compute_predicted_rewards_with_variance(
    result.posteriors[:θ][end],  # Use full posteriors
    result.posteriors[:τ][end],  # Precision posterior
    test_arm_choices,
    test_context_choices,
    test_rewards
)

test_mse, test_ll = display_evaluation_results(
    test_predicted_rewards,
    test_reward_variances,
    test_rewards,
    test_arm_choices,
    test_context_choices,
    "Test"
)

# Summary comparison
println("\n=== SUMMARY COMPARISON ===")
println("Dataset    | MSE      | Log-Likelihood")
println("-----------|----------|---------------")
println("Training   | $(rpad(round(train_mse, digits=4), 8)) | $(round(train_ll, digits=4))")
println("Test       | $(rpad(round(test_mse, digits=4), 8)) | $(round(test_ll, digits=4))")

if test_mse > train_mse * 1.2
    println("\nNote: Test MSE is significantly higher than training MSE - possible overfitting")
elseif test_mse < train_mse * 0.8
    println("\nNote: Test MSE is lower than training MSE - good generalization!")
else
    println("\nNote: Test and training performance are similar - good generalization")
end

# Additional diagnostics
println("\n=== ADDITIONAL DIAGNOSTICS ===")

# Show precision/variance information
precision_posterior = result.posteriors[:τ][end]
println("Inferred noise precision: mean=$(round(mean(precision_posterior), digits=4)), " *
        "std=$(round(std(precision_posterior), digits=4))")
println("Inferred noise variance: $(round(1/mean(precision_posterior), digits=4))")
println("True noise variance: $(round(noise_sd^2, digits=4))")

# Show arm coefficient statistics
println("\nArm coefficient comparison:")
for i in eachindex(arms)
    true_arm = arms[i]
    inferred_arm_posterior = result.posteriors[:θ][end][i]
    inferred_arm_mean = mean(inferred_arm_posterior)
    println("Arm $i:")
    println("  True:     $(round.(true_arm, digits=3))")
    println("  Inferred: $(round.(inferred_arm_mean, digits=3))")
    println("  MSE:      $(round(mean((inferred_arm_mean .- true_arm).^2), digits=4))")
end

println("\nNumber of contexts with missing values: ", sum(any(ismissing, ctx) for ctx in contexts))

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
        # context might have missing values, so we use the mean of the context
        augmented_context = mean(context_to_mvnormal(current_context))
        expected_rewards[k] = dot(theta_sample, augmented_context)
    end

    # Choose best arm based on sampled parameters
    chosen_arm = argmax(expected_rewards)

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
    latent_context ~ current_context
    reward ~ softdot(arm_vals, latent_context, τ)
end

function predictive_strategy(; rng, n_arms, current_context, posteriors)

    priors = Dict(
        :θ => posteriors[:θ],
        :γ => posteriors[:γ],
        :τ => posteriors[:τ]
    )

    latent_context = context_to_mvnormal(current_context)

    init = @initialization begin
        q(θ) = priors[:θ]
        q(τ) = priors[:τ]
        q(γ) = priors[:γ]
        q(latent_context) = latent_context
        q(chosen_arm) = Categorical(ones(n_arms) ./ n_arms)
    end

    result = infer(
        model=contextual_bandit_predictive(
            priors=priors,
            current_context=latent_context
        ),
        data=(reward=10maximum(train_rewards),),
        constraints=MeanField(),
        initialization=init,
        showprogress=true,
        iterations=20,
    )

    chosen_arm = argmax(probvec(result.posteriors[:chosen_arm][end]))

    return chosen_arm
end

# Helper functions
function select_context(rng, n_contexts)
    idx = rand(rng, 1:n_contexts)
    return (index=idx, value=contexts[idx])
end

function plan(rng, n_arms, context, posteriors)
    # Generate actions from different strategies
    return Dict(
        :random => random_strategy(rng=rng, n_arms=n_arms),
        :thompson => thompson_strategy(rng=rng, n_arms=n_arms, current_context=context, posteriors=posteriors),
        :predictive => predictive_strategy(rng=rng, n_arms=n_arms, current_context=context, posteriors=posteriors)
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
        rewards[strategy] = compute_reward(arms[arm_idx], context, noise_sd)
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
        q(latent_context) = MvNormalMeanPrecision(zeros(context_dim), Diagonal(ones(context_dim)))
    end

    # Run inference
    results = infer(
        model=conditional_regression(
            n_arms=n_arms,
            priors=priors,
            past_contexts=context_to_mvnormal.(past_contexts),
        ),
        data=(
            past_rewards=past_rewards,
            past_choices=past_choices,
        ),
        returnvars=KeepLast(),
        constraints=MeanField(),
        initialization=init,
        iterations=20,
        free_energy=false
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
        if mod(epoch, window_length) == 0 && length(history[:real][:rewards]) > 0
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
n_epochs = 5000
window_length = 100

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
    # Extract choices
    random_choices = history[:choices][:random]
    thompson_choices = history[:choices][:thompson]
    predictive_choices = history[:choices][:predictive]

    # Convert to arm indices
    random_arms = [argmax(choice) for choice in random_choices]
    thompson_arms = [argmax(choice) for choice in thompson_choices]
    predictive_arms = [argmax(choice) for choice in predictive_choices]

    # Count frequencies
    arm_counts_random = zeros(Int, n_arms)
    arm_counts_thompson = zeros(Int, n_arms)
    arm_counts_predictive = zeros(Int, n_arms)

    for arm in random_arms
        arm_counts_random[arm] += 1
    end

    for arm in thompson_arms
        arm_counts_thompson[arm] += 1
    end

    for arm in predictive_arms
        arm_counts_predictive[arm] += 1
    end

    # Create grouped bar plot
    bar_plot = groupedbar(
        1:n_arms,
        [arm_counts_random arm_counts_thompson arm_counts_predictive],
        title="Arm Selection Distribution",
        xlabel="Arm Index",
        ylabel="Selection Count",
        bar_position=:dodge,
        bar_width=0.8,
        alpha=0.7,
        legend=:topright,
        labels=["Random" "Thompson" "Predictive"]
    )

    return bar_plot
end

# Plot arm distribution
arm_distribution_plot = plot_arm_distribution(history, 10)
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

function analyze_doubly_robust_uplift(history, target_strategy=:predictive, baseline_strategy=:random)
    """
    Compute doubly robust uplift estimate from simulation history
    """
    target_rewards = history[:rewards][target_strategy]
    baseline_rewards = history[:rewards][baseline_strategy]

    # Simple Direct Method - just difference in average rewards
    direct_method = mean(target_rewards) - mean(baseline_rewards)

    # For IPW, we use the fact that all strategies were evaluated on same contexts
    # So propensity is uniform across arms for random, and we can estimate others
    n_epochs = length(target_rewards)
    n_arms = length(history[:choices][target_strategy][1])

    # IPW correction (simplified since we have parallel evaluation)
    ipw_correction = 0.0
    for i in 1:n_epochs
        # Get actual choice and reward (using predictive as "real" policy)
        real_choice = history[:choices][:predictive][i]
        real_reward = history[:rewards][:predictive][i]

        target_choice = history[:choices][target_strategy][i]
        baseline_choice = history[:choices][baseline_strategy][i]

        # Simple propensity estimates
        target_propensity = target_strategy == :random ? 1 / n_arms : 0.5  # rough estimate
        baseline_propensity = baseline_strategy == :random ? 1 / n_arms : 0.5

        # IPW terms (simplified)
        if target_choice == real_choice
            ipw_correction += real_reward / target_propensity
        end
        if baseline_choice == real_choice
            ipw_correction -= real_reward / baseline_propensity
        end
    end
    ipw_correction /= n_epochs

    # Doubly robust = direct method + IPW correction
    doubly_robust = direct_method + ipw_correction * 0.1  # damped correction

    return Dict(
        :direct_method => direct_method,
        :doubly_robust => doubly_robust,
        :target_mean => mean(target_rewards),
        :baseline_mean => mean(baseline_rewards),
        :uplift_percent => (direct_method / mean(baseline_rewards)) * 100
    )
end

# Analyze uplift - no changes to existing code needed!
predictive_vs_random = analyze_doubly_robust_uplift(history, :predictive, :random)
thompson_vs_random = analyze_doubly_robust_uplift(history, :thompson, :random)

println("Predictive vs Random:")
println("  Direct Method: $(round(predictive_vs_random[:direct_method], digits=4))")
println("  Doubly Robust: $(round(predictive_vs_random[:doubly_robust], digits=4))")
println("  Uplift: $(round(predictive_vs_random[:uplift_percent], digits=2))%")

println("\nThompson vs Random:")
println("  Direct Method: $(round(thompson_vs_random[:direct_method], digits=4))")
println("  Doubly Robust: $(round(thompson_vs_random[:doubly_robust], digits=4))")
println("  Uplift: $(round(thompson_vs_random[:uplift_percent], digits=2))%")

function plot_moving_averages(history, n_epochs, ma_window=20)
    # Calculate moving average rewards
    ma_rewards_random = [mean(history[:rewards][:random][max(1, i - ma_window + 1):i]) for i in 1:n_epochs]
    ma_rewards_thompson = [mean(history[:rewards][:thompson][max(1, i - ma_window + 1):i]) for i in 1:n_epochs]
    ma_rewards_predictive = [mean(history[:rewards][:predictive][max(1, i - ma_window + 1):i]) for i in 1:n_epochs]

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
      ma_random = [mean(history[:rewards][:random][max(1, i - window + 1):i]) for i in 1:length(history[:rewards][:random])][1:k:end]
      ma_thompson = [mean(history[:rewards][:thompson][max(1, i - window + 1):i]) for i in 1:length(history[:rewards][:thompson])][1:k:end]
      ma_predictive = [mean(history[:rewards][:predictive][max(1, i - window + 1):i]) for i in 1:length(history[:rewards][:predictive])][1:k:end]

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

function plot_moving_averages(history, n_epochs, ma_window=20)
    # Calculate moving average rewards
    ma_rewards_random = [mean(history[:rewards][:random][max(1, i - ma_window + 1):i]) for i in 1:n_epochs]
    ma_rewards_thompson = [mean(history[:rewards][:thompson][max(1, i - ma_window + 1):i]) for i in 1:n_epochs]
    ma_rewards_predictive = [mean(history[:rewards][:predictive][max(1, i - ma_window + 1):i]) for i in 1:n_epochs]

    # Plot moving average
    plot(1:n_epochs, [ma_rewards_random, ma_rewards_thompson, ma_rewards_predictive],
        label=["Random" "Thompson" "Predictive"],
        title="Moving Average Reward",
        xlabel="Epoch", ylabel="Average Reward",
        lw=2)
end

# Plot moving averages
plot_moving_averages(history, n_epochs)