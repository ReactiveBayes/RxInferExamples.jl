# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Advanced Examples/Integrating Neural Networks with Flux.jl/Integrating Neural Networks with Flux.jl.ipynb
# by notebooks_to_scripts.jl at 2025-03-27T06:11:19.957
#
# Source notebook: Integrating Neural Networks with Flux.jl.ipynb

using RxInfer, Flux, Random, Plots, LinearAlgebra, StableRNGs, ForwardDiff

# Also import Distributions for the posterior visualization
using Distributions

# Create output directory for saving plots and animations
const OUTPUT_DIR = joinpath(@__DIR__, "Balance", "output_images")
const ANIMATION_DIR = joinpath(OUTPUT_DIR, "animations")
const BIOMECHANICAL_DIR = joinpath(OUTPUT_DIR, "biomechanical")
mkpath(OUTPUT_DIR)
mkpath(ANIMATION_DIR)
mkpath(BIOMECHANICAL_DIR)
println("Saving visualizations to: $OUTPUT_DIR")
println("Saving animations to: $ANIMATION_DIR")
println("Saving biomechanical animations to: $BIOMECHANICAL_DIR")

# This example demonstrates a vestibular-cochlear balance model where:
# - The true underlying dynamics represent postural sway (modeled as a Lorenz attractor)
# - Noisy measurements come from proprioceptive sensory systems with measurement noise
# - A neural network learns to filter noisy proprioceptive inputs
# - The inferred states represent conscious perception of postural location

# Postural sway dynamics modeled by Lorenz system equations
Base.@kwdef mutable struct PosturalSway
    dt::Float64     # Time step
    σ::Float64      # Anterior-posterior oscillation parameter
    ρ::Float64      # Medio-lateral oscillation parameter  
    β::Float64      # Vertical stability parameter
    x::Float64      # Anterior-posterior position
    y::Float64      # Medio-lateral position
    z::Float64      # Vertical position/balance
end

# Define the postural sway dynamics (Lorenz-based)
function step!(ps::PosturalSway)
    dx = ps.σ * (ps.y - ps.x);         ps.x += ps.dt * dx  # Anterior-posterior change
    dy = ps.x * (ps.ρ - ps.z) - ps.y;  ps.y += ps.dt * dy  # Medio-lateral change
    dz = ps.x * ps.y - ps.β * ps.z;    ps.z += ps.dt * dz  # Vertical balance change
end

function create_proprioceptive_dataset(rng, σ, ρ, β_nom; 
                                      sensory_noise = 1f0, 
                                      n_steps = 100, 
                                      p_train = 0.8, 
                                      p_test = 0.2)
    # Initialize postural sway simulation
    postural_dynamics = PosturalSway(0.02, σ, ρ, β_nom/3.0, 1, 1, 1)
    
    # Initialize datasets
    true_posture     = [Float32[1.0, 1.0, 1.0]]  # True postural state
    proprioceptive_signal = [last(true_posture) + randn(rng, Float32, 3) * sensory_noise]  # Noisy proprioceptive signal
    
    # Simulate postural dynamics
    for i in 1:(n_steps - 1)
        step!(postural_dynamics)
        push!(true_posture, Float32[postural_dynamics.x, postural_dynamics.y, postural_dynamics.z])
        push!(proprioceptive_signal, last(true_posture) + randn(rng, Float32, 3) * sensory_noise) 
    end

    return (
        parameters = (σ, ρ, β_nom),
        true_posture = true_posture, 
        proprioceptive_signal = proprioceptive_signal
    )
end

# Simulation parameters
rng = StableRNG(999) # Seed for reproducibility
proprioceptive_noise = 2f0  # Level of proprioceptive sensory noise
dataset = create_proprioceptive_dataset(rng, 11, 23, 6; sensory_noise = proprioceptive_noise, n_steps = 200);

# Extract samples from datasets
true_postural_states = dataset.true_posture
proprioceptive_measurements = dataset.proprioceptive_signal

# Pre-allocate arrays for better performance
n_points = length(true_postural_states)
gx, gy, gz = zeros(n_points), zeros(n_points), zeros(n_points)  # True postural state
rx, ry, rz = zeros(n_points), zeros(n_points), zeros(n_points)  # Proprioceptive measurements

# Extract coordinates
for i in 1:n_points
    # Proprioceptive signals (noisy observations)
    rx[i], ry[i], rz[i] = proprioceptive_measurements[i][1], proprioceptive_measurements[i][2], proprioceptive_measurements[i][3]
    # True postural states
    gx[i], gy[i], gz[i] = true_postural_states[i][1], true_postural_states[i][2], true_postural_states[i][3]
end

# Create three projection plots of postural state space
p1 = scatter(rx, ry, label="Proprioceptive signals", alpha=0.7, markersize=2, title = "Anterior-Posterior vs Medio-Lateral")
plot!(p1, gx, gy, label="True postural state", linewidth=2)

p2 = scatter(rx, rz, label="Proprioceptive signals", alpha=0.7, markersize=2, title = "Anterior-Posterior vs Vertical")
plot!(p2, gx, gz, label="True postural state", linewidth=2)

p3 = scatter(ry, rz, label="Proprioceptive signals", alpha=0.7, markersize=2, title = "Medio-Lateral vs Vertical")
plot!(p3, gy, gz, label="True postural state", linewidth=2)

# Combine plots with improved layout
initial_proj_plot = plot(p1, p2, p3, size=(900, 250), layout=(1,3), margin=5Plots.mm)
savefig(initial_proj_plot, joinpath(OUTPUT_DIR, "01_initial_postural_projections.png"))
println("Saved: 01_initial_postural_projections.png")

function make_perception_network(rng = StableRNG(1234))
    # Neural network that transforms proprioceptive signals into perceptual estimates
    model = Dense(3 => 3)

    # Initialize the weights and biases of the neural network
    flat, rebuild = Flux.destructure(model)

    # We use a fixed random seed for reproducibility
    rand!(rng, flat)

    # Return the neural network with fixed weights and biases
    return rebuild(flat)
end

@model function vestibular_perception_model(y, As, Q, B, R)
    # This model represents how the brain might process proprioceptive signals
    # y: proprioceptive measurements
    # x: perceptual states (conscious awareness of posture)
    # As: dynamically learned transition matrices
    # Q: process noise in perceptual system
    # B: observation matrix
    # R: proprioceptive measurement noise
    
    # Prior beliefs about initial postural state
    x_prior_mean = ones(Float32, 3)
    x_prior_cov  = Matrix(Diagonal(ones(Float32, 3)))
    
    # Initial perceptual state and measurement
    x[1] ~ MvNormal(mean = x_prior_mean, cov = x_prior_cov)
    y[1] ~ MvNormal(mean = B * x[1], cov = R)
    
    # Perceptual dynamics and measurements over time
    for i in 2:length(y)
        # Perceptual state evolution (conscious awareness)
        x[i] ~ MvNormal(mean = As[i - 1] * x[i - 1], cov = Q) 
        # Proprioceptive measurement given perceptual state
        y[i] ~ MvNormal(mean = B * x[i], cov = R)
    end
end

# Define covariance matrices
Q = diageye(Float32, 3)  # Process noise for perceptual dynamics
B = diageye(Float32, 3)  # Observation matrix
R = proprioceptive_noise * diageye(Float32, 3)  # Proprioceptive measurement noise
;

function get_transition_matrices_from_network(data, network)
    # Transform proprioceptive data through neural network to get transition matrices
    dd = hcat(data...)
    As = network(dd)
    return map(c -> Matrix(Diagonal(c)), eachcol(As))
end

# Initial perception model (before learning)
untrained_perception_network = make_perception_network()
untrained_transition_matrices = get_transition_matrices_from_network(dataset.proprioceptive_signal, untrained_perception_network)

untrained_perception = infer(
    model = vestibular_perception_model(As = untrained_transition_matrices, Q = Q, B = B, R = R), 
    data  = (y = dataset.proprioceptive_signal, ), 
    returnvars = (x = KeepLast(), )
)

# Helper function for plotting postural dimensions
function plot_postural_dimension(result, i; title = "")
    p = scatter(getindex.(dataset.proprioceptive_signal, i), label="Proprioceptive signals", alpha=0.7, markersize=2, title = title)
    plot!(getindex.(dataset.true_posture, i), label="True postural states", linewidth=2)
    plot!(getindex.(mean.(result.posteriors[:x]), i), ribbon=sqrt.(getindex.(var.(result.posteriors[:x]), i)), label="Perceived states", linewidth=2)
    return p
end

function plot_postural_dimensions(result; filename = nothing)
    dimension_names = ["Anterior-Posterior", "Medio-Lateral", "Vertical"]
    p1 = plot_postural_dimension(result, 1, title = dimension_names[1])
    p2 = plot_postural_dimension(result, 2, title = dimension_names[2])
    p3 = plot_postural_dimension(result, 3, title = dimension_names[3])
    plt = plot(p1, p2, p3, size = (1000, 600), layout = (3, 1), legend=:bottomleft)
    if !isnothing(filename)
        savefig(plt, joinpath(OUTPUT_DIR, filename))
        println("Saved: $filename")
    end
    return plt
end

untrained_perception_plot = plot_postural_dimensions(untrained_perception, filename = "02_untrained_perception.png")
println("Saved: 02_untrained_perception.png")

# Free energy objective to be optimized during perceptual learning
function make_perceptual_learning_objective(rebuild, data; Q = Q, B = B, R = R)
    function fe_tot_est(v)
        nn = rebuild(v)
        result = infer(
            model = vestibular_perception_model(As = get_transition_matrices_from_network(data, nn), Q = Q, B = B, R = R), 
            data  = (y = data, ), 
            returnvars = (x = KeepLast(), ),
            free_energy = true,
            session = nothing
        )
        return result.free_energy[end]
    end
end

function train_perception_network!(network, data; num_epochs = 500)
    # Initialize optimizer
    rule = Flux.Optimise.Adam()
    
    # Handle different Flux versions for optimizer setup
    state = try
        # Modern Flux versions
        Flux.Optimise.setup(rule, network)
    catch
        # Older Flux versions
        Flux.Optimisers.setup(rule, network)
    end

    x, rebuild = Flux.destructure(network)
    fe_tot_est_ = make_perceptual_learning_objective(rebuild, data)

    run_perceptual_learning!(rebuild, fe_tot_est_, state, network; num_epochs = num_epochs)
end

function run_perceptual_learning!(rebuild::F, fe_tot_est::I, state::S, network::N; num_epochs::Int = 100) where {F, I, S, N}
    print_each = max(1, num_epochs ÷ 10)
    start_time = time()
    
    # Metrics tracking
    free_energies = Float64[]
    timestamps = Float64[]
    
    # Initial free energy
    flat, _ = Flux.destructure(network)
    initial_fe = fe_tot_est(flat)
    push!(free_energies, initial_fe)
    push!(timestamps, 0.0)
    
    println("Initial Perceptual Uncertainty (Free Energy): $initial_fe")
    println("Starting perceptual learning for $num_epochs epochs...")
    
    for epoch in 1:num_epochs
        epoch_start = time()
        flat, _ = Flux.destructure(network)
        
        # Compute gradients
        grads = ForwardDiff.gradient(fe_tot_est, flat)
        
        # Update parameters
        try
            Flux.update!(state, network, rebuild(grads))
        catch e
            if epoch == 1
                @warn "Error in Flux.update!: $e. Trying alternative API..."
                Flux.Optimise.update!(state, network, rebuild(grads))
            else
                rethrow(e)
            end
        end
        
        # Logging
        if epoch % print_each == 0 || epoch == num_epochs
            current_value = fe_tot_est(flat)
            push!(free_energies, current_value)
            push!(timestamps, time() - start_time)
            
            elapsed = time() - start_time
            remaining = elapsed / epoch * (num_epochs - epoch)
            epoch_time = time() - epoch_start
            
            improvement = initial_fe - current_value
            improvement_pct = 100 * abs(improvement) / abs(initial_fe)
            
            println("Epoch $epoch/$num_epochs:")
            println("  Perceptual Uncertainty = $current_value ($(round(improvement_pct; digits=2))% improvement)")
            println("  Epoch time: $(round(epoch_time; digits=3))s")
            println("  ETA: $(round(remaining; digits=1))s")
        end
    end
    
    # Final metrics
    total_time = time() - start_time
    final_fe = fe_tot_est(flat)
    improvement = initial_fe - final_fe
    improvement_pct = 100 * abs(improvement) / abs(initial_fe)
    
    println("\nPerceptual learning summary:")
    println("  Finished in $(round(total_time; digits=2)) seconds")
    println("  Initial perceptual uncertainty: $initial_fe")
    println("  Final perceptual uncertainty: $final_fe")
    println("  Perceptual precision improvement: $improvement ($(round(improvement_pct; digits=2))%)")
    
    # Plot training progress
    if @isdefined(Plots)
        p = Plots.plot(timestamps, free_energies, 
                       xlabel="Time (seconds)", 
                       ylabel="Perceptual Uncertainty (Free Energy)",
                       title="Perceptual Learning Progress",
                       legend=false,
                       linewidth=2)
        savefig(p, joinpath(OUTPUT_DIR, "03_perceptual_learning_progress.png"))
        println("Saved: 03_perceptual_learning_progress.png")
    end
    
    return (free_energies=free_energies, timestamps=timestamps)
end

# Initialize and train the perception network
trained_perception_network = make_perception_network()

# Train the network to better perceive posture from noisy proprioceptive signals
train_perception_network!(trained_perception_network, dataset.proprioceptive_signal; num_epochs = 2000)

# Get transition matrices from trained network
trained_transition_matrices = get_transition_matrices_from_network(dataset.proprioceptive_signal, trained_perception_network)

# Infer postural states using trained perception model
trained_perception = infer(
    model = vestibular_perception_model(As = trained_transition_matrices, Q = Q, B = B, R = R), 
    data  = (y = dataset.proprioceptive_signal, ), 
    returnvars = (x = KeepLast(), )
)

# Plot results with trained perception model
trained_perception_plot = plot_postural_dimensions(trained_perception, filename = "04_trained_perception.png")
println("Saved: 04_trained_perception.png")

ix, iy, iz = zeros(n_points), zeros(n_points), zeros(n_points)  # Perceived states

perceived_states = mean.(trained_perception.posteriors[:x])

# Extract coordinates of perceived postural states
for i in 1:n_points
    # Perceived postural states
    ix[i], iy[i], iz[i] = perceived_states[i][1], perceived_states[i][2], perceived_states[i][3]
end

# Function to evaluate perceptual performance with metrics
function evaluate_perceptual_performance(true_posture, proprioceptive_signals, perceived_states)
    n = length(true_posture)
    
    # Calculate MSE for proprioceptive signals vs true posture
    prop_mse = sum(sum((hcat(proprioceptive_signals...) - hcat(true_posture...)).^2)) / (n * 3)
    
    # Calculate MSE for perceived vs true posture
    perc_array = hcat(perceived_states...)
    true_array = hcat(true_posture...)
    perc_mse = sum(sum((perc_array - true_array).^2)) / (n * 3)
    
    # Calculate improvement percentage (perceptual precision)
    perceptual_precision = (prop_mse - perc_mse) / prop_mse * 100
    
    println("\nPerceptual Performance Metrics:")
    println("  Proprioceptive Signal MSE: $(round(prop_mse; digits=4))")
    println("  Perceptual MSE: $(round(perc_mse; digits=4))")
    println("  Perceptual Precision: $(round(perceptual_precision; digits=2))%")
    
    return (prop_mse=prop_mse, perc_mse=perc_mse, perceptual_precision=perceptual_precision)
end

# Function to visualize neural network parameters
function visualize_network_parameters(nn, filename)
    flat, _ = Flux.destructure(nn)
    p = Plots.heatmap(reshape(flat, 1, :), 
                     title="Vestibular-Cochlear Neural Network Parameters", 
                     ylabel="Parameters", 
                     xlabel="Index",
                     color=:viridis,
                     size=(800, 200))
    savefig(p, joinpath(OUTPUT_DIR, filename))
    println("Saved: $filename")
    return p
end

# Function to visualize transition matrices
function visualize_transition_matrices(matrices, filename)
    n = length(matrices)
    sample_size = min(20, n)  # Sample a reasonable number to visualize
    indices = round.(Int, range(1, n, length=sample_size))
    
    dimension_labels = ["AP", "ML", "V"]  # Anterior-Posterior, Medio-Lateral, Vertical
    
    heatmaps = []
    for (i, idx) in enumerate(indices)
        push!(heatmaps, heatmap(matrices[idx], 
                              title="Matrix $idx", 
                              color=:viridis,
                              aspect_ratio=:equal,
                              xticks=(1:3, dimension_labels),
                              yticks=(1:3, dimension_labels)))
    end
    
    p = Plots.plot(heatmaps..., size=(900, 700), layout=(4, 5))
    savefig(p, joinpath(OUTPUT_DIR, filename))
    println("Saved: $filename")
    return p
end

# Visualize untrained neural network parameters
println("\nVisualizing untrained vestibular-cochlear network parameters...")
visualize_network_parameters(untrained_perception_network, "06_untrained_network_parameters.png")

# Visualize untrained transition matrices
println("Visualizing untrained perceptual transition matrices...")
visualize_transition_matrices(untrained_transition_matrices, "07_untrained_transition_matrices.png")

# Evaluate perceptual performance
perceptual_metrics = evaluate_perceptual_performance(dataset.true_posture, dataset.proprioceptive_signal, perceived_states)

# Visualize trained neural network parameters
println("\nVisualizing trained vestibular-cochlear network parameters...")
visualize_network_parameters(trained_perception_network, "08_trained_network_parameters.png")

# Visualize trained transition matrices
println("Visualizing trained perceptual transition matrices...")
visualize_transition_matrices(trained_transition_matrices, "09_trained_transition_matrices.png")

# Visualize parameter changes during training
println("\nVisualizing neural network parameter changes...")
untrained_params, _ = Flux.destructure(untrained_perception_network)
trained_params, _ = Flux.destructure(trained_perception_network)
param_changes = trained_params - untrained_params

p_changes = Plots.bar(param_changes, 
                    title="Vestibular-Cochlear Neural Network Parameter Changes", 
                    ylabel="Change in Value", 
                    xlabel="Parameter Index",
                    legend=false,
                    color=:blue,
                    alpha=0.7,
                    size=(800, 400))
savefig(p_changes, joinpath(OUTPUT_DIR, "10_parameter_changes.png"))
println("Saved: 10_parameter_changes.png")

# Visualize uncertainty in the perceived states (perceptual precision)
println("\nVisualizing perceptual precision...")
perceptual_uncertainty = var.(trained_perception.posteriors[:x])
uncertainty_matrix = hcat([sqrt.([u[1], u[2], u[3]]) for u in perceptual_uncertainty]...)

dimension_labels = ["Anterior-Posterior", "Medio-Lateral", "Vertical"]
p_uncert = Plots.plot(1:n_points, uncertainty_matrix', 
                    title="Perceptual Precision", 
                    ylabel="Perceptual Uncertainty (Std Dev)", 
                    xlabel="Time Step",
                    label=dimension_labels,
                    linewidth=2,
                    size=(800, 400))
savefig(p_uncert, joinpath(OUTPUT_DIR, "11_perceptual_precision.png"))
println("Saved: 11_perceptual_precision.png")

# Visualize perceptual posterior distributions for selected time points
println("\nVisualizing perceptual distributions...")
function visualize_perceptual_distributions(result, timepoints; filename = nothing)
    n_points = min(length(timepoints), 9)  # Maximum 9 points to visualize
    selected_points = timepoints[1:n_points]
    dimension_labels = ["Anterior-Posterior", "Medio-Lateral", "Vertical"]
    
    plots = []
    for t in selected_points
        posterior = result.posteriors[:x][t]
        μ = mean(posterior)
        Σ = cov(posterior)
        
        # Create distribution plots for each dimension
        for dim in 1:3
            dim_μ = μ[dim]
            dim_σ = sqrt(Σ[dim, dim])
            
            # Generate points for the distribution curve
            x_range = range(dim_μ - 3*dim_σ, dim_μ + 3*dim_σ, length=100)
            pdf_values = [pdf(Normal(dim_μ, dim_σ), x) for x in x_range]
            
            # Plot the distribution
            p = plot(x_range, pdf_values, 
                     title="t=$t, dim=$(dimension_labels[dim])", 
                     label="Perceptual Distribution", 
                     fillalpha=0.3, 
                     fill=true, 
                     linewidth=2, 
                     xlabel="Postural Position", 
                     ylabel="Perceptual Density")
            
            # Add the true value marker
            true_value = dataset.true_posture[t][dim]
            vline!([true_value], label="True Posture", linestyle=:dash, linewidth=2)
            
            # Add the proprioceptive signal marker
            prop_value = dataset.proprioceptive_signal[t][dim]
            vline!([prop_value], label="Proprioceptive Signal", linestyle=:dot, linewidth=2)
            
            push!(plots, p)
        end
    end
    
    # Create a grid of plots
    n_dim_plots = length(plots)
    grid_size = ceil(Int, sqrt(n_dim_plots))
    layout_dims = (grid_size, grid_size)
    
    posterior_plot = plot(plots..., layout=layout_dims, size=(800, 800), legend=:topright)
    
    if !isnothing(filename)
        savefig(posterior_plot, joinpath(OUTPUT_DIR, filename))
        println("Saved: $filename")
    end
    
    return posterior_plot
end

# Select representative timepoints for visualization
# Beginning, middle and end points
selected_timepoints = [1, n_points÷4, n_points÷2, 3*n_points÷4, n_points]
visualize_perceptual_distributions(trained_perception, selected_timepoints, filename="12_perceptual_distributions.png")

# Visualize postural state evolution with perceptual uncertainty
println("\nVisualizing postural state evolution with perceptual uncertainty...")
function visualize_postural_evolution(result, filename)
    # Extract the means and standard deviations for each dimension
    means = mean.(result.posteriors[:x])
    stds = [sqrt.(diag(cov(post))) for post in result.posteriors[:x]]
    
    # Extract true values
    true_values = dataset.true_posture
    
    # Create arrays for plotting
    x_mean = getindex.(means, 1)  # Anterior-Posterior
    y_mean = getindex.(means, 2)  # Medio-Lateral
    z_mean = getindex.(means, 3)  # Vertical
    
    x_std = getindex.(stds, 1)
    y_std = getindex.(stds, 2)
    z_std = getindex.(stds, 3)
    
    # Create evolving 3D plot with uncertainty
    p = plot3d(
        x_mean, y_mean, z_mean,
        title="Postural State Evolution with Perceptual Uncertainty",
        xlabel="Anterior-Posterior",
        ylabel="Medio-Lateral",
        zlabel="Vertical",
        label="Perceived Posture",
        linewidth=2,
        legend=:topright,
        alpha=0.8,
        camera=(30, 30)
    )
    
    # Add uncertainty tube (perceptual precision)
    for i in 1:min(n_points, 100) # Limit points for clarity
        if i % 5 == 0 # Only plot some points to avoid clutter
            # Use average std as marker size
            marker_size = (x_std[i] + y_std[i] + z_std[i])/3 * 2
            scatter3d!(
                [x_mean[i]], [y_mean[i]], [z_mean[i]],
                markersize=marker_size,
                alpha=0.2,
                label=i==5 ? "Perceptual Uncertainty" : false,
                color=:blue
            )
        end
    end
    
    # Add true postural trajectory
    plot3d!(
        getindex.(true_values, 1),
        getindex.(true_values, 2),
        getindex.(true_values, 3),
        label="True Postural State",
        linewidth=2,
        color=:red
    )
    
    # Save and display
    savefig(p, joinpath(OUTPUT_DIR, filename))
    return p
end

visualize_postural_evolution(trained_perception, "13_postural_evolution_3d.png")

# Visualize comparison between pre-learning and post-learning perception
println("\nVisualizing perceptual learning effects...")
function compare_perceptual_learning(untrained_result, trained_result, filename)
    # Get means from both models
    pre_learning_perception = mean.(untrained_result.posteriors[:x])
    post_learning_perception = mean.(trained_result.posteriors[:x])
    
    # Get true values
    true_posture = dataset.true_posture
    
    # Calculate errors
    pre_learning_errors = [norm(pre_learning_perception[i] - true_posture[i]) for i in 1:n_points]
    post_learning_errors = [norm(post_learning_perception[i] - true_posture[i]) for i in 1:n_points]
    
    # Plot comparison
    p = plot(
        1:n_points, 
        [pre_learning_errors post_learning_errors], 
        title="Perceptual Learning Effects",
        xlabel="Time Step",
        ylabel="Perceptual Error (L2 Norm)",
        label=["Pre-Learning Perception" "Post-Learning Perception"],
        linewidth=2,
        alpha=0.8
    )
    
    # Add improvement percentage annotation
    mean_pre_error = mean(pre_learning_errors)
    mean_post_error = mean(post_learning_errors)
    improvement = (mean_pre_error - mean_post_error) / mean_pre_error * 100
    
    annotate!(
        n_points/2, 
        maximum(pre_learning_errors) * 0.9,
        text("Perceptual Learning: $(round(improvement; digits=2))% improvement", :center, 10)
    )
    
    # Save and display
    savefig(p, joinpath(OUTPUT_DIR, filename))
    println("Saved: $filename")
    return p
end

compare_perceptual_learning(untrained_perception, trained_perception, "14_perceptual_learning.png")

# Create three projection plots with perceived states
p1 = scatter(rx, ry, label="Proprioceptive signals", alpha=0.7, markersize=2, title = "Anterior-Posterior vs Medio-Lateral")
plot!(p1, gx, gy, label="True postural state", linewidth=2)
plot!(p1, ix, iy, label="Perceived posture", linewidth=2)

p2 = scatter(rx, rz, label="Proprioceptive signals", alpha=0.7, markersize=2, title = "Anterior-Posterior vs Vertical")
plot!(p2, gx, gz, label="True postural state", linewidth=2)
plot!(p2, ix, iz, label="Perceived posture", linewidth=2)

p3 = scatter(ry, rz, label="Proprioceptive signals", alpha=0.7, markersize=2, title = "Medio-Lateral vs Vertical")
plot!(p3, gy, gz, label="True postural state", linewidth=2)
plot!(p3, iy, iz, label="Perceived posture", linewidth=2)

# Combine plots with improved layout
final_proj_plot = plot(p1, p2, p3, size=(900, 250), layout=(1,3), margin=5Plots.mm)
savefig(final_proj_plot, joinpath(OUTPUT_DIR, "05_final_postural_projections.png"))
println("Saved: 05_final_postural_projections.png")

# Save metrics to a summary file
open(joinpath(OUTPUT_DIR, "vestibular_perception_summary.txt"), "w") do f
    println(f, "=== Vestibular-Cochlear Balance Performance Metrics ===")
    println(f, "Proprioceptive Signal MSE: $(round(perceptual_metrics.prop_mse; digits=4))")
    println(f, "Perceptual MSE: $(round(perceptual_metrics.perc_mse; digits=4))")
    println(f, "Perceptual Precision: $(round(perceptual_metrics.perceptual_precision; digits=2))%")
    
    # Add more detailed metrics
    println(f, "\n=== Neural Network Parameters ===")
    println(f, "Number of parameters: $(length(trained_params))")
    println(f, "Mean parameter value: $(round(mean(trained_params); digits=4))")
    println(f, "Parameter value range: [$(round(minimum(trained_params); digits=4)), $(round(maximum(trained_params); digits=4))]")
    println(f, "Parameter change summary:")
    println(f, "  Mean absolute change: $(round(mean(abs.(param_changes)); digits=4))")
    println(f, "  Max absolute change: $(round(maximum(abs.(param_changes)); digits=4))")
    
    println(f, "\n=== Perceptual Precision Analysis ===")
    println(f, "Mean perceptual uncertainty (std dev):")
    println(f, "  Anterior-Posterior: $(round(mean(uncertainty_matrix[1,:]); digits=4))")
    println(f, "  Medio-Lateral: $(round(mean(uncertainty_matrix[2,:]); digits=4))")
    println(f, "  Vertical: $(round(mean(uncertainty_matrix[3,:]); digits=4))")
    println(f, "Max perceptual uncertainty (std dev):")
    println(f, "  Anterior-Posterior: $(round(maximum(uncertainty_matrix[1,:]); digits=4))")
    println(f, "  Medio-Lateral: $(round(maximum(uncertainty_matrix[2,:]); digits=4))")
    println(f, "  Vertical: $(round(maximum(uncertainty_matrix[3,:]); digits=4))")
    
    # Add interpretation of vestibular-cochlear balance
    println(f, "\n=== Vestibular-Cochlear Balance Interpretation ===")
    println(f, "The model demonstrates how the brain might integrate noisy proprioceptive")
    println(f, "signals to form a coherent perception of postural state. The neural network")
    println(f, "learns to filter noisy proprioceptive inputs, resulting in improved")
    println(f, "perceptual precision ($(round(perceptual_metrics.perceptual_precision; digits=2))% improvement).")
    println(f, "")
    println(f, "Dimension-specific findings:")
    println(f, "1. Anterior-Posterior perception precision: $(round(100*(1-mean(uncertainty_matrix[1,:])/maximum(uncertainty_matrix[1,:])); digits=2))%")
    println(f, "2. Medio-Lateral perception precision: $(round(100*(1-mean(uncertainty_matrix[2,:])/maximum(uncertainty_matrix[2,:])); digits=2))%")
    println(f, "3. Vertical perception precision: $(round(100*(1-mean(uncertainty_matrix[3,:])/maximum(uncertainty_matrix[3,:])); digits=2))%")
    println(f, "")
    println(f, "This suggests that the vestibular-cochlear system achieves different levels")
    println(f, "of perceptual precision across postural dimensions, potentially reflecting")
    println(f, "the relative importance of each dimension for maintaining balance.")
end

# Create a visualization showing all saved images for a quick overview
println("\nCreating visualization overview...")
try
    image_files = filter(f -> endswith(f, ".png"), readdir(OUTPUT_DIR))
    overview_plots = []
    
    for file in image_files
        img = Plots.plot(title=file)
        try
            # Try to load and display the image
            img = Plots.plot(Plots.plot!(img, joinpath(OUTPUT_DIR, file)), title=file, framestyle=:box)
        catch
            # If loading fails, just show the title
            img = Plots.plot(title="Failed to load: $file", annotations=(0.5, 0.5, file))
        end
        push!(overview_plots, img)
    end
    
    n_imgs = length(overview_plots)
    cols = min(3, n_imgs)
    rows = ceil(Int, n_imgs / cols)
    
    overview = Plots.plot(overview_plots..., layout=(rows, cols), size=(1200, 200*rows), title="Vestibular-Cochlear Balance Visualization Overview")
    savefig(overview, joinpath(OUTPUT_DIR, "15_visualization_overview.png"))
    println("Saved: 15_visualization_overview.png")
catch e
    println("Could not create visualization overview: $e")
end

println("\nAll visualizations saved to: $OUTPUT_DIR")

# Function to create animated 3D postural trajectory
function animate_postural_trajectory(result, true_posture, proprioceptive_signals; filename="postural_trajectory.gif", fps=15)
    println("\nCreating 3D postural trajectory animation...")
    
    # Extract means and variances
    means = mean.(result.posteriors[:x])
    stds = [sqrt.(diag(cov(post))) for post in result.posteriors[:x]]
    
    # Extract coordinates
    x_mean = getindex.(means, 1)  # Anterior-Posterior
    y_mean = getindex.(means, 2)  # Medio-Lateral
    z_mean = getindex.(means, 3)  # Vertical
    
    # True postural coordinates
    true_x = getindex.(true_posture, 1)
    true_y = getindex.(true_posture, 2)
    true_z = getindex.(true_posture, 3)
    
    # Proprioceptive signal coordinates
    prop_x = getindex.(proprioceptive_signals, 1)
    prop_y = getindex.(proprioceptive_signals, 2)
    prop_z = getindex.(proprioceptive_signals, 3)
    
    # Determine axis limits
    x_lim = (min(minimum(x_mean), minimum(true_x), minimum(prop_x)) * 1.1,
             max(maximum(x_mean), maximum(true_x), maximum(prop_x)) * 1.1)
    y_lim = (min(minimum(y_mean), minimum(true_y), minimum(prop_y)) * 1.1,
             max(maximum(y_mean), maximum(true_y), maximum(prop_y)) * 1.1)
    z_lim = (min(minimum(z_mean), minimum(true_z), minimum(prop_z)) * 1.1,
             max(maximum(z_mean), maximum(true_z), maximum(prop_z)) * 1.1)
    
    # Create animation
    anim = @animate for t in 1:length(means)
        # Create 3D plot
        p = plot3d(
            # Plot trajectory up to current time
            x_mean[1:t], y_mean[1:t], z_mean[1:t],
            title="Postural State Evolution Over Time (t=$t)",
            xlabel="Anterior-Posterior", ylabel="Medio-Lateral", zlabel="Vertical",
            label="Perceived Posture",
            linewidth=2,
            color=:blue,
            xlim=x_lim, ylim=y_lim, zlim=z_lim,
            camera=(30 + 50*sin(t/length(means)*π), 30 + 20*cos(t/length(means)*π)),
            legend=:topright
        )
        
        # Add true trajectory
        plot3d!(
            true_x[1:t], true_y[1:t], true_z[1:t],
            label="True Postural State",
            linewidth=2,
            color=:red
        )
        
        # Add proprioceptive signals with transparency
        scatter3d!(
            prop_x[1:t], prop_y[1:t], prop_z[1:t],
            label="Proprioceptive Signals",
            markersize=2,
            markerstrokewidth=0,
            color=:gray,
            alpha=0.3
        )
        
        # Add current point with uncertainty (perceptual precision)
        if t > 0
            # Use average std as marker size
            marker_size = sum([stds[t][i] for i in 1:3])/3 * 3
            scatter3d!(
                [x_mean[t]], [y_mean[t]], [z_mean[t]],
                markersize=marker_size,
                alpha=0.5,
                color=:blue,
                label="Current Perceptual State"
            )
        end
    end
    
    # Save animation
    gif(anim, joinpath(ANIMATION_DIR, filename), fps=fps)
    println("Saved animation: $filename")
end

# Function to animate postural time series data
function animate_postural_time_series(result, true_posture, proprioceptive_signals; filename="postural_time_series.gif", fps=15)
    println("\nCreating postural time series animation...")
    
    # Extract means and standard deviations
    means = mean.(result.posteriors[:x])
    stds = [sqrt.(diag(cov(post))) for post in result.posteriors[:x]]
    
    # Pre-allocate arrays for better performance
    n_points = length(means)
    dimension_labels = ["Anterior-Posterior", "Medio-Lateral", "Vertical"]
    
    # Create animation
    anim = @animate for t in 1:n_points
        p = plot(layout=(3,1), size=(800, 600), legend=:topright)
        
        for dim in 1:3
            # Extract coordinates for this dimension
            dim_means = [mean[dim] for mean in means[1:t]]
            dim_stds = [std[dim] for std in stds[1:t]]
            
            # True and proprioceptive signals
            true_vals = [s[dim] for s in true_posture[1:t]]
            prop_vals = [s[dim] for s in proprioceptive_signals[1:t]]
            
            # Plot dimension
            subplot = plot!(p, 1:t, dim_means, 
                          ribbon=dim_stds,
                          subplot=dim,
                          title="$(dimension_labels[dim]) (t=$t)",
                          ylabel="Position",
                          label="Perceived Posture",
                          linewidth=2,
                          fillalpha=0.3)
            
            # Add true signal
            plot!(p, 1:t, true_vals,
                 subplot=dim,
                 label="True Postural State",
                 linewidth=2,
                 color=:red)
            
            # Add proprioceptive signals
            scatter!(p, 1:t, prop_vals,
                   subplot=dim,
                   label="Proprioceptive Signals",
                   markersize=2,
                   markerstrokewidth=0,
                   color=:gray,
                   alpha=0.7)
        end
        
        plot!(p, xlabel="Time Step", bottom_margin=5Plots.mm)
    end
    
    # Save animation
    gif(anim, joinpath(ANIMATION_DIR, filename), fps=fps)
    println("Saved animation: $filename")
end

# Function to animate transition matrices evolution
function animate_transition_matrices(matrices; filename="perceptual_transitions.gif", fps=10)
    println("\nCreating perceptual transition matrices animation...")
    
    n = length(matrices)
    # Use a subset if there are too many matrices
    step_size = max(1, n ÷ 100)
    selected_indices = 1:step_size:n
    
    dimension_labels = ["AP", "ML", "V"]
    
    anim = @animate for i in selected_indices
        matrix = matrices[i]
        
        p = heatmap(matrix, 
                  title="Perceptual Transition Matrix (t=$i)",
                  color=:viridis,
                  aspect_ratio=:equal,
                  clim=(0, max(1.0, maximum(matrix))),
                  xticks=(1:3, dimension_labels),
                  yticks=(1:3, dimension_labels),
                  size=(500, 500))
                  
        annotate!([(j, k, text(round(matrix[j,k], digits=2), 8, :white)) 
                  for j in 1:3 for k in 1:3])
    end
    
    # Save animation
    gif(anim, joinpath(ANIMATION_DIR, filename), fps=fps)
    println("Saved animation: $filename")
end

# Function to animate neural network parameter changes
function animate_parameter_changes(untrained_nn, trained_nn, training_steps=100; filename="perceptual_learning_evolution.gif", fps=10)
    println("\nCreating perceptual learning evolution animation...")
    
    # Get initial and final parameters
    initial_params, _ = Flux.destructure(untrained_nn)
    final_params, _ = Flux.destructure(trained_nn)
    
    # Interpolate between initial and final parameters
    anim = @animate for t in 0:training_steps
        # Linear interpolation between initial and final parameters
        alpha = t / training_steps
        current_params = (1 - alpha) * initial_params + alpha * final_params
        
        # Plot current parameters
        p = bar(current_params,
              title="Vestibular-Cochlear Neural Network ($(round(Int, alpha*100))% trained)",
              xlabel="Parameter Index",
              ylabel="Value",
              legend=false,
              color=:blue,
              alpha=0.7,
              size=(800, 400))
              
        # Add reference lines for initial and final values
        hline!([0], color=:black, linestyle=:dash, label="Zero")
        
        # Add progress information
        annotate!([(length(current_params)÷2, maximum(current_params)*0.9, 
                   text("Perceptual Learning: $(round(Int, alpha*100))%", 10, :center))])
    end
    
    # Save animation
    gif(anim, joinpath(ANIMATION_DIR, filename), fps=fps)
    println("Saved animation: $filename")
end

# Function to create animated perceptual distribution evolution
function animate_perceptual_distribution(result, dim=1; filename="perceptual_distribution_evolution.gif", fps=10)
    dimension_labels = ["Anterior-Posterior", "Medio-Lateral", "Vertical"]
    println("\nCreating perceptual distribution evolution animation for $(dimension_labels[dim])...")
    
    posteriors = result.posteriors[:x]
    n_points = length(posteriors)
    
    # Determine global limits for consistent animation
    all_means = [mean(post)[dim] for post in posteriors]
    all_stds = [sqrt(cov(post)[dim,dim]) for post in posteriors]
    
    x_min = minimum(all_means) - 4 * maximum(all_stds)
    x_max = maximum(all_means) + 4 * maximum(all_stds)
    y_max = maximum([pdf(Normal(μ, σ), μ) for (μ, σ) in zip(all_means, all_stds)]) * 1.2
    
    anim = @animate for t in 1:n_points
        posterior = posteriors[t]
        μ = mean(posterior)[dim]
        σ = sqrt(cov(posterior)[dim,dim])
        
        # Generate points for the distribution curve
        x_range = range(x_min, x_max, length=200)
        pdf_values = [pdf(Normal(μ, σ), x) for x in x_range]
        
        # Plot the distribution
        p = plot(x_range, pdf_values, 
                title="Perceptual Distribution for $(dimension_labels[dim]) (t=$t)", 
                label="Current Perception",
                fill=true, 
                fillalpha=0.3,
                linewidth=2, 
                xlabel="Postural Position", 
                ylabel="Perceptual Density",
                xlim=(x_min, x_max),
                ylim=(0, y_max),
                size=(700, 400))
        
        # Add the true value marker if available
        if @isdefined(dataset) && t <= length(dataset.true_posture)
            true_value = dataset.true_posture[t][dim]
            vline!([true_value], label="True Posture", linestyle=:dash, linewidth=2, color=:red)
            
            # Add the proprioceptive signal marker
            prop_value = dataset.proprioceptive_signal[t][dim]
            vline!([prop_value], label="Proprioceptive Signal", linestyle=:dot, linewidth=2, color=:gray)
        end
        
        # Show mean and std dev as text
        annotate!([
            (x_min + (x_max - x_min) * 0.1, y_max * 0.9, 
             text("Perceived position = $(round(μ, digits=2))", 10, :left)),
            (x_min + (x_max - x_min) * 0.1, y_max * 0.8, 
             text("Perceptual uncertainty = $(round(σ, digits=2))", 10, :left))
        ])
    end
    
    # Save animation
    gif(anim, joinpath(ANIMATION_DIR, filename), fps=fps)
    println("Saved animation: $filename")
end

# Create animations after all visualizations
println("\nGenerating animations...")

# 3D postural trajectory animation
animate_postural_trajectory(trained_perception, dataset.true_posture, dataset.proprioceptive_signal, 
                     filename="01_3d_postural_trajectory.gif", fps=20)

# Postural time series animation
animate_postural_time_series(trained_perception, dataset.true_posture, dataset.proprioceptive_signal,
                   filename="02_postural_time_series.gif", fps=20)

# Perceptual transition matrices animation
animate_transition_matrices(trained_transition_matrices,
                           filename="03_perceptual_transition_matrices.gif", fps=10)

# Perceptual learning evolution animation
animate_parameter_changes(untrained_perception_network, trained_perception_network, 100,
                         filename="04_perceptual_learning_evolution.gif", fps=15)

# Perceptual distribution evolution animations (one for each dimension)
animate_perceptual_distribution(trained_perception, 1, filename="05_anterior_posterior_perception.gif")
animate_perceptual_distribution(trained_perception, 2, filename="06_medio_lateral_perception.gif")
animate_perceptual_distribution(trained_perception, 3, filename="07_vertical_perception.gif")

println("\nAll vestibular-cochlear balance visualizations saved to: $OUTPUT_DIR")
println("Vestibular-cochlear balance animations saved to: $ANIMATION_DIR")

# Print a brief summary of findings
println("\n================ Vestibular-Cochlear Balance Study Summary ================")
println("This study modeled how the brain processes proprioceptive signals to maintain")
println("balance during postural sway.")
println("")
println("Key findings:")
println("1. Perceptual Precision: $(round(perceptual_metrics.perceptual_precision; digits=2))% improvement over raw proprioceptive signals")
println("2. Dimension-specific perception:")
println("   - Anterior-Posterior: $(round(100*(1-mean(uncertainty_matrix[1,:])/maximum(uncertainty_matrix[1,:])); digits=2))% precision")
println("   - Medio-Lateral: $(round(100*(1-mean(uncertainty_matrix[2,:])/maximum(uncertainty_matrix[2,:])); digits=2))% precision")
println("   - Vertical: $(round(100*(1-mean(uncertainty_matrix[3,:])/maximum(uncertainty_matrix[3,:])); digits=2))% precision")
println("3. The neural network learned to filter noisy proprioceptive signals to create")
println("   a more stable and precise perception of postural state.")
println("")
println("Full details available in: vestibular_perception_summary.txt")
println("========================================================================")

# Function to generate biomechanical stick figure animation of postural sway
function animate_biomechanical_postural_sway(true_posture, perceived_posture, proprioceptive_signals; 
                                           filename="biomechanical_postural_sway.gif", fps=15)
    println("\nCreating biomechanical stick figure animation...")
    
    # Extract the data
    true_states = hcat(true_posture...)
    perceived_states = hcat(perceived_posture...)
    prop_signals = hcat(proprioceptive_signals...)
    
    # Scale factors for biomechanical visualization (adjust as needed)
    scale_factor = 10.0    # Overall size scaling
    sway_amplify = 0.5     # Amplify sway movements
    height = 70.0          # Height of stick figure (base value)
    
    # Number of frames for the animation
    n_frames = size(true_states, 2)
    
    # Create animation
    anim = @animate for t in 1:n_frames
        # Extract current positions
        true_pos = true_states[:, t]
        perceived_pos = perceived_states[:, t]
        prop_pos = prop_signals[:, t]
        
        # Create plot with larger canvas for stick figures
        p = plot(
            size=(800, 600),
            aspect_ratio=:equal,
            xlim=(-10, 30),
            ylim=(0, 80),
            legend=:topleft,
            title="Biomechanical Postural Sway Animation (Frame $t)"
        )
        
        # Draw stick figures for each data source with appropriate offsets
        
        # Ground reference line
        hline!([0], color=:black, linewidth=2, label=false)
        
        # Draw true posture stick figure (middle)
        draw_stick_figure!(p, true_pos, 10, color=:red, label="True Posture", scale=scale_factor, 
                          sway_factor=sway_amplify, height=height)
        
        # Draw proprioceptive signal stick figure (left)
        draw_stick_figure!(p, prop_pos, 0, color=:gray, alpha=0.7, label="Proprioceptive Signals", 
                          scale=scale_factor, sway_factor=sway_amplify, height=height)
        
        # Draw perceived posture stick figure (right)
        draw_stick_figure!(p, perceived_pos, 20, color=:blue, label="Perceived Posture", 
                          scale=scale_factor, sway_factor=sway_amplify, height=height)
        
        # Add informative text
        annotate!(0, 75, text("Frame: $t/$n_frames", 10, :left))
        
        # Add balance indicators
        balance_true = norm(true_pos)
        balance_perceived = norm(perceived_pos)
        balance_prop = norm(prop_pos)
        
        annotate!(0, 70, text("Balance Quality (lower is better):", 8, :left))
        annotate!(0, 67, text("True: $(round(balance_true; digits=2))", 8, :left, :red))
        annotate!(0, 64, text("Perceived: $(round(balance_perceived; digits=2))", 8, :left, :blue))
        annotate!(0, 61, text("Proprioceptive: $(round(balance_prop; digits=2))", 8, :left, :gray))
    end
    
    # Save animation
    gif(anim, joinpath(BIOMECHANICAL_DIR, filename), fps=fps)
    println("Saved biomechanical animation: $filename")
    return anim
end

# Helper function to draw a stick figure with postural dynamics
function draw_stick_figure!(p, position, x_offset=0; 
                           color=:blue, alpha=1.0, label=nothing, 
                           scale=10.0, sway_factor=1.0, height=70.0)
    # Unpack the position (Lorenz coordinates)
    x, y, z = position
    
    # Convert Lorenz coordinates to postural sway movements
    anterior_posterior = x * sway_factor    # Anterior-posterior sway (x coordinate)
    medio_lateral = y * sway_factor        # Medio-lateral sway (leaning left/right)
    vertical = height - z * sway_factor/2  # Vertical position (slight bobbing)
    
    # Basic stick figure points (center-positioned)
    # Head
    head_x = x_offset + anterior_posterior
    head_y = vertical + 15
    head_radius = 2.5
    
    # Torso
    torso_top_x = head_x
    torso_top_y = vertical + 10
    torso_bottom_x = head_x + medio_lateral  # Shift based on medio-lateral sway
    torso_bottom_y = vertical
    
    # Arms - adjusted for sway
    shoulder_width = 5.0
    left_shoulder_x = torso_top_x - shoulder_width/2
    right_shoulder_x = torso_top_x + shoulder_width/2
    shoulders_y = torso_top_y
    
    arm_length = 8.0
    left_hand_x = left_shoulder_x - arm_length/2 + medio_lateral/2
    right_hand_x = right_shoulder_x + arm_length/2 + medio_lateral/2
    hands_y = shoulders_y - arm_length/2
    
    # Legs - adjusted for balance posture
    hip_width = 3.0
    left_hip_x = torso_bottom_x - hip_width/2
    right_hip_x = torso_bottom_x + hip_width/2
    hips_y = torso_bottom_y
    
    leg_length = 10.0
    left_foot_x = left_hip_x - medio_lateral
    right_foot_x = right_hip_x - medio_lateral  
    feet_y = 0  # Ground level
    
    # Draw stick figure components
    
    # Head (circle)
    plot!(p, circle(head_x, head_y, head_radius), seriestype=:shape, 
          color=color, alpha=alpha, label=label)
    
    # Torso (line)
    plot!(p, [torso_top_x, torso_bottom_x], [torso_top_y, torso_bottom_y], 
          color=color, alpha=alpha, label=false, linewidth=2)
    
    # Arms (lines)
    plot!(p, [left_shoulder_x, left_hand_x], [shoulders_y, hands_y], 
          color=color, alpha=alpha, label=false, linewidth=2)
    plot!(p, [right_shoulder_x, right_hand_x], [shoulders_y, hands_y], 
          color=color, alpha=alpha, label=false, linewidth=2)
    
    # Legs (lines)
    plot!(p, [left_hip_x, left_foot_x], [hips_y, feet_y], 
          color=color, alpha=alpha, label=false, linewidth=2)
    plot!(p, [right_hip_x, right_foot_x], [hips_y, feet_y], 
          color=color, alpha=alpha, label=false, linewidth=2)
    
    # Center of mass indicator (small point)
    scatter!(p, [torso_bottom_x], [vertical/2], 
            color=color, alpha=alpha, label=false, markersize=4)
    
    # Direction arrow showing movement tendency
    arrow_length = 3.0
    # Show anterior-posterior movement direction
    arrow_x = head_x
    arrow_y = head_y + head_radius + 1
    quiver!(p, [arrow_x], [arrow_y], 
           quiver=([anterior_posterior/2], [0]), 
           color=color, alpha=alpha, label=false)
    
    return p
end

# Helper function for drawing circles
function circle(x, y, r)
    θ = LinRange(0, 2π, 50)
    return x .+ r*sin.(θ), y .+ r*cos.(θ)
end

# Create biomechanical animation
animate_biomechanical_postural_sway(
    dataset.true_posture, 
    perceived_states, 
    dataset.proprioceptive_signal,
    filename="biomechanical_postural_sway.gif", 
    fps=10
)