# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Advanced Examples/Integrating Neural Networks with Flux.jl/Integrating Neural Networks with Flux.jl.ipynb
# by notebooks_to_scripts.jl at 2025-03-27T06:11:19.957
#
# Source notebook: Integrating Neural Networks with Flux.jl.ipynb

using RxInfer, Flux, Random, Plots, LinearAlgebra, StableRNGs, ForwardDiff

# Also import Distributions for the posterior visualization
using Distributions

# Create output directory for saving plots
const OUTPUT_DIR = joinpath(@__DIR__, "output_images")
mkpath(OUTPUT_DIR)
println("Saving visualizations to: $OUTPUT_DIR")

# Lorenz system equations to be used to generate dataset
Base.@kwdef mutable struct Lorenz
    dt::Float64
    σ::Float64
    ρ::Float64
    β::Float64
    x::Float64
    y::Float64
    z::Float64
end

# Define the Lorenz dynamics
function step!(l::Lorenz)
    dx = l.σ * (l.y - l.x);         l.x += l.dt * dx
    dy = l.x * (l.ρ - l.z) - l.y;   l.y += l.dt * dy
    dz = l.x * l.y - l.β * l.z;     l.z += l.dt * dz
end

function create_dataset(rng, σ, ρ, β_nom; variance = 1f0, n_steps = 100, p_train = 0.8, p_test = 0.2)
    attractor = Lorenz(0.02, σ, ρ, β_nom/3.0, 1, 1, 1)
    signal       = [Float32[1.0, 1.0, 1.0]]
    noisy_signal = [last(signal) + randn(rng, Float32, 3) * variance]
    for i in 1:(n_steps - 1)
        step!(attractor)
        push!(signal, Float32[attractor.x, attractor.y, attractor.z])
        push!(noisy_signal, last(signal) + randn(rng, Float32, 3) * variance) 
    end

    return (
        parameters = (σ, ρ, β_nom),
        signal = signal, 
        noisy_signal = noisy_signal
    )
end

rng      = StableRNG(999) # dummy rng
variance = 2f0
dataset  = create_dataset(rng, 11, 23, 6; variance = variance, n_steps = 200);

# Extract first samples from datasets
sample_clean = dataset.signal
sample_noisy = dataset.noisy_signal

# Pre-allocate arrays for better performance
n_points = length(sample_clean)
gx, gy, gz = zeros(n_points), zeros(n_points), zeros(n_points)
rx, ry, rz = zeros(n_points), zeros(n_points), zeros(n_points)

# Extract coordinates
for i in 1:n_points
    # Noisy observations
    rx[i], ry[i], rz[i] = sample_noisy[i][1], sample_noisy[i][2], sample_noisy[i][3]
    # True state
    gx[i], gy[i], gz[i] = sample_clean[i][1], sample_clean[i][2], sample_clean[i][3]
end

# Create three projection plots
p1 = scatter(rx, ry, label="Noisy observations", alpha=0.7, markersize=2, title = "X-Y Projection")
plot!(p1, gx, gy, label="True state", linewidth=2)

p2 = scatter(rx, rz, label="Noisy observations", alpha=0.7, markersize=2, title = "X-Z Projection")
plot!(p2, gx, gz, label="True state", linewidth=2)

p3 = scatter(ry, rz, label="Noisy observations", alpha=0.7, markersize=2, title = "Y-Z Projection")
plot!(p3, gy, gz, label="True state", linewidth=2)

# Combine plots with improved layout
initial_proj_plot = plot(p1, p2, p3, size=(900, 250), layout=(1,3), margin=5Plots.mm)
savefig(initial_proj_plot, joinpath(OUTPUT_DIR, "01_initial_projections.png"))
println("Saved: 01_initial_projections.png")

function make_neural_network(rng = StableRNG(1234))
    model = Dense(3 => 3)

    # Initialize the weights and biases of the neural network
    flat, rebuild = Flux.destructure(model)

    # We use a fixed random seed for reproducibility
    rand!(rng, flat)

    # Return the neural network with fixed weights and biases
    return rebuild(flat)
end

@model function ssm(y, As, Q, B, R)
    
    x_prior_mean = ones(Float32, 3)
    x_prior_cov  = Matrix(Diagonal(ones(Float32, 3)))
    
    x[1] ~ MvNormal(mean = x_prior_mean, cov = x_prior_cov)
    y[1] ~ MvNormal(mean = B * x[1], cov = R)
    
    for i in 2:length(y)
        x[i] ~ MvNormal(mean = As[i - 1] * x[i - 1], cov = Q) 
        y[i] ~ MvNormal(mean = B * x[i], cov = R)
    end
end

Q = diageye(Float32, 3)
B = diageye(Float32, 3)
R = variance * diageye(Float32, 3)
;

function get_matrices_from_neural_network(data, neural_network)
    dd = hcat(data...)
    As = neural_network(dd)
    return map(c -> Matrix(Diagonal(c)), eachcol(As))
end

# Performance on an instance from the testset before training
untrained_neural_network = make_neural_network()
untrained_transition_matrices = get_matrices_from_neural_network(dataset.noisy_signal, untrained_neural_network)

untrained_result = infer(
    model = ssm(As = untrained_transition_matrices, Q = Q, B = B, R = R), 
    data  = (y = dataset.noisy_signal, ), 
    returnvars = (x = KeepLast(), )
)

# A helper function for plotting
function plot_coordinate(result, i; title = "")
    p = scatter(getindex.(dataset.noisy_signal, i), label="Observations", alpha=0.7, markersize=2, title = title)
    plot!(getindex.(dataset.signal, i), label="True states", linewidth=2)
    plot!(getindex.(mean.(result.posteriors[:x]), i), ribbon=sqrt.(getindex.(var.(result.posteriors[:x]), i)), label="Inferred states", linewidth=2)
    return p
end

function plot_coordinates(result; filename = nothing)
    p1 = plot_coordinate(result, 1, title = "First coordinate")
    p2 = plot_coordinate(result, 2, title = "Second coordinate")
    p3 = plot_coordinate(result, 3, title = "Third coordinate")
    plt = plot(p1, p2, p3, size = (1000, 600), layout = (3, 1), legend=:bottomleft)
    if !isnothing(filename)
        savefig(plt, joinpath(OUTPUT_DIR, filename))
        println("Saved: $filename")
    end
    return plt
end

untrained_plot = plot_coordinates(untrained_result, filename = "02_untrained_coordinates.png")
println("Saved: 02_untrained_coordinates.png")

# free energy objective to be optimized during training
function make_fe_tot_est(rebuild, data; Q = Q, B = B, R = R)
    function fe_tot_est(v)
        nn = rebuild(v)
        result = infer(
            model = ssm(As = get_matrices_from_neural_network(data, nn), Q = Q, B = B, R = R), 
            data  = (y = data, ), 
            returnvars = (x = KeepLast(), ),
            free_energy = true,
            session = nothing
        )
        return result.free_energy[end]
    end
end

function train!(neural_network, data; num_epochs = 500)
    # Initialize optimizer
    rule = Flux.Optimise.Adam()
    
    # Handle different Flux versions for optimizer setup
    state = try
        # Modern Flux versions
        Flux.Optimise.setup(rule, neural_network)
    catch
        # Older Flux versions
        Flux.Optimisers.setup(rule, neural_network)
    end

    x, rebuild = Flux.destructure(neural_network)
    fe_tot_est_ = make_fe_tot_est(rebuild, data)

    run_epochs!(rebuild, fe_tot_est_, state, neural_network; num_epochs = num_epochs)
end

function run_epochs!(rebuild::F, fe_tot_est::I, state::S, neural_network::N; num_epochs::Int = 100) where {F, I, S, N}
    print_each = max(1, num_epochs ÷ 10)
    start_time = time()
    
    # Metrics tracking
    free_energies = Float64[]
    timestamps = Float64[]
    
    # Initial free energy
    flat, _ = Flux.destructure(neural_network)
    initial_fe = fe_tot_est(flat)
    push!(free_energies, initial_fe)
    push!(timestamps, 0.0)
    
    println("Initial Free Energy: $initial_fe")
    println("Starting training for $num_epochs epochs...")
    
    for epoch in 1:num_epochs
        epoch_start = time()
        flat, _ = Flux.destructure(neural_network)
        
        # Compute gradients
        grads = ForwardDiff.gradient(fe_tot_est, flat)
        
        # Update parameters
        try
            Flux.update!(state, neural_network, rebuild(grads))
        catch e
            if epoch == 1
                @warn "Error in Flux.update!: $e. Trying alternative API..."
                Flux.Optimise.update!(state, neural_network, rebuild(grads))
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
            println("  Free Energy = $current_value ($(round(improvement_pct; digits=2))% improvement)")
            println("  Epoch time: $(round(epoch_time; digits=3))s")
            println("  ETA: $(round(remaining; digits=1))s")
        end
    end
    
    # Final metrics
    total_time = time() - start_time
    final_fe = fe_tot_est(flat)
    improvement = initial_fe - final_fe
    improvement_pct = 100 * abs(improvement) / abs(initial_fe)
    
    println("\nTraining summary:")
    println("  Finished in $(round(total_time; digits=2)) seconds")
    println("  Initial free energy: $initial_fe")
    println("  Final free energy: $final_fe")
    println("  Improvement: $improvement ($(round(improvement_pct; digits=2))%)")
    
    # Plot training progress
    if @isdefined(Plots)
        p = Plots.plot(timestamps, free_energies, 
                       xlabel="Time (seconds)", 
                       ylabel="Free Energy",
                       title="Training Progress",
                       legend=false,
                       linewidth=2)
        savefig(p, joinpath(OUTPUT_DIR, "03_training_progress.png"))
        println("Saved: 03_training_progress.png")
    end
    
    return (free_energies=free_energies, timestamps=timestamps)
end

trained_neural_network = make_neural_network()

train!(trained_neural_network, dataset.noisy_signal; num_epochs = 2000)

trained_transition_matrices = get_matrices_from_neural_network(dataset.noisy_signal, trained_neural_network)

trained_result = infer(
    model = ssm(As = trained_transition_matrices, Q = Q, B = B, R = R), 
    data  = (y = dataset.noisy_signal, ), 
    returnvars = (x = KeepLast(), )
)

trained_plot = plot_coordinates(trained_result, filename = "04_trained_coordinates.png")
println("Saved: 04_trained_coordinates.png")

ix, iy, iz = zeros(n_points), zeros(n_points), zeros(n_points)

inferred_mean = mean.(trained_result.posteriors[:x])

# Extract coordinates
for i in 1:n_points
    # Inferred mean
    ix[i], iy[i], iz[i] = inferred_mean[i][1], inferred_mean[i][2], inferred_mean[i][3]
end

# Add function to evaluate model performance with metrics
function evaluate_model_performance(true_signal, noisy_signal, inferred_mean)
    n = length(true_signal)
    
    # Calculate MSE for observations vs truth
    obs_mse = sum(sum((hcat(noisy_signal...) - hcat(true_signal...)).^2)) / (n * 3)
    
    # Calculate MSE for inferred vs truth
    inf_array = hcat(inferred_mean...)
    true_array = hcat(true_signal...)
    inf_mse = sum(sum((inf_array - true_array).^2)) / (n * 3)
    
    # Calculate improvement percentage
    improvement = (obs_mse - inf_mse) / obs_mse * 100
    
    println("\nPerformance Metrics:")
    println("  Observation MSE: $(round(obs_mse; digits=4))")
    println("  Inference MSE: $(round(inf_mse; digits=4))")
    println("  Improvement: $(round(improvement; digits=2))%")
    
    return (obs_mse=obs_mse, inf_mse=inf_mse, improvement=improvement)
end

# Function to visualize neural network parameters
function visualize_nn_parameters(nn, filename)
    flat, _ = Flux.destructure(nn)
    p = Plots.heatmap(reshape(flat, 1, :), 
                     title="Neural Network Parameters", 
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
    
    heatmaps = []
    for (i, idx) in enumerate(indices)
        push!(heatmaps, heatmap(matrices[idx], 
                              title="Matrix $idx", 
                              color=:viridis,
                              aspect_ratio=:equal,
                              xticks=1:3,
                              yticks=1:3))
    end
    
    p = Plots.plot(heatmaps..., size=(900, 700), layout=(4, 5))
    savefig(p, joinpath(OUTPUT_DIR, filename))
    println("Saved: $filename")
    return p
end

# Visualize untrained neural network parameters
println("\nVisualizing untrained neural network parameters...")
visualize_nn_parameters(untrained_neural_network, "06_untrained_nn_parameters.png")

# Visualize untrained transition matrices
println("Visualizing untrained transition matrices...")
visualize_transition_matrices(untrained_transition_matrices, "07_untrained_transition_matrices.png")

# Evaluate performance
performance_metrics = evaluate_model_performance(dataset.signal, dataset.noisy_signal, inferred_mean)

# Visualize trained neural network parameters
println("\nVisualizing trained neural network parameters...")
visualize_nn_parameters(trained_neural_network, "08_trained_nn_parameters.png")

# Visualize trained transition matrices
println("Visualizing trained transition matrices...")
visualize_transition_matrices(trained_transition_matrices, "09_trained_transition_matrices.png")

# Visualize parameter changes during training
println("\nVisualizing neural network parameter changes...")
untrained_params, _ = Flux.destructure(untrained_neural_network)
trained_params, _ = Flux.destructure(trained_neural_network)
param_changes = trained_params - untrained_params

p_changes = Plots.bar(param_changes, 
                    title="Neural Network Parameter Changes", 
                    ylabel="Change in Value", 
                    xlabel="Parameter Index",
                    legend=false,
                    color=:blue,
                    alpha=0.7,
                    size=(800, 400))
savefig(p_changes, joinpath(OUTPUT_DIR, "10_parameter_changes.png"))
println("Saved: 10_parameter_changes.png")

# Visualize uncertainty in the inferred states
println("\nVisualizing inference uncertainty...")
uncertainty = var.(trained_result.posteriors[:x])
uncertainty_matrix = hcat([sqrt.([u[1], u[2], u[3]]) for u in uncertainty]...)

p_uncert = Plots.plot(1:n_points, uncertainty_matrix', 
                    title="Inference Uncertainty", 
                    ylabel="Standard Deviation", 
                    xlabel="Time Step",
                    label=["x dimension" "y dimension" "z dimension"],
                    linewidth=2,
                    size=(800, 400))
savefig(p_uncert, joinpath(OUTPUT_DIR, "11_inference_uncertainty.png"))
println("Saved: 11_inference_uncertainty.png")

# Visualize model posterior distributions for selected time points
println("\nVisualizing posterior distributions...")
function visualize_posterior_distributions(result, timepoints; filename = nothing)
    n_points = min(length(timepoints), 9)  # Maximum 9 points to visualize
    selected_points = timepoints[1:n_points]
    
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
                     title="t=$t, dim=$dim", 
                     label="Posterior", 
                     fillalpha=0.3, 
                     fill=true, 
                     linewidth=2, 
                     xlabel="Value", 
                     ylabel="Density")
            
            # Add the true value marker
            true_value = dataset.signal[t][dim]
            vline!([true_value], label="True", linestyle=:dash, linewidth=2)
            
            # Add the observation marker
            obs_value = dataset.noisy_signal[t][dim]
            vline!([obs_value], label="Observed", linestyle=:dot, linewidth=2)
            
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
visualize_posterior_distributions(trained_result, selected_timepoints, filename="12_posterior_distributions.png")

# Visualize model state evolution with uncertainty
println("\nVisualizing state evolution with uncertainty...")
function visualize_state_evolution(result, filename)
    # Extract the means and standard deviations for each dimension
    means = mean.(result.posteriors[:x])
    stds = [sqrt.(diag(cov(post))) for post in result.posteriors[:x]]
    
    # Extract true values
    true_values = dataset.signal
    
    # Create arrays for plotting
    x_mean = getindex.(means, 1)
    y_mean = getindex.(means, 2)
    z_mean = getindex.(means, 3)
    
    x_std = getindex.(stds, 1)
    y_std = getindex.(stds, 2)
    z_std = getindex.(stds, 3)
    
    # Create evolving 3D plot with uncertainty
    p = plot3d(
        x_mean, y_mean, z_mean,
        title="State Evolution with Uncertainty",
        label="Inferred Mean",
        linewidth=2,
        legend=:topright,
        alpha=0.8,
        camera=(30, 30)
    )
    
    # Add uncertainty tube
    for i in 1:min(n_points, 100) # Limit points for clarity
        if i % 5 == 0 # Only plot some points to avoid clutter
            # Use average std as marker size
            marker_size = (x_std[i] + y_std[i] + z_std[i])/3 * 2
            scatter3d!(
                [x_mean[i]], [y_mean[i]], [z_mean[i]],
                markersize=marker_size,
                alpha=0.2,
                label=i==5 ? "Uncertainty" : false,
                color=:blue
            )
        end
    end
    
    # Add true trajectory
    plot3d!(
        getindex.(true_values, 1),
        getindex.(true_values, 2),
        getindex.(true_values, 3),
        label="True State",
        linewidth=2,
        color=:red
    )
    
    # Save and display
    savefig(p, joinpath(OUTPUT_DIR, filename))
    return p
end

visualize_state_evolution(trained_result, "13_state_evolution_3d.png")

# Visualize comparison between prior and posterior model
println("\nVisualizing prior vs posterior model performance...")
function compare_prior_posterior(untrained_result, trained_result, filename)
    # Get means from both models
    prior_means = mean.(untrained_result.posteriors[:x])
    posterior_means = mean.(trained_result.posteriors[:x])
    
    # Get true values
    true_values = dataset.signal
    
    # Calculate errors
    prior_errors = [norm(prior_means[i] - true_values[i]) for i in 1:n_points]
    posterior_errors = [norm(posterior_means[i] - true_values[i]) for i in 1:n_points]
    
    # Plot comparison
    p = plot(
        1:n_points, 
        [prior_errors posterior_errors], 
        title="Prior vs Posterior Model Error",
        xlabel="Time Step",
        ylabel="Error (L2 Norm)",
        label=["Prior Model" "Posterior Model"],
        linewidth=2,
        alpha=0.8
    )
    
    # Add improvement percentage annotation
    mean_prior_error = mean(prior_errors)
    mean_posterior_error = mean(posterior_errors)
    improvement = (mean_prior_error - mean_posterior_error) / mean_prior_error * 100
    
    annotate!(
        n_points/2, 
        maximum(prior_errors) * 0.9,
        text("Improvement: $(round(improvement; digits=2))%", :center, 10)
    )
    
    # Save and display
    savefig(p, joinpath(OUTPUT_DIR, filename))
    println("Saved: $filename")
    return p
end

compare_prior_posterior(untrained_result, trained_result, "14_prior_vs_posterior.png")

# Create three projection plots
p1 = scatter(rx, ry, label="Noisy observations", alpha=0.7, markersize=2, title = "X-Y Projection")
plot!(p1, gx, gy, label="True state", linewidth=2)
plot!(p1, ix, iy, label="Inferred Mean", linewidth=2)

p2 = scatter(rx, rz, label="Noisy observations", alpha=0.7, markersize=2, title = "X-Z Projection")
plot!(p2, gx, gz, label="True state", linewidth=2)
plot!(p2, ix, iz, label="Inferred Mean", linewidth=2)

p3 = scatter(ry, rz, label="Noisy observations", alpha=0.7, markersize=2, title = "Y-Z Projection")
plot!(p3, gy, gz, label="True state", linewidth=2)
plot!(p3, iy, iz, label="Inferred Mean", linewidth=2)

# Combine plots with improved layout
final_proj_plot = plot(p1, p2, p3, size=(900, 250), layout=(1,3), margin=5Plots.mm)
savefig(final_proj_plot, joinpath(OUTPUT_DIR, "05_final_projections.png"))
println("Saved: 05_final_projections.png")

# Save metrics to a summary file
open(joinpath(OUTPUT_DIR, "performance_summary.txt"), "w") do f
    println(f, "=== Performance Metrics ===")
    println(f, "Observation MSE: $(round(performance_metrics.obs_mse; digits=4))")
    println(f, "Inference MSE: $(round(performance_metrics.inf_mse; digits=4))")
    println(f, "Improvement: $(round(performance_metrics.improvement; digits=2))%")
    
    # Add more detailed metrics
    println(f, "\n=== Neural Network Parameters ===")
    println(f, "Number of parameters: $(length(trained_params))")
    println(f, "Mean parameter value: $(round(mean(trained_params); digits=4))")
    println(f, "Parameter value range: [$(round(minimum(trained_params); digits=4)), $(round(maximum(trained_params); digits=4))]")
    println(f, "Parameter change summary:")
    println(f, "  Mean absolute change: $(round(mean(abs.(param_changes)); digits=4))")
    println(f, "  Max absolute change: $(round(maximum(abs.(param_changes)); digits=4))")
    
    println(f, "\n=== Uncertainty Analysis ===")
    println(f, "Mean uncertainty (std dev):")
    println(f, "  x dimension: $(round(mean(uncertainty_matrix[1,:]); digits=4))")
    println(f, "  y dimension: $(round(mean(uncertainty_matrix[2,:]); digits=4))")
    println(f, "  z dimension: $(round(mean(uncertainty_matrix[3,:]); digits=4))")
    println(f, "Max uncertainty (std dev):")
    println(f, "  x dimension: $(round(maximum(uncertainty_matrix[1,:]); digits=4))")
    println(f, "  y dimension: $(round(maximum(uncertainty_matrix[2,:]); digits=4))")
    println(f, "  z dimension: $(round(maximum(uncertainty_matrix[3,:]); digits=4))")
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
    
    overview = Plots.plot(overview_plots..., layout=(rows, cols), size=(1200, 200*rows), title="Visualization Overview")
    savefig(overview, joinpath(OUTPUT_DIR, "15_visualization_overview.png"))
    println("Saved: 15_visualization_overview.png")
catch e
    println("Could not create visualization overview: $e")
end

println("\nAll visualizations saved to: $OUTPUT_DIR")