# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Advanced Examples/Integrating Neural Networks with Flux.jl/Integrating Neural Networks with Flux.jl.ipynb
# by notebooks_to_scripts.jl at 2025-03-27T06:11:19.957
#
# Source notebook: Integrating Neural Networks with Flux.jl.ipynb

using RxInfer, Flux, Random, Plots, LinearAlgebra, StableRNGs, ForwardDiff

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
plot(p1, p2, p3, size=(900, 250), layout=(1,3), margin=5Plots.mm)

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

function plot_coordinates(result)
    p1 = plot_coordinate(result, 1, title = "First coordinate")
    p2 = plot_coordinate(result, 2, title = "Second coordinate")
    p3 = plot_coordinate(result, 3, title = "Third coordinate")
    return plot(p1, p2, p3, size = (1000, 600), layout = (3, 1), legend=:bottomleft)
end

plot_coordinates(untrained_result)

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
    
    # Optional: plot training progress
    if @isdefined(Plots)
        p = Plots.plot(timestamps, free_energies, 
                       xlabel="Time (seconds)", 
                       ylabel="Free Energy",
                       title="Training Progress",
                       legend=false,
                       linewidth=2)
        display(p)
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

plot_coordinates(trained_result)

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

# Evaluate performance
performance_metrics = evaluate_model_performance(dataset.signal, dataset.noisy_signal, inferred_mean)

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
plot(p1, p2, p3, size=(900, 250), layout=(1,3), margin=5Plots.mm)