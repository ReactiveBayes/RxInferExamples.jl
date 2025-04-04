#!/usr/bin/env julia
# Parameter Optimisation with Optim.jl
# This script demonstrates parameter optimization using RxInfer.jl with Optim.jl
# Originally from: Parameter Optimisation with Optim.jl.ipynb

using RxInfer, StableRNGs, LinearAlgebra, Plots, Optim, Dates, StatsPlots, LaTeXStrings

# Load visualization utilities
include(joinpath(@__DIR__, "visualization_utils.jl"))

# Check if the required packages are installed
for pkg in ["RxInfer", "StableRNGs", "LinearAlgebra", "Plots", "Optim", "Dates", "StatsPlots", "LaTeXStrings", "Measures"]
    try
        @eval using $(Symbol(pkg))
    catch
        @error "Package $pkg is missing. Installing..."
        import Pkg
        Pkg.add(pkg)
        @eval using $(Symbol(pkg))
    end
end

# Create timestamped output directory
output_dir, log_file = create_output_dir()
log_message("Output will be saved to: $output_dir", log_file)

log_message("\n========== PART 1: Univariate State Space Model ==========\n", log_file)

@model function univariate_state_space_model(y, x_prior, c, v)
    
    x0 ~ Normal(mean = mean(x_prior), variance = var(x_prior))
    x_prev = x0

    for i in eachindex(y)
        x[i] := x_prev + c
        y[i] ~ Normal(mean = x[i], variance = v)
        x_prev = x[i]
    end
end

log_message("Generating synthetic data for univariate state space model...", log_file)
rng    = StableRNG(42)
v      = 1.0
n      = 250
c_real = -5.0
signal = c_real .+ collect(1:n)
data   = map(x -> rand(rng, NormalMeanVariance(x, v)), signal)
log_message("✓ Data generated: $n points with true c = $c_real and v = $v", log_file)

# Plot univariate data
plot_univariate_data(signal, data, v, output_dir=output_dir)
log_message("✓ Saved univariate data plot to $(joinpath(output_dir, "univariate_data.png"))", log_file)

log_message("\nDefining optimization function...", log_file)
# params[1] is C
# params[2] is μ1
function f(params)
    x_prior = NormalMeanVariance(params[2], 100.0)
    result = infer(
        model = univariate_state_space_model(
            x_prior = x_prior, 
            c       = params[1], 
            v       = v
        ), 
        data  = (y = data,), 
        free_energy = true
    )
    return result.free_energy[end]
end

# Function to collect parameters during optimization
univariate_params_history = Vector{Vector{Float64}}()
function f_with_history(params)
    push!(univariate_params_history, copy(params))
    return f(params)
end

log_message("\nOptimizing parameters using Gradient Descent...", log_file)
res = optimize(f_with_history, ones(2), GradientDescent(), 
    Optim.Options(g_tol = 1e-3, iterations = 100, store_trace = true, show_trace = true, show_every = 10))

# Save optimization trace
if hasfield(typeof(res), :trace)
    trace_values = [state.value for state in res.trace]
    save_optimization_trace(trace_values, "Univariate Model Optimization", output_dir=output_dir)
    log_message("✓ Saved optimization trace to $(joinpath(output_dir, "optimization_trace.png"))", log_file)
    
    # Create animation of the optimization process if we have enough iterations
    if length(trace_values) > 1
        true_params = [c_real, 0.0]  # Approximate true initial state mean
        anim = animate_optimization(trace_values, univariate_params_history, true_params, output_dir=output_dir)
        if !isnothing(anim)
            log_message("✓ Saved optimization animation to $(joinpath(output_dir, "optimization_animation.gif"))", log_file)
        else
            log_message("⚠ Not enough data points for optimization animation", log_file)
        end
        
        # Create free energy evolution animation
        fe_anim = animate_free_energy_evolution(trace_values, univariate_params_history, output_dir=output_dir)
        if !isnothing(fe_anim)
            log_message("✓ Saved free energy evolution animation to $(joinpath(output_dir, "free_energy_evolution.gif"))", log_file)
        end
        
        # Create stacked free energy visualization
        create_stacked_free_energy_visualization(trace_values, univariate_params_history, true_params, output_dir=output_dir)
        log_message("✓ Saved stacked free energy visualization to $(joinpath(output_dir, "free_energy_stacked.png"))", log_file)
        
        # Visualize the free energy landscape within a reasonable range around the solution
        # Define parameter ranges for visualization
        param1_range = [c_real - 2.0, c_real + 2.0]
        param2_range = [-5.0, 5.0]
        
        # Create free energy landscape visualizations
        visualize_free_energy_landscape(f, [param1_range, param2_range], true_params, output_dir=output_dir)
        log_message("✓ Saved free energy landscape visualizations", log_file)
        
        # Create free energy descent animation showing optimization on the landscape
        fe_descent = animate_free_energy_descent(f, [param1_range, param2_range], 
                                                univariate_params_history, true_params, 
                                                output_dir=output_dir)
        if !isnothing(fe_descent)
            log_message("✓ Saved free energy descent animation to $(joinpath(output_dir, "free_energy_descent.gif"))", log_file)
        end
        
        # Plot parameter correlation if we have enough data
        if length(univariate_params_history) > 1
            plot_parameter_correlation(univariate_params_history, true_params, output_dir=output_dir)
            log_message("✓ Saved parameter trajectory to $(joinpath(output_dir, "parameter_trajectory.png"))", log_file)
        end
    else
        log_message("⚠ Not enough iterations for optimization animation", log_file)
    end
end

log_message("\nResults for univariate model:", log_file)
log_message("Real value vs Optimized", log_file)
log_message("Real:      $([ c_real, 0.0 ])", log_file)
log_message("Optimized: $(res.minimizer)", log_file)
log_message("Converged: $(Optim.converged(res))", log_file)

# Save results to file
open(joinpath(output_dir, "univariate_results.txt"), "w") do io
    println(io, "Real value vs Optimized")
    println(io, "Real:      $([ c_real, 0.0 ])")
    println(io, "Optimized: $(res.minimizer)")
    println(io, "Converged: $(Optim.converged(res))")
end

# Run inference with optimized parameters
log_message("\nRunning inference with optimized parameters...", log_file)
x_prior = NormalMeanVariance(res.minimizer[2], 100.0)
univariate_result = infer(
    model = univariate_state_space_model(
        x_prior = x_prior, 
        c       = res.minimizer[1], 
        v       = v
    ), 
    data  = (y = data,)
)

# Plot results with inferred states
x_posterior = univariate_result.posteriors[:x]
plot_univariate_results(signal, data, x_posterior, v, output_dir=output_dir)
log_message("✓ Saved univariate inference results to $(joinpath(output_dir, "univariate_results.png"))", log_file)

log_message("\n\n========== PART 2: Multivariate State Space Model ==========\n", log_file)

@model function multivariate_state_space_model(y, θ, x0, Q, P)
    
    x_prior ~ MvNormal(mean = mean(x0), cov = cov(x0))
    x_prev = x_prior
    
    A = [ cos(θ) -sin(θ); sin(θ) cos(θ) ]
    
    for i in eachindex(y)
        x[i] ~ MvNormal(mean = A * x_prev, covariance = Q)
        y[i] ~ MvNormal(mean = x[i], covariance = P)
        x_prev = x[i]
    end
    
end

log_message("Generating data for multivariate model...", log_file)
# Generate data
function generate_rotate_ssm_data()
    rng = StableRNG(1234)

    θ = π / 8
    A = [ cos(θ) -sin(θ); sin(θ) cos(θ) ]
    Q = Matrix(Diagonal(1.0 * ones(2)))
    P = Matrix(Diagonal(1.0 * ones(2)))

    n = 300

    x_prev = [ 10.0, -10.0 ]

    x = Vector{Vector{Float64}}(undef, n)
    y = Vector{Vector{Float64}}(undef, n)

    for i in 1:n
        
        x[i] = rand(rng, MvNormal(A * x_prev, Q))
        y[i] = rand(rng, MvNormal(x[i], Q))
        
        x_prev = x[i]
    end

    return θ, A, Q, P, n, x, y
end

θ, A, Q, P, n, x, y = generate_rotate_ssm_data()
log_message("✓ Generated $n data points with true θ = $(θ) ($(θ*180/π)°)", log_file)

# Plot multivariate data with phase space
time_plot, phase_plot = plot_multivariate_data(x, Q, output_dir=output_dir)
log_message("✓ Saved multivariate data plots:", log_file)
log_message("  - Time series: $(joinpath(output_dir, "multivariate_data.png"))", log_file)
log_message("  - Phase space: $(joinpath(output_dir, "multivariate_phase.png"))", log_file)

log_message("\nDefining optimization function for multivariate model...", log_file)
function f(params)
    x0 = MvNormalMeanCovariance(
        [ params[2], params[3] ], 
        Matrix(Diagonal(0.01 * ones(2)))
    )
    result = infer(
        model = multivariate_state_space_model(
            θ = params[1], 
            x0 = x0, 
            Q = Q, 
            P = P
        ), 
        data  = (y = y,), 
        free_energy = true
    )
    return result.free_energy[end]
end

# Function to collect parameters during optimization
multivariate_params_history = Vector{Vector{Float64}}()
function f_with_history(params)
    push!(multivariate_params_history, copy(params))
    return f(params)
end

log_message("\nOptimizing parameters using LBFGS...", log_file)
res = optimize(f_with_history, zeros(3), LBFGS(), 
    Optim.Options(f_tol = 1e-14, g_tol = 1e-12, show_trace = true, show_every = 10))

# Save optimization trace for multivariate model
if hasfield(typeof(res), :trace)
    trace_values = [state.value for state in res.trace]
    save_optimization_trace(trace_values, "Multivariate Model Optimization", output_dir=output_dir)
    log_message("✓ Saved multivariate optimization trace", log_file)
    
    # Create animation of the optimization process if we have enough iterations
    if length(trace_values) > 1
        true_params = [θ, 10.0, -10.0]
        anim = animate_optimization(trace_values, multivariate_params_history, true_params, output_dir=output_dir)
        if !isnothing(anim)
            log_message("✓ Saved multivariate optimization animation", log_file)
        else
            log_message("⚠ Not enough data points for multivariate optimization animation", log_file)
        end
        
        # Create free energy evolution animation
        fe_anim = animate_free_energy_evolution(trace_values, multivariate_params_history, output_dir=output_dir)
        if !isnothing(fe_anim)
            log_message("✓ Saved free energy evolution animation", log_file)
        end
        
        # Create stacked free energy visualization
        create_stacked_free_energy_visualization(trace_values, multivariate_params_history, true_params, output_dir=output_dir)
        log_message("✓ Saved stacked free energy visualization", log_file)
        
        # Visualize free energy landscape for first two parameters (θ and first component of x0)
        param1_range = [θ - 1.0, θ + 1.0]  # Range around true θ
        param2_range = [5.0, 15.0]        # Range for first component of x0
        
        # Create a function for visualizing just the first two parameters
        function f_vis(params)
            # Use the third parameter from the optimized result
            vis_params = [params[1], params[2], res.minimizer[3]]
            return f(vis_params)
        end
        
        # Only use first two parameters for visualization
        true_params_vis = [true_params[1], true_params[2]]
        
        # Create free energy landscape visualizations
        visualize_free_energy_landscape(f_vis, [param1_range, param2_range], true_params_vis, output_dir=output_dir)
        log_message("✓ Saved multivariate free energy landscape visualizations", log_file)
        
        # Create parameter trajectory for first two parameters
        multivariate_params_history_2d = [[p[1], p[2]] for p in multivariate_params_history]
        
        # Create free energy descent animation
        fe_descent = animate_free_energy_descent(f_vis, [param1_range, param2_range], 
                                               multivariate_params_history_2d, true_params_vis, 
                                               output_dir=output_dir, n_points=30)
        if !isnothing(fe_descent)
            log_message("✓ Saved multivariate free energy descent animation", log_file)
        end
    else
        log_message("⚠ Not enough iterations for optimization animation", log_file)
    end
end

log_message("\nResults for multivariate model:", log_file)
log_message("Real value vs Optimized", log_file)
log_message("θ (true) = $(θ) ($(θ*180/π)°)", log_file)

# Normalize the optimized angle to [-π, π] range
normalized_theta = mod(res.minimizer[1], 2π)
if normalized_theta > π
    normalized_theta -= 2π
end

log_message("θ (optimized) = $(normalized_theta) ($(normalized_theta*180/π)°)", log_file)
log_message("θ (raw optimized) = $(res.minimizer[1])", log_file)
log_message("sinθ = ($(sin(θ)), $(sin(res.minimizer[1])))", log_file)
log_message("cosθ = ($(cos(θ)), $(cos(res.minimizer[1])))", log_file)
log_message("Initial x₀ (true) = [10.0, -10.0]", log_file)
log_message("Initial x₀ (optimized) = [$(res.minimizer[2]), $(res.minimizer[3])]", log_file)
log_message("Converged: $(Optim.converged(res))", log_file)

# Save results to file
open(joinpath(output_dir, "multivariate_results.txt"), "w") do io
    println(io, "Real value vs Optimized")
    println(io, "θ (true) = $(θ) ($(θ*180/π)°)")
    println(io, "θ (optimized) = $(normalized_theta) ($(normalized_theta*180/π)°)")
    println(io, "θ (raw optimized) = $(res.minimizer[1])")
    println(io, "sinθ = ($(sin(θ)), $(sin(res.minimizer[1])))")
    println(io, "cosθ = ($(cos(θ)), $(cos(res.minimizer[1])))")
    println(io, "Initial x₀ (true) = [10.0, -10.0]")
    println(io, "Initial x₀ (optimized) = [$(res.minimizer[2]), $(res.minimizer[3])]")
    println(io, "Converged: $(Optim.converged(res))")
end

log_message("\nPerforming inference with optimized parameters...", log_file)
x0 = MvNormalMeanCovariance([ res.minimizer[2], res.minimizer[3] ], Matrix(Diagonal(100.0 * ones(2))))

result = infer(
    model = multivariate_state_space_model(
        θ = res.minimizer[1], 
        x0 = x0, 
        Q = Q, 
        P = P
    ), 
    data  = (y = y,), 
    free_energy = true
)

xmarginals = result.posteriors[:x]

# Plot multivariate results
result_plots = plot_multivariate_results(x, Q, xmarginals, output_dir=output_dir)
log_message("✓ Saved multivariate results plots:", log_file)
log_message("  - Time series: $(joinpath(output_dir, "multivariate_results.png"))", log_file)
log_message("  - Phase comparison: $(joinpath(output_dir, "multivariate_phase_comparison.png"))", log_file)

# Create animation of state evolution
log_message("\nCreating animation of state evolution...", log_file)
anim = @animate for i in 1:min(300, length(x))
    # Plot the phase space up to frame i
    p = scatter(getindex.(x[1:i], 1), getindex.(x[1:i], 2), 
                title="State Evolution (Frame $i/$(length(x)))",
                xlabel=L"x_1", ylabel=L"x_2",
                markersize=3, markerstrokewidth=0, alpha=0.6,
                label="True trajectory", color=:blue,
                xlim=(minimum(getindex.(x, 1))-1, maximum(getindex.(x, 1))+1),
                ylim=(minimum(getindex.(x, 2))-1, maximum(getindex.(x, 2))+1),
                size=(800, 600), dpi=150, legend=:topright)
    
    # Add the inferred trajectory if we have it
    if i <= length(xmarginals)
        scatter!(p, getindex.(mean.(xmarginals[1:i]), 1), 
                getindex.(mean.(xmarginals[1:i]), 2),
                label="Inferred trajectory", markersize=3,
                markerstrokewidth=0, alpha=0.6, color=:red)
    end
    
    # Add the current point with a larger marker
    if i > 0
        scatter!(p, [getindex(x[i], 1)], [getindex(x[i], 2)], 
                markersize=8, color=:green, label="Current state")
    end
    
    # Add rotation matrix direction indicators
    if i > 1
        # Draw the state transition as an arrow
        plot!(p, [getindex(x[i-1], 1), getindex(x[i], 1)], 
              [getindex(x[i-1], 2), getindex(x[i], 2)],
              arrow=true, linewidth=2, color=:black, label=nothing)
    end
end every 5

# Save the animation
gif(anim, joinpath(output_dir, "state_evolution.gif"), fps=10)
log_message("✓ Saved state evolution animation to $(joinpath(output_dir, "state_evolution.gif"))", log_file)

log_message("\n========== Optimization Complete ==========", log_file)
log_message("All results saved to the output directory: $output_dir", log_file)

# Make a copy of the executed script in the output directory for reproducibility
src_file = @__FILE__
dst_file = joinpath(output_dir, "executed_script.jl")
cp(src_file, dst_file, force=true)
viz_file = joinpath(@__DIR__, "visualization_utils.jl")
cp(viz_file, joinpath(output_dir, "visualization_utils.jl"), force=true)
log_message("✓ Saved copies of scripts for reproducibility", log_file)