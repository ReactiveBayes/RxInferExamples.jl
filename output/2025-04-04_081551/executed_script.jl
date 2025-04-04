#!/usr/bin/env julia
# Parameter Optimisation with Optim.jl
# This script demonstrates parameter optimization using RxInfer.jl with Optim.jl
# Originally from: Parameter Optimisation with Optim.jl.ipynb

using RxInfer, StableRNGs, LinearAlgebra, Plots, Optim, Dates

# Load visualization utilities
include(joinpath(@__DIR__, "visualization_utils.jl"))

# Check if the required packages are installed
for pkg in ["RxInfer", "StableRNGs", "LinearAlgebra", "Plots", "Optim", "Dates"]
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

log_message("\nOptimizing parameters using Gradient Descent...", log_file)
res = optimize(f, ones(2), GradientDescent(), 
    Optim.Options(g_tol = 1e-3, iterations = 100, store_trace = true, show_trace = true, show_every = 10))

# Save optimization trace
if hasfield(typeof(res), :trace)
    trace_values = [state.value for state in res.trace]
    save_optimization_trace(trace_values, "Univariate Model Optimization", output_dir=output_dir)
    log_message("✓ Saved optimization trace to $(joinpath(output_dir, "optimization_trace.png"))", log_file)
end

log_message("\nResults for univariate model:", log_file)
log_message("Real value vs Optimized", log_file)
log_message("Real:      $([ 1.0, c_real ])", log_file)
log_message("Optimized: $(res.minimizer)", log_file)
log_message("Converged: $(Optim.converged(res))", log_file)

# Save results to file
open(joinpath(output_dir, "univariate_results.txt"), "w") do io
    println(io, "Real value vs Optimized")
    println(io, "Real:      $([ 1.0, c_real ])")
    println(io, "Optimized: $(res.minimizer)")
    println(io, "Converged: $(Optim.converged(res))")
end

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

# Plot multivariate data
plot_multivariate_data(x, Q, output_dir=output_dir)
log_message("✓ Saved multivariate data plot to $(joinpath(output_dir, "multivariate_data.png"))", log_file)

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

log_message("\nOptimizing parameters using LBFGS...", log_file)
res = optimize(f, zeros(3), LBFGS(), 
    Optim.Options(f_tol = 1e-14, g_tol = 1e-12, show_trace = true, show_every = 10))

# Save optimization trace for multivariate model
if hasfield(typeof(res), :trace)
    trace_values = [state.value for state in res.trace]
    save_optimization_trace(trace_values, "Multivariate Model Optimization", output_dir=output_dir)
    log_message("✓ Saved multivariate optimization trace", log_file)
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
plot_multivariate_results(x, Q, xmarginals, output_dir=output_dir)
log_message("✓ Saved results plot to $(joinpath(output_dir, "multivariate_results.png"))", log_file)

log_message("\n========== Optimization Complete ==========", log_file)
log_message("All results saved to the output directory: $output_dir", log_file)

# Make a copy of the executed script in the output directory for reproducibility
src_file = @__FILE__
dst_file = joinpath(output_dir, "executed_script.jl")
cp(src_file, dst_file, force=true)
log_message("✓ Saved a copy of this script to $dst_file", log_file)