# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Basic Examples/Incomplete Data/Incomplete Data.ipynb
# by notebooks_to_scripts.jl at 2025-08-07T12:32:28.736
#
# Source notebook: Incomplete Data.ipynb

using RxInfer, LinearAlgebra

@model function incomplete_data(y, dim)
    Λ ~ Wishart(dim, diagm(ones(dim)))
    m ~ MvNormal(mean=zeros(dim), precision=diagm(ones(dim)))
    for i in 1:size(y, 1)
        x[i] ~ MvNormal(mean=m, precision=Λ)
        for j in 1:dim
            y[i, j] ~ softdot(x[i], StandardBasisVector(dim, j), huge)
        end
    end
end

n_samples = 100

real_m = [13.0, 1.0, 5.0, 4.0, -20.0, 10.0]
dimension = length(real_m)
real_Λ = diagm(ones(dimension))

real_x = [rand(MvNormal(real_m, inv(real_Λ))) for _ in 1:n_samples]
incomplete_x = Vector{Vector{Union{Float64, Missing}}}(copy(real_x))

for i in 1:n_samples
    incomplete_x[i][rand(1:dimension)] = missing
end

# Create a matrix instead of vector of vectors
observations = Matrix{Union{Float64, Missing}}(undef, n_samples, dimension)

for i in 1:n_samples
    for j in 1:dimension
        observations[i, j] = incomplete_x[i][j]
    end
end

# We assume independence between the precision matrix and other variables.
constraints = @constraints begin
    q(x, m, Λ) = q(x, m)q(Λ) 
end

# We need to initialize the precision matrix.
init = @initialization begin
    q(Λ) = Wishart(dimension, diagm(ones(dimension)))
end

result = infer(model=incomplete_data(dim=dimension), data=(y=observations,), constraints=constraints, initialization=init, showprogress=true, iterations=100);

# Extract final posterior estimates
estimated_covariance = inv(mean(result.posteriors[:Λ][end]))
estimated_mean = mean(result.posteriors[:m][end])

println("True mean: ", real_m[1:dimension])  # Show first 5 elements
println("Estimated mean: ", estimated_mean[1:dimension])
println()
println("True covariance (diagonal): ", diag(inv(real_Λ))[1:dimension])
println("Estimated covariance (diagonal): ", diag(estimated_covariance)[1:dimension])

# Simple plotting code for the RxInfer incomplete data tutorial
using Plots, Distributions

function plot_posterior_distributions(result, real_m, real_Λ, max_dim=3)
    # Get final posteriors
    final_m_posterior = result.posteriors[:m][end]
    final_Λ_posterior = result.posteriors[:Λ][end]
    
    # Plot mean posterior for first few dimensions
    p1 = plot(title="Posterior Distribution of Mean (first $max_dim dimensions)", 
              xlabel="Value", ylabel="Density")
    
    for i in 1:max_dim
        # Extract marginal distribution for dimension i
        marginal_mean = mean(final_m_posterior)[i]
        marginal_var = inv(mean(final_Λ_posterior))[i,i]
        
        # Plot the Gaussian
        x_range = range(marginal_mean - 3*sqrt(marginal_var), 
                       marginal_mean + 3*sqrt(marginal_var), length=100)
        gaussian = Normal(marginal_mean, sqrt(marginal_var))
        plot!(p1, x_range, pdf.(gaussian, x_range), 
              label="Dimension $i", linewidth=2, color=i)
        
        # Add vertical line for true value with same color
        vline!(p1, [real_m[i]], color=i, linestyle=:dash, alpha=0.7, 
               linewidth=2, label="")
    end
    
    plot(p1)
end


plot_posterior_distributions(result, real_m, real_Λ, 6)