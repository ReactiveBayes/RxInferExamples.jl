# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Basic Examples/Bayesian Multinomial Regression/Bayesian Multinomial Regression.ipynb
# by notebooks_to_scripts.jl at 2025-04-04T08:03:37.567
#
# Source notebook: Bayesian Multinomial Regression.ipynb

using RxInfer, Plots, StableRNGs, Distributions, ExponentialFamily, StatsPlots
import ExponentialFamily: softmax 

function generate_multinomial_data(rng=StableRNG(123);N = 20, k=9, nsamples = 1000)
    Ψ = randn(rng, k)
    p = softmax(Ψ)
    X = rand(rng, Multinomial(N, p), nsamples)
    X= [X[:,i] for i in 1:size(X,2)];
    return X, Ψ,p
end

nsamples = 5000
N = 30
k = 40
X, Ψ, p = generate_multinomial_data(N=N,k=k,nsamples=nsamples);

@model function multinomial_model(obs, N, ξ_ψ, W_ψ)
    ψ ~ MvNormalWeightedMeanPrecision(ξ_ψ, W_ψ)
    obs .~ MultinomialPolya(N, ψ) where {dependencies = RequireMessageFunctionalDependencies(ψ = MvNormalWeightedMeanPrecision(ξ_ψ, W_ψ))}
end

result = infer(
    model = multinomial_model(ξ_ψ=zeros(k-1), W_ψ=rand(Wishart(3, diageye(k-1))), N=N),
    data = (obs=X, ),
    iterations = 50,
    free_energy = true,
    showprogress = true,
    options = (
        limit_stack_depth = 100,
    )
)

plot(result.free_energy[1:end], 
     title="Free Energy Over Iterations",
     xlabel="Iteration",
     ylabel="Free Energy",
     linewidth=2,
     legend=false,
     grid=true,
     )

predictive = @call_rule MultinomialPolya(:x, Marginalisation) (q_N = PointMass(N), q_ψ = result.posteriors[:ψ][end], meta = MultinomialPolyaMeta(21))
println("Estimated data generation probabilities: $(predictive.p)")
println("True data generation probabilities: $(p)")

mse = mean((predictive.p - p).^2);
println("MSE between estimated and true data generation probabilities: $mse")

@model function multinomial_regression(obs, N, X, ϕ, ξβ, Wβ)
    β ~ MvNormalWeightedMeanPrecision(ξβ, Wβ)
    for i in eachindex(obs)
        Ψ[i] := ϕ(X[i])*β
        obs[i] ~ MultinomialPolya(N, Ψ[i]) where {dependencies = RequireMessageFunctionalDependencies(ψ = MvNormalWeightedMeanPrecision(zeros(length(obs[i])-1), diageye(length(obs[i])-1)))}
    end
end


function generate_regression_data(rng=StableRNG(123);ϕ = identity,N = 3, k=5, nsamples = 1000)
    β = randn(rng, k)
    X = randn(rng, nsamples, k, k)
    X = [X[i,:,:] for i in 1:size(X,1)];
    Ψ = ϕ.(X)
    p = map(x -> logistic_stick_breaking(x*β), Ψ)
    return map(x -> rand(rng, Multinomial(N, x)), p), X, β, p
end


ϕ = x -> sin(x)
obs_regression, X_regression, β_regression, p_regression = generate_regression_data(;nsamples = 5000, ϕ = ϕ);

reg_results = infer(  
    model = multinomial_regression(N = 3, ϕ = ϕ, ξβ = zeros(5), Wβ = rand(Wishart(5, diageye(5)))),
    data = (obs=obs_regression,X = X_regression ),
    iterations = 20,
    free_energy = true,
    showprogress = true,
    returnvars = KeepLast(),
    options = (
        limit_stack_depth = 100,
    ) 
)

println("estimated β: with mean and covariance: $(mean_cov(reg_results.posteriors[:β]))")
println("true β: $(β_regression)")

plot(reg_results.free_energy,
title="Free Energy Over Iterations",
xlabel="Iteration",
ylabel="Free Energy",
linewidth=2,
legend=false,
grid=true,)

mse_β =  mean((mean(reg_results.posteriors[:β]) - β_regression).^2)
println("MSE of β estimate: $mse_β")


# Previous helper functions remain the same
σ(x) = 1 / (1 + exp(-x))
σ_inv(x) = log(x / (1 - x))

function jacobian_det(π)
    K = length(π)
    det = 1.0
    for k in 1:(K-1)
        num = 1 - sum(π[1:(k-1)])
        den = π[k] * (1 - sum(π[1:k]))
        det *= num / den
    end
    return det
end

function ψ_to_π(ψ::Vector{Float64})
    K = length(ψ) + 1
    π = zeros(K)
    for k in 1:(K-1)
        π[k] = σ(ψ[k]) * (1 - sum(π[1:(k-1)]))
    end
    π[K] = 1 - sum(π[1:(K-1)])
    return π
end

function π_to_ψ(π)
    K = length(π)
    ψ = zeros(K-1)
    ψ[1] = σ_inv(π[1])
    for k in 2:(K-1)
        ψ[k] = σ_inv(π[k] / (1 - sum(π[1:(k-1)])))
    end
    return ψ
end

# Function to compute density in simplex coordinates
function compute_simplex_density(x::Float64, y::Float64, Σ::Matrix{Float64})
    # Check if point is inside triangle
    if y < 0 || y > 1 || x < 0 || x > 1 || (x + y) > 1
        return 0.0
    end
    
    # Convert from simplex coordinates to π
    π1 = x
    π2 = y
    π3 = 1 - x - y
    
    try
        ψ = π_to_ψ([π1, π2, π3])
        # Compute Gaussian density
        dist = MvNormal(zeros(2), Σ)
        return pdf(dist, ψ) * abs(jacobian_det([π1, π2, π3]))
    catch
        return 0.0
    end
   
end

function plot_transformed_densities()
    # Create three different covariance matrices
    ###For higher variances values needs scaling for proper visualization.
    σ² = 1.0
    Σ_corr = [σ² 0.9σ²; 0.9σ² σ²]
    Σ_anticorr = [σ² -0.9σ²; -0.9σ² σ²]
    Σ_uncorr = [σ² 0.0; 0.0 σ²]
    
    # Plot Gaussian densities
    ψ1, ψ2 = range(-4sqrt(σ²), 4sqrt(σ²), length=500), range(-4sqrt(σ²), 4sqrt(σ²), length=100)
    
    p1 = contour(ψ1, ψ2, (x,y) -> pdf(MvNormal(zeros(2), Σ_corr), [x,y]),
                 title="Correlated Prior", xlabel="ψ₁", ylabel="ψ₂")
    p2 = contour(ψ1, ψ2, (x,y) -> pdf(MvNormal(zeros(2), Σ_anticorr), [x,y]),
                 title="Anti-correlated Prior", xlabel="ψ₁", ylabel="ψ₂")
    p3 = contour(ψ1, ψ2, (x,y) -> pdf(MvNormal(zeros(2), Σ_uncorr), [x,y]),
                 title="Uncorrelated Prior", xlabel="ψ₁", ylabel="ψ₂")
    
    # Plot simplex densities
    n_points = 500
    x = range(0, 1, length=n_points)
    y = range(0, 1, length=n_points)
    
    # Plot simplices
    p4 = contour(x, y, (x,y) -> compute_simplex_density(x, y, Σ_corr),
                 title="Correlated Simplex")
    
    # Add simplex boundaries and median lines
    plot!(p4, [0,1,0,0], [0,0,1,0], color=:black, label="")  # Triangle boundaries
    
    p5 = contour(x, y, (x,y) -> compute_simplex_density(x, y, Σ_anticorr),
                 title="Anti-correlated Simplex")
    plot!(p5, [0,1,0,0], [0,0,1,0], color=:black, label="")
    
    p6 = contour(x, y, (x,y) -> compute_simplex_density(x, y, Σ_uncorr),
                 title="Uncorrelated Simplex")
    plot!(p6, [0,1,0,0], [0,0,1,0], color=:black, label="")
    
    # Combine all plots
    plot(p1, p2, p3, p4, p5, p6, layout=(2,3), size=(900,600))
end

# Generate the plots
plot_transformed_densities()