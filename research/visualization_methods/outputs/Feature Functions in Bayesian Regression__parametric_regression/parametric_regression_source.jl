@model function parametric_regression(ϕs, x, y, μ, Σ, Λ)
    # Prior distribution over parameters ω
    ω ~ MvNormal(mean = μ, covariance = Σ)
    
    # Design matrix Φₓ where each element is ϕᵢ(xⱼ)
    Φₓ = [ϕ(xᵢ) for xᵢ in x, ϕ in ϕs]
    
    # Likelihood of observations y given parameters ω
    y ~ MvNormal(mean = Φₓ * ω, covariance = Λ)
end