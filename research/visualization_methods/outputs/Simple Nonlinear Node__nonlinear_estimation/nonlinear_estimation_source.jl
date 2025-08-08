@model function nonlinear_estimation(y, θ_μ, m_μ, θ_σ, m_σ)
    
    # define a distribution for the two variables
    θ ~ Normal(mean = θ_μ, variance = θ_σ)
    m ~ Normal(mean = m_μ, variance = m_σ)

    # define a nonlinear node
    w ~ NonlinearNode(θ)

    # We consider the outcome to be normally distributed
    for i in eachindex(y)
        y[i] ~ Normal(mean = m, precision = w)
    end