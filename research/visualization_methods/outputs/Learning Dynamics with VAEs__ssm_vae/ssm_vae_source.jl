@model function ssm_vae(y)
    Λₛ ~ Wishart(4, diageye(2)) # Precision matrix for the transition matrix
    Hₛ ~ MvNormal(μ = zeros(4), Λ = diageye(4)) # Vectorized 2×2 transition matrix prior

    # Initial state
    x[1] ~ MvNormal(μ = zeros(2), Σ = diageye(2)) 
    y[1] ~ VAENode(x[1])

    # State space model evolution
    for t in 2:length(y)
        x[t] ~ ContinuousTransition(x[t-1], Hₛ, Λₛ)  # equivalent to x[t] := Hₛ * x[t-1] + ϵ[t]
        y[t] ~ VAENode(x[t])
    end