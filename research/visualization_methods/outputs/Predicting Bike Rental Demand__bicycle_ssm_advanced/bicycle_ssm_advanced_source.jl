@model function bicycle_ssm_advanced(x, y, h0, θ0, a0, P0, γ0)

    a ~ a0
    θ ~ θ0
    h_prior ~ h0
    P ~ P0
    γ ~ γ0

    h_prev = h_prior
    for i in eachindex(y)
        
        h[i] ~ CTransition(h_prev, a, P)
        x[i]  ~ MvNormal(μ=h[i], Λ=diageye(state_dim))
        _y[i] ~ softdot(θ, h[i], γ)
        y[i] ~ Normal(μ=softplus(_y[i]), γ=1e4)
        h_prev = h[i]
    end