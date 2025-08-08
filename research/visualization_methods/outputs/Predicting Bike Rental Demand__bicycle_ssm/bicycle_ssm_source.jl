@model function bicycle_ssm(x, y, h0, θ0, a0, Q, s)

    a ~ a0
    θ ~ θ0
    h_prior ~ h0

    h_prev = h_prior
    for i in eachindex(y)
        
        h[i] ~ MvNormal(μ=transition(a, h_prev), Σ=Q)
        x[i] ~ MvNormal(μ=h[i], Σ=diageye(state_dim))
        y[i] ~ Normal(μ=dotsoftplus(θ, h[i]), σ²=s)
        h_prev = h[i]
    end