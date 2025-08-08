@model function ARMA(x, x_prev, priors, p_order, q_order)
    
    # arguments
    c = zeros(q_order); c[1] = 1.0;
    S = shift(q_order); # MA

    # set priors
    γ    ~ priors[:γ]
    η    ~ priors[:η]
    θ    ~ priors[:θ]
    τ    ~ priors[:τ]
    
    h[1] ~ priors[:h]
    z[1] ~ AR(h[1], η, τ)
    e[1] ~ Normal(mean = 0.0, precision = γ)
    x[1] ~ dot(c, z[1]) + dot(θ, x_prev[1]) + e[1]

    for t in 1:length(x)-1
        h[t+1] ~ S * h[t] + c * e[t]
        z[t+1] ~ AR(h[t+1], η, τ)
        e[t+1] ~ Normal(mean = 0.0, precision = γ)
        x[t+1] ~ dot(c, z[t+1]) + dot(θ, x_prev[t+1]) + e[t+1]
    end