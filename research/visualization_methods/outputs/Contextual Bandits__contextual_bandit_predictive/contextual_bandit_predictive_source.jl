@model function contextual_bandit_predictive(reward, priors, current_context)
    local θ
    local γ
    local τ

    # Prior for each arm's parameters
    for k in 1:n_arms
        θ[k] ~ priors[:θ][k]
        γ[k] ~ priors[:γ][k]
    end