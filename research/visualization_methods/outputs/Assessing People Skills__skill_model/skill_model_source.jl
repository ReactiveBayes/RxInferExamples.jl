@model function skill_model(r)

    local s
    # Priors
    for i in eachindex(r)
        s[i] ~ Bernoulli(0.5)
    end