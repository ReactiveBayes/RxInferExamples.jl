@model function online_coin_toss_model(θ_a, θ_b, y)
    θ ~ Beta(θ_a, θ_b)
    y ~ Bernoulli(θ)
end