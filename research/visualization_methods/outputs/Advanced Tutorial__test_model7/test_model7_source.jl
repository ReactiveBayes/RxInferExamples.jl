@model function test_model7(y)
    τ ~ Gamma(shape = 1.0, rate = 1.0) 
    μ ~ Normal(mean = 0.0, variance = 100.0)
    for i in eachindex(y)
        y[i] ~ Normal(mean = μ, precision = τ)
    end