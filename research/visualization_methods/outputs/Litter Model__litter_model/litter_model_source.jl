@model function litter_model(x, αᴳᵃᵐ, θᴳᵃᵐ)
    ## prior on θ parameter of the model
    θ ~ Gamma(shape=αᴳᵃᵐ, rate=θᴳᵃᵐ) ## 1 Gamma factor

    ## assume daily number of litter incidents is a Poisson distribution
    for i in eachindex(x)
        x[i] ~ Poisson(θ) ## not θ̃; N Poisson factors
    end