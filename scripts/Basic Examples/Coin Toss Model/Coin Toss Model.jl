# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Basic Examples/Coin Toss Model/Coin Toss Model.ipynb
# by notebooks_to_scripts.jl at 2025-03-31T09:50:41.006
#
# Source notebook: Coin Toss Model.ipynb

using RxInfer, Random

rng = MersenneTwister(42)
n = 500
θ_real = 0.75
distribution = Bernoulli(θ_real)

dataset = float.(rand(rng, Bernoulli(θ_real), n));

# GraphPPL.jl export `@model` macro for model specification
# It accepts a regular Julia function and builds a factor graph under the hood
@model function coin_model(y, a, b)

    # We endow θ parameter of our model with "a" prior
    θ ~ Beta(a, b)
    # note that, in this particular case, the `Uniform(0.0, 1.0)` prior will also work.
    # θ ~ Uniform(0.0, 1.0)

    # here, the outcome of each coin toss is governed by the Bernoulli distribution
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)
    end

end

result = infer(
    model = coin_model(a = 4.0, b = 8.0), 
    data  = (y = dataset,)
)

θestimated = result.posteriors[:θ]

using Plots

rθ = range(0, 1, length = 1000)

p = plot(title = "Inference results")

plot!(rθ, (x) -> pdf(Beta(4.0, 8.0), x), fillalpha=0.3, fillrange = 0, label="P(θ)", c=1,)
plot!(rθ, (x) -> pdf(θestimated, x), fillalpha=0.3, fillrange = 0, label="P(θ|y)", c=3)
vline!([θ_real], label="Real θ")