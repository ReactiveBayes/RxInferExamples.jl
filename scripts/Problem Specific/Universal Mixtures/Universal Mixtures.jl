# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Problem Specific/Universal Mixtures/Universal Mixtures.ipynb
# by notebooks_to_scripts.jl at 2025-04-29T06:39:07.660
#
# Source notebook: Universal Mixtures.ipynb

using RxInfer, Distributions, Random, Plots

rθ = range(0, 1, length = 1000)
p = plot(title = "prior beliefs")
plot!(rθ, (x) -> pdf(Beta(7.0, 2.0), x), fillalpha=0.3, fillrange = 0, label="P(θ) John", c=1)
plot!(rθ, (x) -> pdf(Beta(2.0, 7.0), x), fillalpha=0.3, fillrange = 0, label="p(θ) Jane", c=3,)

true_coin = Bernoulli(0.25)
nr_throws = 10
dataset = Int.(rand(MersenneTwister(42), true_coin, nr_throws))
nr_heads, nr_tails = sum(dataset), nr_throws-sum(dataset)
println("experimental outcome: \n - heads: ", nr_heads, "\n - tails: ", nr_tails);

@model function beta_model_john(y)

    # specify John's prior model over θ
    θ ~ Beta(7.0, 2.0)

    # create likelihood models
    y .~ Bernoulli(θ)
    
end

@model function beta_model_jane(y)

    # specify Jane's prior model over θ
    θ ~ Beta(2.0, 7.0)

    # create likelihood models
    y .~ Bernoulli(θ)
    
end

result_john = infer(
    model = beta_model_john(), 
    data  = (y = dataset, ),
    free_energy = true,
)

result_jane = infer(
    model = beta_model_jane(), 
    data  = (y = dataset, ),
    free_energy = true
)

rθ = range(0, 1, length = 1000)
p = plot(title = "posterior beliefs")
plot!(rθ, (x) -> pdf(result_john.posteriors[:θ], x), fillalpha=0.3, fillrange = 0, label="P(θ|y) John", c=1)
plot!(rθ, (x) -> pdf(result_jane.posteriors[:θ], x), fillalpha=0.3, fillrange = 0, label="p(θ|y) Jane", c=3,)

rθ = range(0, 1, length = 1000)
p = plot(title = "prior belief")
plot!(rθ, (x) -> pdf(MixtureDistribution([Beta(2.0, 7.0), Beta(7.0, 2.0)], [ 0.3, 0.7 ]), x), fillalpha=0.3, fillrange = 0, label="P(θ) Mary", c=1)
plot!(rθ, (x) -> 0.7*pdf(Beta(7.0, 2.0), x), c=3, label="")
plot!(rθ, (x) -> 0.3*pdf(Beta(2.0, 7.0), x), c=3, label="")

@model function beta_model_mary(y)

    # specify John's and Jane's prior models over θ
    θ_jane ~ Beta(2.0, 7.0)
    θ_john ~ Beta(7.0, 2.0)

    # specify initial guess as to who is right
    john_is_right ~ Bernoulli(0.7) 

    # specify mixture prior Distribution
    θ ~ Mixture(switch = john_is_right, inputs = [θ_jane, θ_john])

    # create likelihood models
    y .~ Bernoulli(θ)
    
end

result_mary = infer(
    model = beta_model_mary(), 
    data  = (y = dataset, ),
    returnvars = (θ = KeepLast(), θ_john = KeepLast(), θ_jane = KeepLast(), john_is_right = KeepLast()),
    addons = AddonLogScale(),
    postprocess = UnpackMarginalPostprocess(),
)

rθ = range(0, 1, length = 1000)
p = plot(title = "posterior belief")
plot!(rθ, (x) -> pdf(result_mary.posteriors[:θ], x), fillalpha=0.3, fillrange = 0, label="P(θ|y) Mary", c=1)
plot!(rθ, (x) -> result_mary.posteriors[:θ].weights[1] * pdf(component(result_mary.posteriors[:θ], 1), x), label="", c=3)
plot!(rθ, (x) -> result_mary.posteriors[:θ].weights[2] * pdf(component(result_mary.posteriors[:θ], 2), x), label="", c=3)