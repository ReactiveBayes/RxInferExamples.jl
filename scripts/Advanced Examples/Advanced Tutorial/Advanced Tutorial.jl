# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Advanced Examples/Advanced Tutorial/Advanced Tutorial.ipynb
# by notebooks_to_scripts.jl at 2025-08-07T12:32:28.195
#
# Source notebook: Advanced Tutorial.ipynb

using RxInfer, Plots

# the `@model` macro accepts a regular Julia function
@model function test_model1(s_mean, s_precision, y)
    
    # the `tilde` operator creates a functional dependency
    # between variables in our model and can be read as 
    # `sampled from` or `is modeled by`
    s ~ Normal(mean = s_mean, precision = s_precision)
    y ~ Normal(mean = s, precision = 1.0)
    
    # It is possible to return something from the model specification (including variables and nodes)
    return "Hello world"
end

@model function test_model2(y)
    
    if length(y) <= 1
        error("The `length` of `y` argument must be greater than one.")
    end
    
    s[1] ~ Normal(mean = 0.0, precision = 0.1)
    y[1] ~ Normal(mean = s[1], precision = 1.0)
    
    for i in eachindex(y)
        s[i] ~ Normal(mean = s[i - 1], precision = 1.0)
        y[i] ~ Normal(mean = s[i], precision = 1.0)
    end
    
end

# Timer that emits a new value every second and has an initial one second delay 
observable = timer(300, 300)

actor = (value) -> println(value)
subscription1 = subscribe!(observable, actor)

# We always need to unsubscribe from some observables
unsubscribe!(subscription1)

# We can modify our observables
modified = observable |> filter(d -> rem(d, 2) === 1) |> map(Int, d -> d ^ 2)

subscription2 = subscribe!(modified, (value) -> println(value))

unsubscribe!(subscription2)

@model function coin_toss_model(y)
    # We endow θ parameter of our model with some prior
    θ  ~ Beta(2.0, 7.0)
    # We assume that the outcome of each coin flip 
    # is modeled by a Bernoulli distribution
    y .~ Bernoulli(θ)
end

p = 0.75 # Bias of a coin

dataset = float.(rand(Bernoulli(p), 500));

result = infer(
    model = coin_toss_model(),
    data  = (y = dataset, )
)

println("Inferred bias is ", mean(result.posteriors[:θ]), " with standard deviation is ", std(result.posteriors[:θ]))

@model function online_coin_toss_model(θ_a, θ_b, y)
    θ ~ Beta(θ_a, θ_b)
    y ~ Bernoulli(θ)
end

autoupdates = @autoupdates begin 
    θ_a, θ_b = params(q(θ))
end

init = @initialization begin
    q(θ) = vague(Beta)
end

rxresult = infer(
    model = online_coin_toss_model(),
    data  = (y = dataset, ),
    autoupdates = autoupdates,
    historyvars = (θ = KeepLast(), ),
    keephistory = length(dataset),
    initialization = init,
    autostart = true
);

animation = @animate for i in 1:length(dataset)
    plot(mean.(rxresult.history[:θ][1:i]), ribbon = std.(rxresult.history[:θ][1:i]), title = "Online coin bias inference", label = "Inferred bias", legend = :bottomright)
    hline!([ p ], label = "Real bias", size = (600, 200))
end

gif(animation, "online-coin-bias-inference.gif", fps = 24, show_msg = false);

@model function test_model6(y)
    τ ~ Gamma(shape = 1.0, rate = 1.0) 
    μ ~ Normal(mean = 0.0, variance = 100.0)
    for i in eachindex(y)
        y[i] ~ Normal(mean = μ, precision = τ)
    end
end

constraints6 = @constraints begin
     q(μ, τ) = q(μ)q(τ) # Mean-Field over `μ` and `τ`
end

init = @initialization begin
    q(μ) = vague(NormalMeanPrecision)
    q(τ) = vague(GammaShapeRate)
end

dataset = rand(Normal(-3.0, inv(sqrt(5.0))), 1000);
result = infer(
    model          = test_model6(),
    data           = (y = dataset, ),
    constraints    = constraints6, 
    initialization = init,
    returnvars     = (μ = KeepLast(), τ = KeepLast()),
    iterations     = 10,
    free_energy    = true,
    showprogress   = true
)

println("μ: mean = ", mean(result.posteriors[:μ]), ", std = ", std(result.posteriors[:μ]))

println("τ: mean = ", mean(result.posteriors[:τ]), ", std = ", std(result.posteriors[:τ]))

@model function test_model7(y)
    τ ~ Gamma(shape = 1.0, rate = 1.0) 
    μ ~ Normal(mean = 0.0, variance = 100.0)
    for i in eachindex(y)
        y[i] ~ Normal(mean = μ, precision = τ)
    end
end

constraints7 = @constraints begin 
    q(μ) :: PointMassFormConstraint()
    
    q(μ, τ) = q(μ)q(τ) # Mean-Field over `μ` and `τ`
end

dataset = rand(Normal(-3.0, inv(sqrt(5.0))), 1000);
result = infer(
    model          = test_model7(),
    data           = (y = dataset, ),
    constraints    = constraints7, 
    initialization = init,
    returnvars     = (μ = KeepLast(), τ = KeepLast()),
    iterations     = 10,
    free_energy    = true,
    showprogress   = true
)

println("μ: mean = ", mean(result.posteriors[:μ]), ", std = ", std(result.posteriors[:μ]))

println("τ: mean = ", mean(result.posteriors[:τ]), ", std = ", std(result.posteriors[:τ]))

@meta begin 
     AR(s, θ, γ) -> ARMeta(Multivariate, 5, ARsafe())
end

@model function coin_toss_model_log(y)
    θ ~ Beta(2.0, 7.0) where { pipeline = LoggerPipelineStage("θ") }
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)  where { pipeline = LoggerPipelineStage("y[$i]") }
    end
end

dataset = float.(rand(Bernoulli(p), 5));
result = infer(
    model = coin_toss_model_log(),
    data  = (y = dataset, )
)