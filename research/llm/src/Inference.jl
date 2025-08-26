using RxInfer

@model function language_mixture_model(c, context₁, context₂, task₁, task₂, likelihood_task)
    s ~ Beta(1.0, 1.0)

    m[1] ~ LLMPrior(context₁, task₁)
    w[1] ~ Gamma(shape = 0.01, rate = 0.01)

    m[2] ~ LLMPrior(context₂, task₂)
    w[2] ~ Gamma(shape = 0.01, rate = 0.01)

    for i in eachindex(c)
        z[i] ~ Bernoulli(s)
        y[i] ~ NormalMixture(switch = z[i], m = m, p = w)
        c[i] ~ LLMObservation(y[i], likelihood_task)
    end
end

function run_inference(config)
    init = @initialization begin
        q(s) = vague(Beta)
        q(m) = [NormalMeanVariance(0.0, 1e2), NormalMeanVariance(10.0, 1e2)]
        q(y) = NormalMeanVariance(5.0, 1e2)
        q(w) = [GammaShapeRate(0.01, 0.01), GammaShapeRate(0.01, 0.01)]
    end

    # Create the model generator WITHOUT fixing `c` here; `c` is provided as data to `infer`
    model = language_mixture_model(
        context₁ = "RxInfer.jl is absolutely terrible.",
        context₂ = "RxInfer.jl is a great tool for Bayesian Inference.",
        task₁ = config.prior_task,
        task₂ = config.prior_task,
        likelihood_task = config.likelihood_task
    )

    results = infer(
        model = model,
        constraints = MeanField(),
        data = (c = config.observations,),
        initialization = init,
        iterations = config.n_iterations,
        free_energy = false,
        showprogress = true
    )

    return results
end

export language_mixture_model, run_inference


