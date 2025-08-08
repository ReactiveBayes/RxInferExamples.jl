@model function univariate_gaussian_mixture_model(y)
    
    s ~ Beta(1.0, 1.0)

    m[1] ~ Normal(mean = -2.0, variance = 1e3)
    w[1] ~ Gamma(shape = 0.01, rate = 0.01)

    m[2] ~ Normal(mean = 2.0, variance = 1e3)
    w[2] ~ Gamma(shape = 0.01, rate = 0.01)

    for i in eachindex(y)
        z[i] ~ Bernoulli(s)
        y[i] ~ NormalMixture(switch = z[i], m = m, p = w)
    end