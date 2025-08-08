@model function multivariate_gaussian_mixture_model(nr_mixtures, priors, y)
    local m
    local w

    for k in 1:nr_mixtures        
        m[k] ~ priors[k]
        w[k] ~ Wishart(3, 1e2*diagm(ones(2)))
    end