@model function gamma_mixture_model(y, parameters)

    # fetch information from struct
    nmixtures = parameters.nmixtures
    priors_as = parameters.priors_as
    priors_bs = parameters.priors_bs
    prior_s   = parameters.prior_s

    # set prior on global selection variable
    s ~ Dirichlet(probvec(prior_s))

    # allocate variables for mixtures
    local as
    local bs

    # set priors on variables of mixtures
    for i in 1:nmixtures
        as[i] ~ Gamma(shape = shape(priors_as[i]), rate = rate(priors_as[i]))
        bs[i] ~ Gamma(shape = shape(priors_bs[i]), rate = rate(priors_bs[i]))
    end