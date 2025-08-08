@model function invertible_neural_network(y)

    # specify prior
    z_μ ~ MvNormalMeanCovariance(zeros(2), huge*diagm(ones(2)))
    z_Λ ~ Wishart(2.0, tiny*diagm(ones(2)))

    # specify observations
    for k in eachindex(y)

        # specify latent state
        x[k] ~ MvNormalMeanPrecision(z_μ, z_Λ)

        # specify transformed latent value
        y_lat[k] ~ Flow(x[k])

        # specify observations
        y[k] ~ MvNormalMeanCovariance(y_lat[k], tiny*diagm(ones(2)))

    end