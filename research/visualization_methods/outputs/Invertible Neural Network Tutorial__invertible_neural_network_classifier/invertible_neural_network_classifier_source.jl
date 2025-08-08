@model function invertible_neural_network_classifier(x, y)

    # specify observations
    for k in eachindex(x)

        # specify latent state
        x_lat[k] ~ MvNormalMeanPrecision(x[k], 1e3*diagm(ones(2)))

        # specify transformed latent value
        y_lat1[k] ~ Flow(x_lat[k])
        y_lat2[k] ~ dot(y_lat1[k], [1, 1])

        # specify observations
        y[k] ~ Probit(y_lat2[k]) # default: where { pipeline = RequireMessage(in = NormalMeanPrecision(0, 1.0)) }

    end