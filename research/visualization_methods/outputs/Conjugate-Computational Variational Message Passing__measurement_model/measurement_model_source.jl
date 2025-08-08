@model function measurement_model(y)

    # set priors on precision parameters
    τ ~ Gamma(shape = 0.01, rate = 0.01)
    γ ~ Gamma(shape = 0.01, rate = 0.01)
    
    # specify estimate of initial location
    z[1] ~ Normal(mean = 0, precision = τ)
    y[1] ~ Normal(mean = compute_squared_distance(z[1]), precision = γ)

    # loop over observations
    for t in 2:length(y)

        # specify state transition model
        z[t] ~ Normal(mean = z[t-1] + 1, precision = τ)

        # specify non-linear observation model
        y[t] ~ Normal(mean = compute_squared_distance(z[t]), precision = γ)
        
    end