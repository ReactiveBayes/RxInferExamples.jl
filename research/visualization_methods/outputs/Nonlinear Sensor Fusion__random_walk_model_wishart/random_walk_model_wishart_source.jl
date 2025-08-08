@model function random_walk_model_wishart(y)
    # set priors on precision matrices
    Q ~ Wishart(3, diageye(2))
    R ~ Wishart(4, diageye(3))

    # specify initial estimates of the location
    z[1] ~ MvNormalMeanCovariance(zeros(2), diageye(2)) 
    y[1] ~ MvNormalMeanCovariance(compute_distances(z[1]), diageye(3))

    # loop over time steps
    for t in 2:length(y)

        # specify random walk state transition model
        z[t] ~ MvNormalMeanPrecision(z[t-1], Q)

        # specify non-linear distance observations model
        y[t] ~ MvNormalMeanPrecision(compute_distances(z[t]), R)
        
    end