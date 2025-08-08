@model function BIFM_smoother(y, A, B, C, μu, Wu, Wy)

    # fetch dimensionality
    dim_lat = size(A, 1)
    
    # set priors
    z_prior ~ MvNormal(mean = zeros(dim_lat), precision = 1e-5*diagm(ones(dim_lat)))
    z[1]  ~ BIFMHelper(z_prior)
    
    # loop through observations
    for i in eachindex(y)

        # specify input as random variable
        u[i]   ~ MvNormal(mean = μu, precision = Wu)

        # specify observation
        yt[i]  ~ BIFM(u[i], z[i], new(z[i+1])) where { meta = BIFMMeta(A, B, C) }
        y[i]   ~ MvNormal(mean = yt[i], precision = Wy)
    end