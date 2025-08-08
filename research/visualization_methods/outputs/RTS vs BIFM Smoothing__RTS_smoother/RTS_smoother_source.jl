@model function RTS_smoother(y, A, B, C, μu, Wu, Wy)
    
    # fetch dimensionality
    dim_lat = size(A, 1)
    dim_out = size(C, 1)
    
    # set initial hidden state
    z_prev ~ MvNormal(mean = zeros(dim_lat), precision = 1e-5*diagm(ones(dim_lat)))

    # loop through observations
    for i in eachindex(y)

        # specify input as random variable
        u[i] ~ MvNormal(mean = μu, precision = Wu)
        
        # specify updated hidden state
        z[i] ~ A * z_prev + B * u[i]
        
        # specify observation
        y[i] ~ MvNormal(mean = C * z[i], precision = Wy)
        
        # update last/previous hidden state
        z_prev = z[i]

    end