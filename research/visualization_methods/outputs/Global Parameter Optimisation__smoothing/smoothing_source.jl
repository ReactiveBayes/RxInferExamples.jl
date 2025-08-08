@model function smoothing(y, x0, c, P)
    
    x_prior ~ Normal(mean = mean(x0), var = var(x0)) 
    x_prev = x_prior

    for i in eachindex(y)
        x[i] ~ x_prev + c
        y[i] ~ Normal(mean = x[i], var = P)
        x_prev = x[i]
    end