@model function smoothing(x0, y)
    
    P ~ Gamma(shape = 0.001, scale = 0.001)
    x_prior ~ Normal(mean = mean(x0), var = var(x0)) 

    local x
    x_prev = x_prior

    for i in 1:length(y)
        x[i] ~ Normal(mean = x_prev, precision = 1.0)
        y[i] ~ Normal(mean = x[i], precision = P)
        
        x_prev = x[i]
    end