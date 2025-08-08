@model function univariate_state_space_model(y, x_prior, c, v)
    
    x0 ~ Normal(mean = mean(x_prior), variance = var(x_prior))
    x_prev = x0

    for i in eachindex(y)
        x[i] := x_prev + c
        y[i] ~ Normal(mean = x[i], variance = v)
        x_prev = x[i]
    end