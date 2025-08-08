@model function multivariate_state_space_model(y, θ, x0, Q, P)
    
    x_prior ~ MvNormal(mean = mean(x0), cov = cov(x0))
    x_prev = x_prior
    
    A = [ cos(θ) -sin(θ); sin(θ) cos(θ) ]
    
    for i in eachindex(y)
        x[i] ~ MvNormal(mean = A * x_prev, covariance = Q)
        y[i] ~ MvNormal(mean = x[i], covariance = P)
        x_prev = x[i]
    end