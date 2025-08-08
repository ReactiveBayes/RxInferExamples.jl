@model function rotate_ssm(y, x0, A, B, P, Q)
    x_prior ~ x0
    x_prev = x_prior
    
    for i in 1:length(y)
        x[i] ~ MvNormalMeanCovariance(A * x_prev, P)
        y[i] ~ MvNormalMeanCovariance(B * x[i], Q)
        x_prev = x[i]
    end