@model function gc_car_model(y, A, B, Q, R, x0_mean, x0_cov)
    x[1] ~ MvNormal(μ = x0_mean, Σ = x0_cov)
    y[1] ~ MvNormal(μ = B * x[1], Σ = R)
    for t in 2:length(y)
        x[t] ~ MvNormal(μ = A * x[t-1], Σ = Q)
        y[t] ~ MvNormal(μ = B * x[t], Σ = R)
    end