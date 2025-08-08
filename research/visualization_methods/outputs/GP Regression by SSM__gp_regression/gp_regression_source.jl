@model function gp_regression(y, P, A, Q, H, var_noise)
    f_prev ~ MvNormal(μ = zeros(length(H)), Σ = P) #initial state
    for i in eachindex(y)
        f[i] ~ MvNormal(μ = A[i] * f_prev,Σ = Q[i])
        y[i] ~ Normal(μ = dot(H, f[i]), var = var_noise)
        f_prev = f[i]
    end