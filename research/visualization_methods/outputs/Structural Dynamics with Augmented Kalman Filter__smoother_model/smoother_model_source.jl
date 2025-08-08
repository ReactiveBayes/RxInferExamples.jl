@model function smoother_model(y, x0, A, G, Q, R)

    x_prior ~ x0
    x_prev = x_prior  # initialize previous state with x_prior

    for i in 1:length(y)
        x[i] ~ MvNormal(mean=A * x_prev, cov=Q)
        y[i] ~ MvNormal(mean=G * x[i], cov=R)
        x_prev = x[i]
    end