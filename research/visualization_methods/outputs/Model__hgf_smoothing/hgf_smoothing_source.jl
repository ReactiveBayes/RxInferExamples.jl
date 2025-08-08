@model function hgf_smoothing(y, z_variance, y_variance)
    z_prev ~ Normal(mean = 0.0, variance = 5.0)
    x_prev ~ Normal(mean = 0.0, variance = 5.0)
    κ ~ Normal(mean = 1.5, variance = 1.0)
    ω ~ Normal(mean = 0.0, variance = 0.05)
    for i in eachindex(y)
        z[i] ~ Normal(mean = z_prev, variance = z_variance)
        x[i] ~ GCV(x_prev, z[i], κ, ω)
        y[i] ~ Normal(mean = x[i], variance = y_variance)
        z_prev = z[i]
        x_prev = x[i]
    end