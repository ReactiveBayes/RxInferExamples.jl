@model function hgf(y, κ, ω, z_variance, y_variance, z_prev_mean, z_prev_var, x_prev_mean, x_prev_var)
    z_prev ~ Normal(mean = z_prev_mean, variance = z_prev_var)
    x_prev ~ Normal(mean = x_prev_mean, variance = x_prev_var)
    z_next ~ Normal(mean = z_prev, variance = z_variance)
    x_next ~ GCV(x_prev, z_next, κ, ω)
    y ~ Normal(mean = x_next, variance = y_variance)
end