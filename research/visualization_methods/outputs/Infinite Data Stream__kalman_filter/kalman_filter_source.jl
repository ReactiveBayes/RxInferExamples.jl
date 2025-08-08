@model function kalman_filter(x_prev_mean, x_prev_var, τ_shape, τ_rate, y)
    x_prev ~ Normal(mean = x_prev_mean, variance = x_prev_var)
    τ ~ Gamma(shape = τ_shape, rate = τ_rate)

    # Random walk with fixed precision
    x_current ~ Normal(mean = x_prev, precision = 1.0)
    y ~ Normal(mean = x_current, precision = τ)
    
end