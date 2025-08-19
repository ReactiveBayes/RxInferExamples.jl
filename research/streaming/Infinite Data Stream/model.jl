module InfiniteDataStreamModel

using RxInfer

export kalman_filter, filter_constraints

@model function kalman_filter(x_prev_mean, x_prev_var, τ_shape, τ_rate, y)
    x_prev ~ Normal(mean = x_prev_mean, variance = x_prev_var)
    τ ~ Gamma(shape = τ_shape, rate = τ_rate)

    x_current ~ Normal(mean = x_prev, precision = 1.0)
    y ~ Normal(mean = x_current, precision = τ)
end

@constraints function filter_constraints()
    q(x_prev, x_current, τ) = q(x_prev, x_current)q(τ)
end

end # module

