### Model and Constraints

Defined in `model.jl`:
```julia
@model function kalman_filter(x_prev_mean, x_prev_var, τ_shape, τ_rate, y)
    x_prev ~ Normal(mean = x_prev_mean, variance = x_prev_var)
    τ ~ Gamma(shape = τ_shape, rate = τ_rate)
    x_current ~ Normal(mean = x_prev, precision = 1.0)
    y ~ Normal(mean = x_current, precision = τ)
end

@constraints function filter_constraints()
    q(x_prev, x_current, τ) = q(x_prev, x_current)q(τ)
end
```

Notes:
- Maintains the same structure as the notebook: a random-walk latent `x_current` with fixed transition precision and an unknown observation precision `τ`.
- The mean-field factorization between `(x_prev, x_current)` and `τ` matches the notebook’s variational family.

