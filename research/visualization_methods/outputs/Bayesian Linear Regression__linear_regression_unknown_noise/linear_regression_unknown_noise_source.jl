@model function linear_regression_unknown_noise(x, y)
    a ~ Normal(mean = 0.0, variance = 1.0)
    b ~ Normal(mean = 0.0, variance = 100.0)
    s ~ InverseGamma(1.0, 1.0)
    y .~ Normal(mean = a .* x .+ b, variance = s)
end