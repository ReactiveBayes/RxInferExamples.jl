return (
    title = "Stochastic Volatility",
    description = """
    The stochastic volatility model smoothed with no hand-written rules: a Laplace version of the Kalman smoother under LocalLaplace(), compared with a global Laplace approximation and an elliptical slice sampler and checked against the exact Kalman smoother on a linear-Gaussian model; then LaplacePropagation(), which reaches the global Laplace approximation, and ExpectationPropagation().
    """,
    tags = ["universal inference", "laplace approximation", "laplace propagation", "expectation propagation", "non-conjugate", "state space model", "stochastic volatility", "time series", "rule fallback"],
    wide_output = true
)
