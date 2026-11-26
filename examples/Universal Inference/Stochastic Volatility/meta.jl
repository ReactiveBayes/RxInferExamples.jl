return (
    title = "Stochastic Volatility",
    description = """
    The stochastic volatility model smoothed without hand-written rules. Under LocalLaplace() the result is a Laplace version of the Kalman smoother, which is compared with a global Laplace approximation and an elliptical slice sampler and checked against the exact Kalman smoother on a linear-Gaussian model. The example then runs LaplacePropagation(), which reaches the global Laplace approximation, and ExpectationPropagation().
    """,
    tags = ["universal inference", "laplace approximation", "laplace propagation", "expectation propagation", "non-conjugate", "state space model", "stochastic volatility", "time series", "rule fallback"],
    wide_output = true
)
