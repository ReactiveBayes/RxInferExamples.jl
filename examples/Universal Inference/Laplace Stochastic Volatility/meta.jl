return (
    title = "Laplace and EP: Stochastic Volatility",
    description = """
    A Laplace version of the Kalman smoother for the stochastic volatility model through fallback = LaplaceApproximation(), compared with a global Laplace approximation and an elliptical slice sampler; Laplace propagation, which reaches the global Laplace approximation, and expectation propagation; and checked against the exact Kalman smoother and its free energy on a linear-Gaussian model.
    """,
    tags = ["universal inference", "laplace approximation", "non-conjugate", "state space model", "stochastic volatility", "time series", "rule fallback", "laplace propagation", "expectation propagation"],
    wide_output = true
)
