return (
    title = "Pendulum Filter and Smoother",
    description = """
    Särkkä's noisy pendulum with its nonlinear transition and observation written as plain functions in the model, computed by the fallback with no approximation method per node: filtered online under LocalLaplace() and CVI(), against the unscented transform chosen by hand and a particle filter; then smoothed as a whole under ExpectationPropagation() and LaplacePropagation(), against a particle smoother.
    """,
    tags = ["universal inference", "laplace approximation", "cvi", "expectation propagation", "laplace propagation", "nonlinear filtering", "smoothing", "state space model", "deterministic nodes", "streaming"],
    wide_output = true
)
