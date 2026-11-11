return (
    title = "Laplace, CVI and EP: A Pendulum Filter and Smoother",
    description = """
    Särkkä's noisy pendulum filtered online with its nonlinear transition and observation written as plain functions in the model, computed by the fallback with no approximation method per node, under LocalLaplace() and CVI(), against the unscented transform chosen by hand and a particle filter; then smoothed as a whole by expectation propagation and Laplace propagation, against a particle smoother.
    """,
    tags = ["universal inference", "laplace approximation", "cvi", "nonlinear filtering", "state space model", "deterministic nodes", "streaming", "expectation propagation", "smoothing"],
    wide_output = true
)
