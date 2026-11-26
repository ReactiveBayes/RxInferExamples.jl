return (
    title = "Pendulum Filter and Smoother",
    description = """
    Särkkä's noisy pendulum, with the nonlinear transition and observation written as plain functions in the model and handled by the fallback, without an approximation method per node. The pendulum is filtered online under LocalLaplace() and NaturalGradientPropagation() and compared with the unscented transform chosen by hand and with a particle filter. It is then smoothed as a whole under ExpectationPropagation() and LaplacePropagation() and compared with a particle smoother.
    """,
    tags = ["universal inference", "laplace approximation", "natural gradient", "expectation propagation", "laplace propagation", "nonlinear filtering", "smoothing", "state space model", "deterministic nodes", "streaming"],
    wide_output = true
)
