return (
    title = "Poisson Count Series",
    description = """
    A latent autoregressive log-rate observed through Poisson counts. The observation node has no rules, and the series is smoothed by belief propagation under LocalLaplace(), ExpectationPropagation(), LaplacePropagation() and NaturalGradientPropagation(), and compared with the exact smoother on a fine grid and its evidence. The precision of the transition is then learned under mean field.
    """,
    tags = ["universal inference", "laplace approximation", "expectation propagation", "laplace propagation", "natural gradient", "state space model", "poisson", "time series", "free energy"],
    wide_output = true
)
