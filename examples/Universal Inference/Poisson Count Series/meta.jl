return (
    title = "Poisson Count Series",
    description = """
    A latent autoregressive log-rate observed through Poisson counts, smoothed by belief propagation with a rule-free observation node under LocalLaplace(), CVI(), ExpectationPropagation() and LaplacePropagation(), against the exact smoother of a fine grid and its evidence; then the transition's precision learned under mean field.
    """,
    tags = ["universal inference", "laplace approximation", "cvi", "expectation propagation", "laplace propagation", "state space model", "poisson", "time series", "free energy"],
    wide_output = true
)
