return (
    title = "Laplace Approximation: Constrained Variables",
    description = """
    A zero-inflated Poisson model with Beta and Gamma priors through fallback = LaplaceApproximation(), where the Laplace approximation is computed automatically in logit and log coordinates chosen from the supports of the priors, without rewriting the model or initialising it, and compared with the exact posterior on a grid.
    """,
    tags = ["universal inference", "laplace approximation", "constrained variables", "zero-inflated poisson", "mean field", "rule fallback", "quadrature"]
)
