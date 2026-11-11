return (
    title = "Constrained Variables",
    description = """
    A zero-inflated Poisson model with Beta and Gamma priors under fallback = LocalLaplace(), where the Laplace approximation is computed automatically in logit and log coordinates chosen from the supports of the priors, without rewriting the model or initialising it, compared with the exact posterior on a grid; then a Beta posterior through ProjectedTo and a deterministic projection.
    """,
    tags = ["universal inference", "laplace approximation", "constrained variables", "zero-inflated poisson", "mean field", "rule fallback", "quadrature", "projection"],
    wide_output = true
)
