return (
    title = "Constrained Variables",
    description = """
    A zero-inflated Poisson model with Beta and Gamma priors under fallback = LocalLaplace(). The Laplace approximation is computed in logit and log coordinates, chosen from the supports of the priors, without rewriting or initialising the model, and is compared with the exact posterior on a grid. The example then fits a Beta posterior with ProjectedTo and a deterministic projection.
    """,
    tags = ["universal inference", "laplace approximation", "constrained variables", "zero-inflated poisson", "mean field", "rule fallback", "quadrature", "projection"],
    wide_output = true
)
