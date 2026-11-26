return (
    title = "A Hierarchical Model",
    description = """
    The pump failure model from the BUGS manual, a Gamma-Poisson hierarchy whose shape parameter has no conjugate posterior. Every node has rules, and the fallback approximates the one product of messages that has no closed form: by a log-normal under LocalLaplace() and by a Gamma under NaturalGradientPropagation(). Both are compared with the exact posterior on a grid and with the best mean-field posterior.
    """,
    tags = ["universal inference", "laplace approximation", "natural gradient", "hierarchical model", "poisson", "gamma", "mean field", "free energy"],
    wide_output = true
)
