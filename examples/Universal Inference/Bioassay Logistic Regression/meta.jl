return (
    title = "Bioassay Logistic Regression",
    description = """
    The bioassay logistic regression from Bayesian Data Analysis, fitted without hand-written rules. With fallback = LocalLaplace() the posterior is a Laplace approximation, checked against the exact posterior and evidence on a grid. With fallback = NaturalGradientPropagation() it is the variational Gaussian found by natural-gradient message passing. The example also writes the model with deterministic nodes and fits a Poisson regression with the same keyword.
    """,
    tags = ["universal inference", "laplace approximation", "natural gradient", "non-conjugate", "logistic regression", "poisson regression", "rule fallback", "deterministic nodes", "free energy"],
    wide_output = true
)
