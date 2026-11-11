return (
    title = "Gaussian Mixture without Conjugacy",
    description = """
    A Gaussian mixture with Gaussian priors on its log-precisions, which have no conjugate rules, under fallback = LocalLaplace(): Laplace approximations for the means and log-precisions, exact beliefs for the discrete assignments and variational messages computed as expectations, and its free energy, compared with the conjugate NormalMixture solution and with a plug-in rule fallback.
    """,
    tags = ["universal inference", "laplace approximation", "mixture model", "discrete variables", "mean field", "rule fallback", "quadrature", "free energy"],
    wide_output = true
)
