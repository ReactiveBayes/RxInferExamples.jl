return (
    title = "Gaussian Mixture without Conjugacy",
    description = """
    A Gaussian mixture with Gaussian priors on its log-precisions, which have no conjugate rules, under fallback = LocalLaplace(). The means and log-precisions get Laplace approximations, the discrete assignments get exact beliefs, and the variational messages are computed as expectations, which also gives the free energy. The result is compared with the conjugate NormalMixture solution and with a plug-in rule fallback.
    """,
    tags = ["universal inference", "laplace approximation", "mixture model", "discrete variables", "mean field", "rule fallback", "quadrature", "free energy"],
    wide_output = true
)
