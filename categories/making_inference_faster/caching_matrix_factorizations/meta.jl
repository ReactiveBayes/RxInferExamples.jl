return (
    title = "Caching Matrix Factorizations",
    description = """
    The message passing rules factorize the matrices they receive anew at every node. This example explains `BayesBase.MatrixWithCachedInverse`, a matrix type that remembers its Cholesky factor and inverse, and shows the two places a user can put it without changing the model, the rules or `infer`: on a constant the model takes, where a dense sensor-noise covariance in a wide-sensor smoother is factorized at every time step, and inside a shared posterior through a form constraint, where the weights of a `softdot` regression are inverted by every node that reads them. Both are measured; both give two to three times faster inference with identical results.
    """,
    tags = ["making inference faster", "performance", "linear algebra", "form constraints", "kalman smoother", "regression"]
)
