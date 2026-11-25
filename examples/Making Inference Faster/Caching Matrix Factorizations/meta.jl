return (
    title = "Caching Matrix Factorizations",
    description = """
    RxInfer's message passing rules factorize the matrices they receive at every node. This example describes `BayesBase.MatrixWithCachedInverse`, a matrix type that stores its Cholesky factor and inverse, and shows two places to use it without changing the model, the rules or `infer`. As a constant passed to the model, it stops a smoother with many sensors from factorizing the dense sensor-noise covariance at every time step. Inside a shared posterior, through a form constraint, it stops every node of a `softdot` regression from inverting the precision of the weights again. Inference is two to three times faster in the first case and about twice as fast in the second, with the same results.
    """,
    tags = ["making inference faster", "performance", "linear algebra", "form constraints", "kalman smoother", "regression"]
)
