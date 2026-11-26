return (
    title = "Reading the Inference Report",
    description = """
    The inference report of RxInfer, infer(...; report = true), which records what was approximated and how. The example reads the report for a logistic regression with an unknown prior precision under fallback = LocalLaplace(), under mean field and under belief propagation, and shows its diagnostics, its records as data and its timings.
    """,
    tags = ["universal inference", "laplace approximation", "inference report", "diagnostics", "mean field", "belief propagation", "rule fallback"],
    wide_output = true
)
