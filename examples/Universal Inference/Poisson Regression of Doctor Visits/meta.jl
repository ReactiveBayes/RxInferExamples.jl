return (
    title = "Poisson Regression (Doctor Visits)",
    description = """
    The doctor visits of 5190 adults from the Australian Health Survey, modelled by a Poisson regression with 13 coefficients in which exp is a plain function. fallback = NaturalGradientPropagation() fits it without a rule for exp, and the result is checked against self-normalised importance sampling from the joint Laplace approximation. The data are downloaded when the example runs.
    """,
    tags = ["universal inference", "natural gradient", "non-conjugate", "poisson regression", "regression", "real data", "importance sampling", "deterministic nodes"],
    wide_output = true
)
