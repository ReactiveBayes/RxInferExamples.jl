return (
    title = "Precision without Mean Field",
    description = """
    A Gaussian with unknown mean and precision, the example of Lukashchuk et al., without the mean-field factorisation: under fallback = NaturalGradientPropagation() the precision gets Gamma sites from belief propagation's messages, and one rule makes the messages towards the mean belief propagation's too; against the exact posterior on a grid and mean-field variational message passing, and how the divergences fall with the number of observations.
    """,
    tags = ["universal inference", "natural gradient", "variational inference", "mean field", "belief propagation", "gamma", "normal", "free energy"],
    wide_output = true
)
