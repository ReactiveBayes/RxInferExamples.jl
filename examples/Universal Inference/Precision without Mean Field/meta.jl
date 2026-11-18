return (
    title = "Precision without Mean Field",
    description = """
    A Gaussian with unknown mean and precision, the example of Lukashchuk et al., without the mean-field factorisation. Under fallback = NaturalGradientPropagation(), the precision gets Gamma sites from the belief propagation messages, and one extra rule makes the messages towards the mean belief propagation messages too. The result is compared with the exact posterior on a grid and with mean-field variational message passing, including how the divergences decrease with the number of observations.
    """,
    tags = ["universal inference", "natural gradient", "variational inference", "mean field", "belief propagation", "gamma", "normal", "free energy"],
    wide_output = true
)
