return (
    title = "Tracking with a Squared-Distance Sensor",
    description = """
    A random walk observed through its squared distance to a reference point, with unknown precisions of the motion and of the sensor. fallback = NaturalGradientPropagation() runs the model without a rule for the squared distance, a plain Julia function. Under mean field both precisions are conjugate. With the sensor precision kept together with the path, its messages are belief propagation messages and its posterior is a Gamma formed by natural-gradient sites. Both cases are compared with the exact posterior on a grid.
    """,
    tags = ["universal inference", "natural gradient", "non-conjugate", "variational inference", "state space model", "nonlinear observation", "gamma", "mean field", "free energy"],
    wide_output = true
)
