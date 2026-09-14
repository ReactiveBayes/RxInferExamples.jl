return (
    title = "Solving Linear Systems with Message Passing",
    description = """
    This tutorial turns a linear system into a Gaussian factor graph, solves it with the `GaussianCoupling` node, and visualizes convergence and the difference between tree and loopy uncertainty estimates. It then puts the same graph on a heated grid whose boundary sources drift over time: a state-space submodel with a learned Wishart coupling is plugged into the grid, and the temperature field is recovered from source sensors that occasionally go offline.
    """,
    tags = ["advanced examples", "linear algebra", "message passing", "gaussian belief propagation", "nested models", "missing data", "smoothing"]
)
