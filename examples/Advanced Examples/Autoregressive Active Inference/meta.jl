return (
    title = "Autoregressive Active Inference",
    description = """
    This example demonstrates active inference for a differential-drive robot. The agent has no prior knowledge of the robot's dynamics: it learns them online with a Bayesian multivariate autoregressive model with exogenous inputs (MARX) and selects goal-directed actions by minimizing expected free energy through message passing with custom nodes and rules.
    """,
    tags = ["advanced examples", "active inference", "control", "robotics", "autoregressive model", "expected free energy"]
)
