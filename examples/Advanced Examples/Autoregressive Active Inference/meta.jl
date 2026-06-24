return (
    title = "Autoregressive Active Inference",
    description = """
    This example demonstrates active inference for a 2D thermal-coupled positioning stage. The agent has no prior knowledge of the stage's dynamics: it learns the mass, damping, and thermal expansion online with a Bayesian multivariate autoregressive model with exogenous inputs (MARX) and selects goal-directed actions by minimizing expected free energy through message passing with custom nodes and rules.
    """,
    tags = ["advanced examples", "active inference", "control", "thermal stage", "online learning", "autoregressive model", "expected free energy"]
)
