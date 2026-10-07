return (
    title = "Compiling RxInfer with Reactant",
    description = """
    Compiles one step of a Kalman filter written as an RxInfer model with Reactant.jl into a single XLA program, runs it over a stream of observations, and compares its speed with RxInfer for states of 2 to 64 dimensions.
    """,
    tags = ["advanced examples", "reactant", "compilation", "state space model", "kalman filter", "performance"]
)
