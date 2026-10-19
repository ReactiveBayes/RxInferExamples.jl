return (
    title = "Compiling RxInfer with Reactant",
    description = """
    Compiles one step of a Kalman filter written as an RxInfer model with Reactant.jl into a single XLA program: why the model with RxInfer's standard nodes does not compile yet, how to convert it with a distribution, a node and rules written for traced numbers, and how the compiled step compares with RxInfer for states of 2 to 64 dimensions.
    """,
    tags = ["advanced examples", "reactant", "compilation", "state space model", "kalman filter", "performance", "custom nodes"],
    # Reactant needs JSON 1.4 or later, which Weave, the build's, excludes
    own_environment = true
)
