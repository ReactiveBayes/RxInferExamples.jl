return (
    title = "Integrating Neural Networks with Flux.jl and Lux.jl",
    description = """
    This example shows how to use RxInfer.jl together with Flux.jl and Lux.jl to incorporate neural networks into probabilistic models, and compares the two libraries side by side on the same problem.
    """,
    tags = ["advanced examples", "neural networks", "deep learning", "integration", "Flux.jl", "Lux.jl"],
    # Trains the same network twice, once per library - among the slowest examples to
    # build, so start it first. See `docs/src/how_build_works.md`.
    build_priority = true
)
