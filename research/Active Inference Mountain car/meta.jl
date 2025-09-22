return (
    title = "Active Inference Mountain Car",
    description = """
    This example demonstrates RxInfer usage in the Active Inference setting for the mountain car control problem.

    The example is organized in a modular fashion with separate modules for:
    - Configuration management (config.jl)
    - Physics simulation (src/physics.jl)
    - World/environment management (src/world.jl)
    - Active inference agent (src/agent.jl)
    - Visualization and plotting (src/visualization.jl)
    - Main execution script (run.jl)
    - Test suite (test/runtests.jl)

    The active inference agent learns to navigate the mountain car environment by predicting future states
    and selecting actions that minimize expected free energy, demonstrating sophisticated control
    without traditional reinforcement learning.
    """,
    tags = ["advanced examples", "active inference", "control", "reinforcement learning", "modular design", "probabilistic programming"],
    execution = "julia run.jl",
    tests = "julia test/runtests.jl"
)
