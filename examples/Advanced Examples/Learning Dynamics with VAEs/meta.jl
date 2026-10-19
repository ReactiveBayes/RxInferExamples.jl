return (
    title = "Learning Dynamics with VAEs",
    description = """
    This example shows how to use RxInfer.jl automated inference to learn dynamics in latent space of a VAE.
    """,
    tags = ["advanced examples", "vae", "dynamics"],
    # AutoEncoderToolkit pins Flux 0.14, which the Reactant example's Functors 0.5 excludes
    own_environment = true
)
