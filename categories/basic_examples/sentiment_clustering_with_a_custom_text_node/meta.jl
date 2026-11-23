return (
    title = "Sentiment Clustering with a Custom Text Node",
    description = """
    Clusters short product reviews by sentiment with a Gaussian mixture, observing the raw text through a custom factor node whose rule turns a `PointMass{String}` into a Gaussian likelihood for a latent satisfaction score.
    """,
    tags = ["basic examples", "custom node", "custom rules", "text data", "mixture model", "variational inference", "free energy"]
)
