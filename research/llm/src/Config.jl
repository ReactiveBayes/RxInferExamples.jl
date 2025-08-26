using JSON

# Configuration constant for RxInferLLM (included into RxInferLLM module)
const CONFIG = (
    # secret_key should come from the environment. Do NOT hardcode secrets in source.
    secret_key = get(ENV, "OPENAI_KEY", ""),
    HAS_OPENAI_KEY = haskey(ENV, "OPENAI_KEY") && !isempty(ENV["OPENAI_KEY"]),
    llm_model = "gpt-4o-mini-2024-07-18",
    n_iterations = 5,
    observations = [
        "RxInfer.jl is confusing and frustrating to use. I wouldn't recommend it.",
        "RxInfer.jl made my Bayesian modeling workflow much easier and more efficient!",
        "Absolutely love RxInfer.jl! It's revolutionized my approach to probabilistic programming.",
        "I gave RxInfer.jl a try, but it just doesn't work for my needs at all.",
        "I prefer apples over oranges."
    ],
    prior_task = """
Provide a distribution of the statement.
- **Mean**: Most likely satisfaction score (0-10 scale)
- **Variance**: Uncertainty in your interpretation
    - Low variance (2.0-4.0): Very clear sentiment
    - Medium variance (4.1-6.0): Some ambiguity
    - High variance (6.0-10.0): Unclear or mixed signals
""",
    likelihood_task = """
Evaluation of sentiment about RxInfer.jl and provide satisfaction score distribution.
If expression is not related to RxInfer.jl, return distribution with mean 5 and high variance of 100.
- **Mean**: Most likely satisfaction score (0-10 scale)
- **Variance**: Uncertainty in interpretation
    - Low variance (0.1-1.0): Very clear sentiment, confident interpretation
    - Medium variance (1.1-5.0): Some ambiguity
    - High variance (5.1-10.0): Unclear/mixed signals, or not related to RxInfer.jl
""",
)

# Runtime accessors (read ENV at runtime to avoid precompilation caching)
openai_key() = get(ENV, "OPENAI_KEY", "")
has_openai_key() = haskey(ENV, "OPENAI_KEY") && !isempty(ENV["OPENAI_KEY"])



