# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Advanced Examples/Assessing People Skills/Assessing People Skills.ipynb
# by notebooks_to_scripts.jl at 2025-08-07T12:32:28.196
#
# Source notebook: Assessing People Skills.ipynb

using RxInfer, Random

# Create Score node
struct Score end

@node Score Stochastic [out, in]

# Adding update rule for the Score node
@rule Score(:in, Marginalisation) (q_out::PointMass,) = begin
    return Bernoulli(mean(q_out))
end

# GraphPPL.jl exports the `@model` macro for model specification
# It accepts a regular Julia function and builds an FFG under the hood
@model function skill_model(r)

    local s
    # Priors
    for i in eachindex(r)
        s[i] ~ Bernoulli(0.5)
    end

    # Domain logic
    t[1] ~ ¬s[1]
    t[2] ~ t[1] || s[2]
    t[3] ~ t[2] && s[3]

    # Results
    for i in eachindex(r)
        r[i] ~ Score(t[i])
    end
end

test_results = [0.1, 0.1, 0.1]
inference_result = infer(
    model=skill_model(),
    data=(r=test_results,)
)

# Inspect the results
map(params, inference_result.posteriors[:s])