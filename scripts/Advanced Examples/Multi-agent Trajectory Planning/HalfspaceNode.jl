module HalfspaceNode

using RxInfer
using LinearAlgebra

export Halfspace

# Define the probabilistic model for obstacles using halfspace constraints
struct Halfspace end

@node Halfspace Stochastic [out, a, σ2, γ]

# rule specification
@rule Halfspace(:out, Marginalisation) (q_a::Any, q_σ2::Any, q_γ::Any) = begin
    return NormalMeanVariance(mean(q_a) + mean(q_γ) * mean(q_σ2), mean(q_σ2))
end

@rule Halfspace(:σ2, Marginalisation) (q_out::Any, q_a::Any, q_γ::Any, ) = begin
    # `BayesBase.TerminalProdArgument` is used to ensure that the result of the posterior computation is equal to this value
    return BayesBase.TerminalProdArgument(PointMass( 1 / mean(q_γ) * sqrt(abs2(mean(q_out) - mean(q_a)) + var(q_out))))
end

end # module 