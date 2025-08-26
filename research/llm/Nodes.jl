module Nodes

using RxInfer

"""Definitions of LLM-related RxInfer nodes."""

struct LLMPrior end
@node LLMPrior Stochastic [ (b, aliases = [belief]), (c, aliases = [context]), (t, aliases = [task]) ]

struct LLMObservation end
@node LLMObservation Stochastic [ out, (b, aliases = [belief]), (t, aliases = [task]) ]

end # module

export LLMPrior, LLMObservation


