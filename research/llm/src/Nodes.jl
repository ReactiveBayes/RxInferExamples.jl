
using RxInfer

struct LLMPrior end
@node LLMPrior Stochastic [ (b, aliases = [belief]), (c, aliases = [context]), (t, aliases = [task]) ]

struct LLMObservation end
@node LLMObservation Stochastic [ out, (b, aliases = [belief]), (t, aliases = [task]) ]



