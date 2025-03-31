# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Problem Specific/Simple Nonlinear Node/Simple Nonlinear Node.ipynb
# by notebooks_to_scripts.jl at 2025-03-31T09:50:41.430
#
# Source notebook: Simple Nonlinear Node.ipynb

using RxInfer, Random, StableRNGs

struct NonlinearNode end # Dummy structure just to make Julia happy

struct NonlinearMeta{R, F}
    rng      :: R
    fn       :: F   # Nonlinear function, we assume 1 float input - 1 float output
    nsamples :: Int # Number of samples used in approximation
end

@node NonlinearNode Deterministic [ out, in ]

# Rule for outbound message on `out` edge given inbound message on `in` edge
@rule NonlinearNode(:out, Marginalisation) (m_in::NormalMeanVariance, meta::NonlinearMeta) = begin 
    samples = rand(meta.rng, m_in, meta.nsamples)
    return SampleList(map(meta.fn, samples))
end

# Rule for outbound message on `in` edge given inbound message on `out` edge
@rule NonlinearNode(:in, Marginalisation) (m_out::Gamma, meta::NonlinearMeta) = begin     
    return ContinuousUnivariateLogPdf((x) -> logpdf(m_out, meta.fn(x)))
end

@model function nonlinear_estimation(y, θ_μ, m_μ, θ_σ, m_σ)
    
    # define a distribution for the two variables
    θ ~ Normal(mean = θ_μ, variance = θ_σ)
    m ~ Normal(mean = m_μ, variance = m_σ)

    # define a nonlinear node
    w ~ NonlinearNode(θ)

    # We consider the outcome to be normally distributed
    for i in eachindex(y)
        y[i] ~ Normal(mean = m, precision = w)
    end
    
end

@constraints function nconstsraints(nsamples)
    q(θ) :: SampleListFormConstraint(nsamples, LeftProposal())
    q(w) :: SampleListFormConstraint(nsamples, RightProposal())
    
    q(θ, w, m) = q(θ)q(m)q(w)
end

@meta function nmeta(fn, nsamples)
    NonlinearNode(θ, w) -> NonlinearMeta(StableRNG(123), fn, nsamples)
end

@initialization function ninit()
    q(m) = vague(NormalMeanPrecision)
    q(w) = vague(Gamma)
end

nonlinear_fn(x) = abs(exp(x) * sin(x))

seed = 123
rng  = StableRNG(seed)

niters   = 15 # Number of VMP iterations
nsamples = 5_000 # Number of samples in approximation

n = 500 # Number of IID samples
μ = -10.0
θ = -1.0
w = nonlinear_fn(θ)

data = rand(rng, NormalMeanPrecision(μ, w), n);

result = infer(
    model = nonlinear_estimation(θ_μ = 0.0, m_μ = 0.0, θ_σ=100.0, m_σ=1.0),
    meta =  nmeta(nonlinear_fn, nsamples),
    constraints = nconstsraints(nsamples),
    data = (y = data, ), 
    initialization = ninit(),
    returnvars = (θ = KeepLast(), ),
    iterations = niters,  
    showprogress = true
)

θposterior = result.posteriors[:θ]

using Plots, StatsPlots

estimated = Normal(mean_std(θposterior)...)

plot(estimated, title="Posterior for θ", label = "Estimated", legend = :bottomright, fill = true, fillopacity = 0.2, xlim = (-3, 3), ylim = (0, 2))
vline!([ θ ], label = "Real value of θ")