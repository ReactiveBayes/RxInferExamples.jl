# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Advanced Examples/Conjugate-Computational Variational Message Passing/Conjugate-Computational Variational Message Passing.ipynb
# by notebooks_to_scripts.jl at 2025-08-07T12:32:28.205
#
# Source notebook: Conjugate-Computational Variational Message Passing.ipynb

using RxInfer, Random, LinearAlgebra, Plots, Optimisers, StableRNGs, SpecialFunctions

# data generating process
nr_observations = 50
reference_point = 53
hidden_location = collect(1:nr_observations) + rand(StableRNG(124), NormalMeanVariance(0.0, sqrt(5)), nr_observations)
measurements = (hidden_location .- reference_point).^2 + rand(MersenneTwister(124), NormalMeanVariance(0.0, 5), nr_observations);

# plot hidden location and reference frame
p1 = plot(1:nr_observations, hidden_location, linewidth=3, legend=:topleft, label="hidden location")
hline!([reference_point], linewidth=3, label="reference point")
xlabel!("time [sec]"), ylabel!("location [cm]")

# plot measurements
p2 = scatter(1:nr_observations, measurements, linewidth=3, label="measurements")
xlabel!("time [sec]"), ylabel!("squared distance [cm2]")

plot(p1, p2, size=(1200, 500))

function compute_squared_distance(z)
    (z - reference_point)^2
end;

@model function measurement_model(y)

    # set priors on precision parameters
    τ ~ Gamma(shape = 0.01, rate = 0.01)
    γ ~ Gamma(shape = 0.01, rate = 0.01)
    
    # specify estimate of initial location
    z[1] ~ Normal(mean = 0, precision = τ)
    y[1] ~ Normal(mean = compute_squared_distance(z[1]), precision = γ)

    # loop over observations
    for t in 2:length(y)

        # specify state transition model
        z[t] ~ Normal(mean = z[t-1] + 1, precision = τ)

        # specify non-linear observation model
        y[t] ~ Normal(mean = compute_squared_distance(z[t]), precision = γ)
        
    end

end

@meta function measurement_meta(rng, nr_samples, nr_iterations, optimizer)
    compute_squared_distance() -> CVI(rng, nr_samples, nr_iterations, optimizer)
end;

@constraints function measurement_constraints()
    q(z, τ, γ) = q(z)q(τ)q(γ)
end;

initialization = @initialization begin
    μ(z) = NormalMeanVariance(0, 5)
    q(z) = NormalMeanVariance(0, 5)
    q(τ) = GammaShapeRate(1e-12, 1e-3)
    q(γ) = GammaShapeRate(1e-12, 1e-3)
end

results = infer(
    model = measurement_model(),
    data = (y = measurements,),
    iterations = 50,
    free_energy = true,
    returnvars = (z = KeepLast(),),
    constraints = measurement_constraints(),
    meta = measurement_meta(StableRNG(42), 1000, 1000, Optimisers.Descent(0.001)),
    initialization = initialization
)

# plot estimates for location
p1 = plot(collect(1:nr_observations), hidden_location, label = "hidden location", legend=:topleft, linewidth=3, color = :red)
plot!(map(mean, results.posteriors[:z]), label = "estimated location (±2σ)", ribbon = map(x -> 2*std(x), results.posteriors[:z]), fillalpha=0.5, linewidth=3, color = :orange)
xlabel!("time [sec]"), ylabel!("location [cm]")

# plot Bethe free energy
p2 = plot(results.free_energy, linewidth=3, label = "")
xlabel!("iteration"), ylabel!("Bethe free energy [nats]")

plot(p1, p2, size = (1200, 500))

struct CustomDescent 
    learning_rate::Float64
end

# Must return an optimizer and its initial state
function ReactiveMP.cvi_setup(opt::CustomDescent, q)
     return (opt, nothing)
end

# Must return an updated (opt, state) and an updated λ (can use new_λ for inplace operation)
function ReactiveMP.cvi_update!(opt_and_state::Tuple{CustomDescent, Nothing}, new_λ, λ, ∇)
    opt, _ = opt_and_state
    λ̂ = vec(λ) - (opt.learning_rate .* vec(∇))
    copyto!(new_λ, λ̂)
    return opt_and_state, new_λ
end

# generate data
y = rand(StableRNG(123), NormalMeanVariance(19^2, 10), 1000)
histogram(y)

# specify non-linearity
f(x) = x ^ 2

# specify model
@model function normal_square_model(y)
    # describe prior on latent state, we set an arbitrary prior 
    # in a positive domain
    x ~ Normal(mean = 5, precision = 1e-3)
    # transform latent state
    mean := f(x)
    # observation model
    y .~ Normal(mean = mean, precision = 0.1)
end

# specify meta
@meta function normal_square_meta(rng, nr_samples, nr_iterations, optimizer)
    f() ->  CVI(rng, nr_samples, nr_iterations, optimizer)
end

res = infer(
    model = normal_square_model(),
    data = (y = y,),
    iterations = 5,
    free_energy = true,
    meta = normal_square_meta(StableRNG(123), 1000, 1000, CustomDescent(0.001)),
    free_energy_diagnostics = nothing
)

mean(res.posteriors[:x][end])

p1 = plot(mean.(res.posteriors[:x]), ribbon = 3std.(res.posteriors[:x]), label = "Posterior estimation", ylim = (0, 40))
p2 = plot(res.free_energy, label = "Bethe Free Energy")

plot(p1, p2, layout = @layout([ a b ]))