# This file was automatically generated from examples/Basic Examples/Hidden Markov Model/Hidden Markov Model.ipynb
# by notebooks_to_scripts.jl at 2025-03-14T05:52:02.112
#
# Source notebook: Hidden Markov Model.ipynb

using RxInfer, Random, BenchmarkTools, Distributions, LinearAlgebra, Plots

"""
    rand_vec(rng, distribution::Categorical)

This function returns a one-hot encoding of a random sample from a categorical distribution. The sample is drawn with the `rng` random number generator.
"""
function rand_vec(rng, distribution::Categorical) 
    k = ncategories(distribution)
    s = zeros(k)
    drawn_category = rand(rng, distribution)
    s[drawn_category] = 1.0
    return s
end

function generate_data(n_samples; seed = 42)
    
    rng = MersenneTwister(seed)
    
    # Transition probabilities 
    state_transition_matrix = [0.9 0.0 0.1;
                                                        0.0 0.9 0.1; 
                                                        0.05 0.05 0.9] 
    # Observation noise
    observation_distribution_matrix = [0.9 0.05 0.05;
                                                                         0.05 0.9 0.05;
                                                                         0.05 0.05 0.9] 
    # Initial state
    s_initial = [1.0, 0.0, 0.0] 
    
    states = Vector{Vector{Float64}}(undef, n_samples) # one-hot encoding of the states
    observations = Vector{Vector{Float64}}(undef, n_samples) # one-hot encoding of the observations
    
    s_prev = s_initial
    
    for t = 1:n_samples
        s_probvec = state_transition_matrix * s_prev
        states[t] = rand_vec(rng, Categorical(s_probvec ./ sum(s_probvec)))
        obs_probvec = observation_distribution_matrix * states[t]
        observations[t] = rand_vec(rng, Categorical(obs_probvec ./ sum(obs_probvec)))
        s_prev = states[t]
    end
    
    return observations, states
end

# Test data
N = 100
x_data, s_data = generate_data(N);

scatter(argmax.(s_data), leg=false, xlabel="Time",yticks= ([1,2,3],["Bedroom","Living room","Bathroom"]))

# Model specification
@model function hidden_markov_model(x)
    
    A ~ DirichletCollection(ones(3,3))
    B ~ DirichletCollection([ 10.0 1.0 1.0; 
                                            1.0 10.0 1.0; 
                                            1.0 1.0 10.0 ])
    
    s_0 ~ Categorical(fill(1.0 / 3.0, 3))
    
    s_prev = s_0
    
    for t in eachindex(x)
        s[t] ~ DiscreteTransition(s_prev, A) 
        x[t] ~ DiscreteTransition(s[t], B)
        s_prev = s[t]
    end
    
end

# Constraints specification
@constraints function hidden_markov_model_constraints()
    q(s_0, s, A, B) = q(s_0, s)q(A)q(B)
end

imarginals = @initialization begin
    q(A) = vague(DirichletCollection, (3, 3))
    q(B) = vague(DirichletCollection, (3, 3)) 
    q(s) = vague(Categorical, 3)
end

ireturnvars = (
    A = KeepLast(),
    B = KeepLast(),
    s = KeepLast()
)

result = infer(
    model         = hidden_markov_model(), 
    data          = (x = x_data,),
    constraints   = hidden_markov_model_constraints(),
    initialization = imarginals, 
    returnvars    = ireturnvars, 
    iterations    = 20, 
    free_energy   = true
);

println("Posterior Marginal for A:")
mean(result.posteriors[:A])

println("Posterior Marginal for B:")
mean(result.posteriors[:B])

p1 = scatter(argmax.(s_data), 
                        title="Inference results", 
                        label = "Real", 
                        ms = 6, 
                        legend=:right,
                        xlabel="Time" ,
                        yticks= ([1,2,3],["Bedroom","Living room","Bathroom"]),
                        size=(900,550)
                        )

p1 = scatter!(p1, argmax.(ReactiveMP.probvec.(result.posteriors[:s])),
                        label = "Inferred",
                        ms = 3
                        )

p2 = plot(result.free_energy, 
                    label="Free energy",
                    xlabel="Iteration Number"
                    )

plot(p1, p2, layout = @layout([ a; b ]))