# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Problem Specific/Probit Model/Probit Model.ipynb
# by notebooks_to_scripts.jl at 2025-04-29T06:39:07.632
#
# Source notebook: Probit Model.ipynb

using RxInfer, GraphPPL,StableRNGs, Random, Plots, Distributions
using StatsFuns: normcdf

function generate_data(nr_samples::Int64; seed = 123)
    
    rng = StableRNG(seed)
    
    # hyper parameters
    u = 0.1

    # allocate space for data
    data_x = zeros(nr_samples + 1)
    data_y = zeros(nr_samples)
    
    # initialize data
    data_x[1] = -2
    
    # generate data
    for k in eachindex(data_y)
        
        # calculate new x
        data_x[k+1] = data_x[k] + u + sqrt(0.01)*randn(rng)
        
        # calculate y
        data_y[k] = normcdf(data_x[k+1]) > rand(rng)
        
    end
    
    # return data
    return data_x, data_y
    
end;

n = 40

data_x, data_y = generate_data(n);

p = plot(xlabel = "t", ylabel = "x, y")
p = scatter!(p, data_y, label = "y")
p = plot!(p, data_x[2:end], label = "x")

@model function probit_model(y, prior_x)
    
    # specify uninformative prior
    x_prev ~ prior_x
    
    # create model 
    for k in eachindex(y)
        x[k] ~ Normal(mean = x_prev + 0.1, precision = 100)
        y[k] ~ Probit(x[k]) where {
            # Probit node by default uses RequireMessage pipeline with vague(NormalMeanPrecision) message as initial value for `in` edge
            # To change initial value user may specify it manually, like. Changes to the initial message may improve stability in some situations
            dependencies = RequireMessageFunctionalDependencies(in = NormalMeanPrecision(0.0, 0.01))
        }
        x_prev = x[k]
    end
    
end;

result = infer(
    model = probit_model(prior_x=Normal(0.0, 100.0)), 
    data  = (y = data_y, ), 
    iterations = 5, 
    returnvars = (x = KeepLast(),),
    free_energy  = true
)

mx = result.posteriors[:x]

p = plot(xlabel = "t", ylabel = "x, y", legend = :bottomright)
p = scatter!(p, data_y, label = "y")
p = plot!(p, data_x[2:end], label = "x", lw = 2)
p = plot!(mean.(mx)[2:end], ribbon = std.(mx)[2:end], fillalpha = 0.2, label="x (inferred mean)")

f = plot(xlabel = "t", ylabel = "BFE")
f = plot!(result.free_energy, label = "Bethe Free Energy")

plot(p, f, size = (800, 400))