# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Problem Specific/Invertible Neural Network Tutorial/Invertible Neural Network Tutorial.ipynb
# by notebooks_to_scripts.jl at 2025-03-31T09:50:41.331
#
# Source notebook: Invertible Neural Network Tutorial.ipynb

using RxInfer
using Random
using StableRNGs

using ReactiveMP        # ReactiveMP is included in RxInfer, but we explicitly use some of its functionality
using LinearAlgebra     # only used for some matrix specifics
using Plots             # only used for visualisation
using Distributions     # only used for sampling from multivariate distributions
using Optim             # only used for parameter optimisation

model = FlowModel(2,
    (
        AdditiveCouplingLayer(PlanarFlow()),
        AdditiveCouplingLayer(PlanarFlow(); permute=false)
    )
);

model = FlowModel(
    (
        InputLayer(2),
        AdditiveCouplingLayer(PlanarFlow()),
        AdditiveCouplingLayer(PlanarFlow(); permute=false)
    )
);

model = FlowModel(
    (
        InputLayer(2),
        AdditiveCouplingLayer(PlanarFlow(); permute=false),
        PermutationLayer(PermutationMatrix(2)),
        AdditiveCouplingLayer(PlanarFlow(); permute=false)
    )
);

model

compiled_model = compile(model)

compiled_model = compile(model, randn(StableRNG(321), nr_params(model)))

function generate_data(nr_samples::Int64, model::CompiledFlowModel; seed = 123)

    rng = StableRNG(seed)
    
    # specify latent sampling distribution
    dist = MvNormal([1.5, 0.5], I)

    # sample from the latent distribution
    x = rand(rng, dist, nr_samples)

    # transform data
    y = zeros(Float64, size(x))
    for k = 1:nr_samples
        y[:,k] .= ReactiveMP.forward(model, x[:,k])
    end

    # return data
    return y, x

end;

# generate data
y, x = generate_data(1000, compiled_model)

# plot generated data
p1 = scatter(x[1,:], x[2,:], alpha=0.3, title="Original data", size=(800,400))
p2 = scatter(y[1,:], y[2,:], alpha=0.3, title="Transformed data", size=(800,400))
plot(p1, p2, legend = false)

@model function invertible_neural_network(y)

    # specify prior
    z_μ ~ MvNormalMeanCovariance(zeros(2), huge*diagm(ones(2)))
    z_Λ ~ Wishart(2.0, tiny*diagm(ones(2)))

    # specify observations
    for k in eachindex(y)

        # specify latent state
        x[k] ~ MvNormalMeanPrecision(z_μ, z_Λ)

        # specify transformed latent value
        y_lat[k] ~ Flow(x[k])

        # specify observations
        y[k] ~ MvNormalMeanCovariance(y_lat[k], tiny*diagm(ones(2)))

    end

end;

observations = [y[:,k] for k=1:size(y,2)]

fmodel         = invertible_neural_network()
data           = (y = observations, )
initialization = @initialization begin 
    q(z_μ) = MvNormalMeanCovariance(zeros(2), huge*diagm(ones(2)))
    q(z_Λ) = Wishart(2.0, tiny*diagm(ones(2)))
end
returnvars     = (z_μ = KeepLast(), z_Λ = KeepLast(), x = KeepLast(), y_lat = KeepLast())

constraints = @constraints begin
    q(z_μ, x, z_Λ) = q(z_μ)q(z_Λ)q(x)
end

@meta function fmeta(model)
    compiled_model = compile(model, randn(StableRNG(321), nr_params(model)))
    Flow(y_lat, x) -> FlowMeta(compiled_model) # defaults to FlowMeta(compiled_model; approximation=Linearization()). 
                                               # other approximation methods can be e.g. FlowMeta(compiled_model; approximation=Unscented(input_dim))
end

# First execution is slow due to Julia's initial compilation 
result = infer(
    model          = fmodel, 
    data           = data,
    constraints    = constraints,
    meta           = fmeta(model),
    initialization = initialization,
    returnvars     = returnvars,
    free_energy    = true,
    iterations     = 10, 
    showprogress   = false
)

fe_flow = result.free_energy
zμ_flow = result.posteriors[:z_μ]
zΛ_flow = result.posteriors[:z_Λ]
x_flow  = result.posteriors[:x]
y_flow  = result.posteriors[:y_lat];

plot(1:10, fe_flow/size(y,2), xlabel="iteration", ylabel="normalized variational free energy [nats/sample]", legend=false)

# pick a random observation
id = rand(StableRNG(321), 1:size(y,2))
rand_observation = MvNormal(y[:,id], 5e-1*diagm(ones(2)))
warped_observation = MvNormal(ReactiveMP.backward(compiled_model, y[:,id]), ReactiveMP.inv_jacobian(compiled_model, y[:,id])*5e-1*diagm(ones(2))*ReactiveMP.inv_jacobian(compiled_model, y[:,id])');

p1 = scatter(x[1,:], x[2,:], alpha=0.1, title="Latent distribution", size=(1200,500), label="generated data")
contour!(-5:0.1:5, -5:0.1:5, (x, y) -> pdf(MvNormal([1.5, 0.5], I), [x, y]), c=:viridis, colorbar=false, linewidth=2)
scatter!([mean(zμ_flow)[1]], [mean(zμ_flow)[2]], color="red", markershape=:x, markersize=5, label="inferred mean")
contour!(-5:0.01:5, -5:0.01:5, (x, y) -> pdf(warped_observation, [x, y]), colors="red", levels=1, linewidth=2, colorbar=false)
scatter!([mean(warped_observation)[1]], [mean(warped_observation)[2]], color="red", label="transformed noisy observation")
p2 = scatter(y[1,:], y[2,:], alpha=0.1, label="generated data")
scatter!([ReactiveMP.forward(compiled_model, mean(zμ_flow))[1]], [ReactiveMP.forward(compiled_model, mean(zμ_flow))[2]], color="red", marker=:x, label="inferred mean")
contour!(-10:0.1:10, -10:0.1:10, (x, y) -> pdf(MvNormal([1.5, 0.5], I), ReactiveMP.backward(compiled_model, [x, y])), c=:viridis, colorbar=false, linewidth=2)
contour!(-10:0.1:10, -10:0.1:10, (x, y) -> pdf(rand_observation, [x, y]), colors="red", levels=1, linewidth=2, label="random noisy observation", colorba=false)
scatter!([mean(rand_observation)[1]], [mean(rand_observation)[2]], color="red", label="random noisy observation")
plot(p1, p2, legend = true)

function generate_data(nr_samples::Int64; seed = 123)
    
    rng = StableRNG(seed)

    # sample weights
    w = rand(rng, nr_samples, 2)

    # sample appraisal
    y = zeros(Float64, nr_samples)
    for k = 1:nr_samples
        y[k] = 1.0*(w[k,1] > 0.5)*(w[k,2] < 0.5)
    end

    # return data
    return y, w

end;

data_y, data_x = generate_data(50);
scatter(data_x[:,1], data_x[:,2], marker_z=data_y, xlabel="w1", ylabel="w2", colorbar=false, legend=false)

# specify flow model
model = FlowModel(2,
    (
        AdditiveCouplingLayer(PlanarFlow()), # defaults to AdditiveCouplingLayer(PlanarFlow(); permute=true)
        AdditiveCouplingLayer(PlanarFlow()),
        AdditiveCouplingLayer(PlanarFlow()),
        AdditiveCouplingLayer(PlanarFlow(); permute=false)
    )
);

@model function invertible_neural_network_classifier(x, y)

    # specify observations
    for k in eachindex(x)

        # specify latent state
        x_lat[k] ~ MvNormalMeanPrecision(x[k], 1e3*diagm(ones(2)))

        # specify transformed latent value
        y_lat1[k] ~ Flow(x_lat[k])
        y_lat2[k] ~ dot(y_lat1[k], [1, 1])

        # specify observations
        y[k] ~ Probit(y_lat2[k]) # default: where { pipeline = RequireMessage(in = NormalMeanPrecision(0, 1.0)) }

    end

end;

fcmodel       = invertible_neural_network_classifier()
data          = (y = data_y, x = [data_x[k,:] for k=1:size(data_x,1)], )

@meta function fmeta(model, params)
    compiled_model = compile(model, params)
    Flow(y_lat1, x_lat) -> FlowMeta(compiled_model)
end

function f(params)
    Random.seed!(123) # Flow uses random permutation matrices, which is not good for the optimisation procedure
    result = infer(
        model                   = fcmodel, 
        data                    = data,
        meta                    = fmeta(model, params),
        free_energy             = true,
        free_energy_diagnostics = nothing, # Free Energy can be set to NaN due to optimization procedure
        iterations              = 10, 
        showprogress            = false
    );
    
    result.free_energy[end]
end;

res = optimize(f, randn(StableRNG(42), nr_params(model)), GradientDescent(), Optim.Options(f_tol = 1e-3, store_trace = true, show_trace = true, show_every = 100), autodiff=:forward)

params = Optim.minimizer(res)
inferred_model = compile(model, params)
trans_data_x_1 = hcat(map((x) -> ReactiveMP.forward(inferred_model, x), [data_x[k,:] for k=1:size(data_x,1)])...)'
trans_data_x_2 = map((x) -> dot([1, 1], x), [trans_data_x_1[k,:] for k=1:size(data_x,1)])
trans_data_x_2_split = [trans_data_x_2[data_y .== 1.0], trans_data_x_2[data_y .== 0.0]]
p1 = scatter(data_x[:,1], data_x[:,2], marker_z = data_y, size=(1200,400), c=:viridis, colorbar=false, title="original data")
p2 = scatter(trans_data_x_1[:,1], trans_data_x_1[:,2], marker_z = data_y, c=:viridis, size=(1200,400), colorbar=false, title="|> warp")
p3 = histogram(trans_data_x_2_split; stacked=true, bins=50, size=(1200,400), title="|> dot")
plot(p1, p2, p3, layout=(1,3), legend=false)

using StatsFuns: normcdf
p1 = scatter(data_x[:,1], data_x[:,2], marker_z = data_y, title="original labels", xlabel="weight 1", ylabel="weight 2", size=(1200,400), c=:viridis)
p2 = scatter(data_x[:,1], data_x[:,2], marker_z = normcdf.(trans_data_x_2), title="predicted labels", xlabel="weight 1", ylabel="weight 2", size=(1200,400), c=:viridis)
p3 = contour(0:0.01:1, 0:0.01:1, (x, y) -> normcdf(dot([1,1], ReactiveMP.forward(inferred_model, [x,y]))), title="Classification map", xlabel="weight 1", ylabel="weight 2", size=(1200,400), c=:viridis)
plot(p1, p2, p3, layout=(1,3), legend=false)