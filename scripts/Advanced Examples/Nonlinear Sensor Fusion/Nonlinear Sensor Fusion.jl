# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Advanced Examples/Nonlinear Sensor Fusion/Nonlinear Sensor Fusion.ipynb
# by notebooks_to_scripts.jl at 2025-04-29T06:39:07.224
#
# Source notebook: Nonlinear Sensor Fusion.ipynb

using RxInfer, Random, LinearAlgebra, Distributions, Plots, StatsPlots, Optimisers
using DataFrames, DelimitedFiles, StableRNGs

# fetch measurements
beacon_locations = readdlm("data/beacons.txt")
distances = readdlm("data/distances.txt")
position = readdlm("data/position.txt")
nr_observations = size(distances, 1);

# plot beacon and actual location of WALL-E
p1 = scatter(beacon_locations[:,1], beacon_locations[:,2], markershape=:utriangle, markersize=10, legend=:topleft, label="beacon locations")
plot!(position[1,:], position[2,:], label="actual location", linewidth=3, linestyle=:dash, arrow=(:closed, 2.0), aspect_ratio=1.0)
xlabel!("longitude [m]"), ylabel!("latitude [m]")

# plot noisy distance measurements
p2 = plot(distances, legend=:topleft, linewidth=3, label=["distance to beacon 1" "distance to beacon 2" "distance to beacon 3"])
xlabel!("time [sec]"), ylabel!("distance [m]")

plot(p1, p2, size=(1200, 500))

# function to compute distance to beacons
function compute_distances(z)    
    distance1 = norm(z - beacon_locations[1,:])
    distance2 = norm(z - beacon_locations[2,:])
    distance3 = norm(z - beacon_locations[3,:])
    distances = [distance1, distance2, distance3]
end;

@model function random_walk_model(y, W, R)
    # specify initial estimates of the location
    z[1] ~ MvNormalMeanCovariance(zeros(2), diageye(2)) 
    y[1] ~ MvNormalMeanCovariance(compute_distances(z[1]), diageye(3))

    # loop over time steps
    for t in 2:length(y)

        # specify random walk state transition model
        z[t] ~ MvNormalMeanPrecision(z[t-1], W)

        # specify non-linear distance observations model
        y[t] ~ MvNormalMeanPrecision(compute_distances(z[t]), R)
        
    end

end;

@meta function random_walk_model_meta(nr_samples, nr_iterations, rng)
    compute_distances(z) -> CVI(rng, nr_samples, nr_iterations, Optimisers.Descent(0.1), ForwardDiffGrad(), 1, Val(false), false)
end;

@meta function random_walk_linear_meta()
    compute_distances(z) -> Linearization()
end;

@meta function random_walk_unscented_meta()
    compute_distances(z) -> Unscented()
end;

init = @initialization begin 
    μ(z) = MvNormalMeanPrecision(ones(2), 0.1 * diageye(2))
end

results_fast = infer(
    model = random_walk_model(W = diageye(2), R = diageye(3)),
    meta = random_walk_model_meta(1, 3, StableRNG(42)), # or random_walk_unscented_meta()
    data = (y = [distances[t,:] for t in 1:nr_observations],),
    iterations = 20,
    free_energy = false,
    returnvars = (z = KeepLast(),),
    initialization = init,
);

results_accuracy = infer(
    model = random_walk_model(W = diageye(2), R = diageye(3)),
    meta = random_walk_model_meta(1000, 100, StableRNG(42)),
    data = (y = [distances[t,:] for t in 1:nr_observations],),
    iterations = 20,
    free_energy = false,
    returnvars = (z = KeepLast(),),
    initialization = init,
);

# plot beacon and actual and estimated location of WALL-E (fast inference)
p1 = scatter(beacon_locations[:,1], beacon_locations[:,2], markershape=:utriangle, markersize=10, legend=:topleft, label="beacon locations")
plot!(position[1,:], position[2,:], label="actual location", linewidth=3, linestyle=:dash, arrow=(:closed, 2.0), aspect_ratio=1.0)
map(posterior -> covellipse!(mean(posterior), cov(posterior), color="red", label="", n_std=2), results_fast.posteriors[:z])
xlabel!("longitude [m]"), ylabel!("latitude [m]"), title!("Fast (1 sample, 3 iterations)"); p1.series_list[end][:label] = "estimated location ±2σ"

# plot beacon and actual and estimated location of WALL-E (accurate inference)
p2 = scatter(beacon_locations[:,1], beacon_locations[:,2], markershape=:utriangle, markersize=10, legend=:topleft, label="beacon locations")
plot!(position[1,:], position[2,:], label="actual location", linewidth=3, linestyle=:dash, arrow=(:closed, 2.0), aspect_ratio=1.0)
map(posterior -> covellipse!(mean(posterior), cov(posterior), color="red", label="", n_std=2), results_accuracy.posteriors[:z])
xlabel!("longitude [m]"), ylabel!("latitude [m]"), title!("Accurate (1000 samples, 100 iterations)"); p2.series_list[end][:label] = "estimated location ±2σ"

plot(p1, p2, size=(1200, 500))

@model function random_walk_model_wishart(y)
    # set priors on precision matrices
    Q ~ Wishart(3, diageye(2))
    R ~ Wishart(4, diageye(3))

    # specify initial estimates of the location
    z[1] ~ MvNormalMeanCovariance(zeros(2), diageye(2)) 
    y[1] ~ MvNormalMeanCovariance(compute_distances(z[1]), diageye(3))

    # loop over time steps
    for t in 2:length(y)

        # specify random walk state transition model
        z[t] ~ MvNormalMeanPrecision(z[t-1], Q)

        # specify non-linear distance observations model
        y[t] ~ MvNormalMeanPrecision(compute_distances(z[t]), R)
        
    end

end;

meta = @meta begin 
    compute_distances(z) -> CVI(StableRNG(42), 1000, 100, Optimisers.Descent(0.01), ForwardDiffGrad(), 1, Val(false), false)
end;

constraints = @constraints begin
    q(z, Q, R) = q(z)q(Q)q(R)
end;

init = @initialization begin 
    μ(z) = MvNormalMeanPrecision(zeros(2), 0.01 * diageye(2))
    q(R) = Wishart(4, diageye(3))
    q(Q) = Wishart(3, diageye(2))
end;

results_wishart = infer(
    model = random_walk_model_wishart(),
    data = (y = [distances[t,:] for t in 1:nr_observations],),
    iterations = 20,
    free_energy = true,
    returnvars = (z = KeepLast(),),
    constraints = constraints,
    meta = meta,
    initialization = init,
);

# plot beacon and actual and estimated location of WALL-E (fast inference)
p1 = scatter(beacon_locations[:,1], beacon_locations[:,2], markershape=:utriangle, markersize=10, legend=:topleft, label="beacon locations")
plot!(position[1,:], position[2,:], label="actual location", linewidth=3, linestyle=:dash, arrow=(:closed, 2.0), aspect_ratio=1.0)
map(posterior -> covellipse!(mean(posterior), cov(posterior), color="red", label="", n_std=2), results_wishart.posteriors[:z])
xlabel!("longitude [m]"), ylabel!("latitude [m]"); p1.series_list[end][:label] = "estimated location ±2σ"

# plot bethe free energy performance
p2 = plot(results_wishart.free_energy[2:end], label = "")
xlabel!("iteration"), ylabel!("Bethe free energy [nats]")

plot(p1, p2, size=(1200, 500))