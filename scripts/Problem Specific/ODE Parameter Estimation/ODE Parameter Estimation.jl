# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Problem Specific/ODE Parameter Estimation/ODE Parameter Estimation.ipynb
# by notebooks_to_scripts.jl at 2025-03-31T09:50:41.414
#
# Source notebook: ODE Parameter Estimation.ipynb

using RxInfer, Optim, LinearAlgebra, Plots, SeeToDee, StaticArrays, StableRNGs

function lotka_volterra(u, z, p, t)
    α, β, δ, γ = p[SA[1,2,3,4]]
    x, y = u[SA[1, 2]]
    du1 = α * x - β * x * y
    du2 = -δ * y + γ * x * y

    return [du1, du2]
end;

dt = 0.1 # sample_interval

function lotka_volterra_rk4(x, θ, t, dt)
    lotka_volterra_dynamics = SeeToDee.Rk4(lotka_volterra, dt)
    return lotka_volterra_dynamics(x, 0, θ, t)
end

function generate_data(θ; x = ones(2), t =0.0, dt = 0.001, n = 1000, v = 1, seed = 123)
    rng = StableRNG(seed)
    data = Vector{Vector{Float64}}(undef, n)
    ts = Vector{Float64}(undef, n)
    for i in 1:n
        data[i] = lotka_volterra_rk4(x, θ, t, dt)
        x = data[i]
        t += dt
        ts[i] = t
    end
    noisy_data = map(data) do d
        noise = sqrt(v) * [randn(rng), randn(rng)]
        d + noise
    end
    return data, noisy_data, ts
end

noisev = 0.35
n = 10000
true_params = [1.0, 1.5, 3.0, 1.0]
data_long, noisy_data_long, ts_long = generate_data(true_params,dt = dt, n = n, v = noisev);

## We create a smaller dataset for the global parameter optimization. Utilizing the entire dataset for the global optimization will take too much time. 
n_train = 100
data = data_long[1:n_train]
noisy_data = noisy_data_long[1:n_train]
ts = ts_long[1:n_train];


p = plot(layout=(2,1))
plot!(subplot=1, ts, [d[1] for d in data], label="True x₁", color=:blue)
plot!(subplot=1, ts, [d[1] for d in noisy_data], seriestype=:scatter, label="Noisy x₁", color=:blue, alpha=0.3, markersize=1.3)
plot!(subplot=2, ts, [d[2] for d in data], label="True x₂", color=:red)
plot!(subplot=2, ts, [d[2] for d in noisy_data], seriestype=:scatter, label="Noisy x₂", color=:red, alpha=0.3, markersize=1.3)
xlabel!("Time")
ylabel!(subplot=1, "Prey Population")
ylabel!(subplot=2, "Predator Population")


@model function lotka_volterra_model_without_prior(obs, mprev, Vprev, dt, t, θ)
    xprev ~ MvNormalMeanCovariance(mprev, Vprev)
    x     := lotka_volterra_rk4(xprev, θ, t, dt)
    obs   ~ MvNormalMeanCovariance(x,  noisev * diageye(length(mprev)))
end

delta_meta = @meta begin
    lotka_volterra_rk4() ->  Linearization()
end

autoupdates_without_prior = @autoupdates begin
    mprev, Vprev= mean_cov(q(x))
end

@initialization function initialize_without_prior(mx, Vx)
    q(x) = MvNormalMeanCovariance(mx, Vx)
end;

function compute_free_energy_without_prior(θ ; mx = ones(2), Vx = 1e-6 * diageye(2))
    θ = exp.(θ)
    result = infer(
        model = lotka_volterra_model_without_prior(dt = dt, θ = θ),
        data = (obs = noisy_data, t= ts),
        initialization = initialize_without_prior(mx, Vx),
        meta = delta_meta,
        autoupdates = autoupdates_without_prior,
        keephistory = length(noisy_data),
        free_energy = true
    )
    return sum(result.free_energy_final_only_history)
end;

res_without_prior  = optimize(compute_free_energy_without_prior, zeros(4), NelderMead(), Optim.Options(show_trace = true, show_every = 300));

θ_minimizer_without_prior = exp.(res_without_prior.minimizer)
println("\nEstimated point mass valued parameters:")
for (i, (name, val)) in enumerate(zip(["α", "β", "γ", "δ"], θ_minimizer_without_prior))
    println(" * $name: $(round(val, digits=3))")
end

println("\nActual parameters used to generate data:")
for (i, (name, val)) in enumerate(zip(["α", "β", "γ", "δ"], true_params))
    println(" * $name: $(round(val, digits=3))")
end

@model function lotka_volterra_model(obs, mprev, Vprev, dt, t, mθ, Vθ)
    θ     ~ MvNormalMeanCovariance(mθ, Vθ)
    xprev ~ MvNormalMeanCovariance(mprev, Vprev)
    x     := lotka_volterra_rk4(xprev, θ, t, dt)
    obs   ~ MvNormalMeanCovariance(x,  noisev * diageye(length(mprev)))
end

autoupdates = @autoupdates begin
    mprev, Vprev= mean_cov(q(x))
    mθ, Vθ = mean_cov(q(θ))
end

@initialization function initialize(mx, Vx, mθ, Vθ)
    q(x) = MvNormalMeanCovariance(mx, Vx)
    q(θ) = MvNormalMeanCovariance(mθ, Vθ)
end;


function compute_free_energy(θ ; mx = ones(2), Vx = 1e-6 * diageye(2))
    θ = exp.(θ)
    mθ = θ[1:4]
    Vθ = Diagonal(θ[5:end])
    result = infer(
        model = lotka_volterra_model(dt = dt,),
        data = (obs = noisy_data, t = ts),
        initialization = initialize(mx, Vx, mθ, Vθ),
        meta = delta_meta,
        autoupdates = autoupdates,
        keephistory = length(noisy_data),
        free_energy = true
    )
    return sum(result.free_energy_final_only_history)
end;

res = optimize(compute_free_energy, [zeros(4); 0.1ones(4)], NelderMead(), Optim.Options(show_trace = true, show_every = 300));


θ_minimizer = exp.(res.minimizer)
mθ_init = θ_minimizer[1:4]
Vθ_init = Diagonal(θ_minimizer[5:end])

println("\nEstimated initialization parameters for the prior distribution:")
for (i, (name, val, var)) in enumerate(zip(["α", "β", "γ", "δ"], mθ_init, θ_minimizer[5:8]))
    println(" * $name: $(round(val, digits=3)) ± $(round(sqrt(var), digits=3))")
end

println("\nActual parameters used to generate data:")
for (i, (name, val)) in enumerate(zip(["α", "β", "γ", "δ"], true_params))
    println(" * $name: $(round(val, digits=3))")
end



result = infer(
    model = lotka_volterra_model(dt = dt,),
    data = (obs = noisy_data_long, t= ts_long),
    initialization = initialize(ones(2), 1e-6 * diageye(2), mθ_init, Vθ_init),
    meta = delta_meta,
    autoupdates = autoupdates,
    keephistory = length(noisy_data_long),
    free_energy = true
);

mθ_posterior = mean.(result.history[:θ])
Vθ_posterior = var.(result.history[:θ])

p = plot(layout=(4,1), size=(800,1000), legend=:right)

param_names = ["α", "β", "γ", "δ"]

for i in 1:4
    means = [m[i] for m in mθ_posterior]
    stds = [2sqrt(v[i]) for v in Vθ_posterior]
    
    plot!(p[i], means, ribbon=stds, label="Posterior", subplot=i)
    hline!(p[i], [true_params[i]], label="True value", linestyle=:dash, color=:red, subplot=i)
    
    title!(p[i], param_names[i], subplot=i)
    if i == 4 
        xlabel!(p[i], "Time step", subplot=i)
    end
end

# Place legend at top right for all subplots
plot!(p, legend=:topright)

display(p)
final_means = last(mθ_posterior)
final_vars = last(Vθ_posterior)
final_stds = sqrt.(final_vars)

# Print results
println("\nFinal Parameter Estimates:")
for (param, mean, std) in zip(param_names, final_means, final_stds)
    println("$param: $mean ± $(std)")
end

# Get final covariance matrix
final_cov = cov(last(result.history[:θ]))
println("\nFinal Parameter Covariance Matrix:")
display(final_cov)



from = 1
skip = 1        
to = 500

# Get state estimates and variances
mx = mean.(result.history[:x])
Vx = var.(result.history[:x])

# Plot state estimates with uncertainty bands
p1 = plot(ts_long[from:skip:to] , getindex.(mx, 1)[from:skip:to], ribbon=2*sqrt.(getindex.(Vx, 1)[from:skip:to]), 
          label="Prey estimate", legend=:topright)
scatter!(p1, ts_long[from:skip:to], getindex.(noisy_data_long, 1)[from:skip:to], label="Noisy prey observations", alpha=0.5,ms=1)
plot!(p1, ts_long[from:skip:to], getindex.(data_long, 1)[from:skip:to], label="True prey", linestyle=:dash)
title!(p1, "Prey Population")

p2 = plot(ts_long[from:skip:to], getindex.(mx, 2)[from:skip:to], ribbon=2*sqrt.(getindex.(Vx, 2)[from:skip:to]), 
          label="Predator estimate", legend=:topright)
scatter!(p2, ts_long[from:skip:to], getindex.(noisy_data_long, 2)[from:skip:to], label="Noisy predator observations", alpha=0.5, ms=1)
plot!(p2, ts_long[from:skip:to], getindex.(data_long, 2)[from:skip:to] , label="True predator", linestyle=:dash)
title!(p2, "Predator Population")

plot(p1, p2, layout=(2,1), size=(1000,600))

expf(θ) = exp.(θ) ## This function is used to apply the exp function to the parameters within the @model macro

@model function lotka_volterra_model2(obs, mprev, Vprev, dt, t, mθ, Vθ)
    θ     ~ MvNormalMeanCovariance(mθ, Vθ)
    xprev ~ MvNormalMeanCovariance(mprev, Vprev)
    θ_exp := expf(θ)
    x     := lotka_volterra_rk4(xprev, θ_exp, t, dt)
    obs   ~ MvNormalMeanCovariance(x,  noisev * diageye(length(mprev)))
end

delta_meta2 = @meta begin
    lotka_volterra_rk4() ->  Unscented()
    expf() ->  Unscented()
end

autoupdates2 = @autoupdates begin
    mprev, Vprev= mean_cov(q(x))
    mθ, Vθ = mean_cov(q(θ))
end

@initialization function initialize2(mx, Vx, mθ, Vθ)
    q(x) = MvNormalMeanCovariance(mx, Vx)
    q(θ) = MvNormalMeanCovariance(mθ, Vθ)
end


result2  = infer(
    model = lotka_volterra_model2(dt = dt,),
    data = (obs = noisy_data_long, t= ts_long),
    initialization = initialize2(ones(2),  1e-6diageye(2), zeros(4), 0.1*diageye(4)),
    meta = delta_meta2,
    autoupdates = autoupdates2,
    keephistory = length(noisy_data_long),
    free_energy = true
)


mθ_exp =  mean.(result2.history[:θ_exp])
Vθ_exp = var.(result2.history[:θ_exp])

# Plot the inferred parameters with uncertainty
p1 = plot(ts_long, getindex.(mθ_exp, 1), ribbon=2*sqrt.(getindex.(Vθ_exp, 1)), label="α", legend=:topright)
plot!(p1, ts_long, fill(true_params[1], length(ts_long)), label="True α", linestyle=:dash)
title!(p1, "Parameter α")

p2 = plot(ts_long, getindex.(mθ_exp, 2), ribbon=2*sqrt.(getindex.(Vθ_exp, 2)), label="β", legend=:topright)
plot!(p2, ts_long, fill(true_params[2], length(ts_long)), label="True β", linestyle=:dash)
title!(p2, "Parameter β")

p3 = plot(ts_long, getindex.(mθ_exp, 3), ribbon=2*sqrt.(getindex.(Vθ_exp, 3)), label="γ", legend=:topright)
plot!(p3, ts_long, fill(true_params[3], length(ts_long)), label="True γ", linestyle=:dash)
title!(p3, "Parameter γ")

p4 = plot(ts_long, getindex.(mθ_exp, 4), ribbon=2*sqrt.(getindex.(Vθ_exp, 4)), label="δ", legend=:topright)
plot!(p4, ts_long, fill(true_params[4], length(ts_long)), label="True δ", linestyle=:dash)
title!(p4, "Parameter δ")

plot(p1, p2, p3, p4, layout=(4,1), size=(1000,800))


# Print final parameter estimates and covariance
final_means = last(mθ_exp)
final_vars = last(Vθ_exp)
final_stds = sqrt.(final_vars)

# Print results
println("\nFinal Parameter Estimates:")
for (param, mean, std) in zip(param_names, final_means, final_stds)
    println("$param: $mean ± $(std)")
end

println("\nFinal parameter covariance matrix:")
display(cov(last(result2.history[:θ_exp])))


# Get state estimates and variances
mx = mean.(result2.history[:x])
Vx = var.(result2.history[:x])

# Plot state estimates with uncertainty bands
p1 = plot(ts_long[from:skip:to] , getindex.(mx, 1)[from:skip:to], ribbon=2*sqrt.(getindex.(Vx, 1)[from:skip:to]), 
          label="Prey estimate", legend=:topright)
scatter!(p1, ts_long[from:skip:to], getindex.(noisy_data_long, 1)[from:skip:to], label="Noisy prey observations", alpha=0.5,ms=1)
plot!(p1, ts_long[from:skip:to], getindex.(data_long, 1)[from:skip:to], label="True prey", linestyle=:dash)
title!(p1, "Prey Population")

p2 = plot(ts_long[from:skip:to], getindex.(mx, 2)[from:skip:to], ribbon=2*sqrt.(getindex.(Vx, 2)[from:skip:to]), 
          label="Predator estimate", legend=:topright)
scatter!(p2, ts_long[from:skip:to], getindex.(noisy_data_long, 2)[from:skip:to], label="Noisy predator observations", alpha=0.5, ms=1)
plot!(p2, ts_long[from:skip:to], getindex.(data_long, 2)[from:skip:to] , label="True predator", linestyle=:dash)
title!(p2, "Predator Population")

plot(p1, p2, layout=(2,1), size=(1000,600))
