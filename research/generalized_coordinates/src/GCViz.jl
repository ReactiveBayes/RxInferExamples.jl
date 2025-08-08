module GCViz

using Plots, Statistics, LinearAlgebra, Distributions

export plot_pos, plot_pos_vel, plot_free_energy_terms, summary_dashboard,
       plot_states, plot_residuals, plot_errors, plot_scatter_true_vs_inferred,
       make_animation_pos, make_animation_states,
       plot_rmse, plot_coverage, plot_residual_hist, plot_fe_cumsum,
       plot_y_fit, plot_stdres_hist, plot_stdres_qq, plot_stdres_acf, plot_mse_time, plot_state_coverage_time

# Use a non-interactive backend for headless environments
try
    gr(fmt = :png)
catch
end

# Internal helper to get means and diagonal stds from marginals
function _means_stds(xmargs)
    n = length(xmargs)
    μ = [mean(xmargs[t]) for t in 1:n]
    vdiag = Vector{Vector{Float64}}(undef, n)
    for t in 1:n
        Σraw = try
            cov(xmargs[t])
        catch
            var(xmargs[t])
        end
        vdiag[t] = Σraw isa AbstractVector ? collect(Σraw) : diag(Σraw)
    end
    σ = [sqrt.(vdiag[t]) for t in 1:n]
    return μ, σ
end

# Safe getindex for array of vectors
getindex!(arr, k) = getindex.(arr, k)

function plot_pos(x_true::Vector{<:AbstractVector}, y::Vector{<:AbstractVector}, xmargs)
    n = length(x_true)
    px = plot(title = "Position (true, obs, inferred)")
    plot!(px, getindex.(x_true, 1), label = "True position", color = :black)
    scatter!(px, 1:n, getindex!(y, 1), ms = 2, alpha = 0.6, label = "Observations", color = :red)
    μ, σ = _means_stds(xmargs)
    plot!(px, getindex!(μ, 1), ribbon = getindex!(σ, 1), fillalpha = 0.3, label = "Inferred position", color = :blue)
    return px
end

function plot_pos_vel(x_true::Vector{<:AbstractVector}, y::Vector{<:AbstractVector}, xmargs)
    p1 = plot_pos(x_true, y, xmargs)
    p2 = plot(title = "Velocity (true, obs?, inferred)")
    plot!(p2, getindex!(x_true, 2), label = "True velocity", color = :black)
    if length(first(y)) > 1
        scatter!(p2, getindex!(y, 2), ms = 2, alpha = 0.6, label = "Observed vel", color = :red)
    end
    μ, σ = _means_stds(xmargs)
    plot!(p2, getindex!(μ, 2), ribbon = getindex!(σ, 2), fillalpha = 0.3, label = "Inferred velocity", color = :blue)
    return plot(p1, p2, layout = (2,1), size=(900,600))
end

# Plot all states with credible intervals
function plot_states(x_true::Vector{<:AbstractVector}, xmargs)
    μ, σ = _means_stds(xmargs)
    p1 = plot(title = "Position", legend=:bottomright)
    plot!(p1, getindex!(x_true, 1), label = "True", color=:black)
    plot!(p1, getindex!(μ, 1), ribbon = getindex!(σ, 1), label = "Inferred", color=:blue, fillalpha=0.3)

    p2 = plot(title = "Velocity", legend=:bottomright)
    plot!(p2, getindex!(x_true, 2), label = "True", color=:black)
    plot!(p2, getindex!(μ, 2), ribbon = getindex!(σ, 2), label = "Inferred", color=:green, fillalpha=0.3)

    p3 = plot(title = "Acceleration", legend=:bottomright)
    plot!(p3, getindex!(x_true, 3), label = "True", color=:black)
    plot!(p3, getindex!(μ, 3), ribbon = getindex!(σ, 3), label = "Inferred", color=:orange, fillalpha=0.3)

    return plot(p1, p2, p3, layout=(3,1), size=(950,900))
end

# Plot per-time free energy contributions
function plot_free_energy_terms(obs_term::AbstractVector, dyn_term::AbstractVector, total::AbstractVector)
    p = plot(title = "Per-time free energy terms", xlabel = "time", ylabel = "value")
    plot!(p, obs_term, label = "Obs term", color = :purple)
    plot!(p, dyn_term, label = "Dyn term", color = :orange)
    plot!(p, total, label = "Total", color = :green, lw=2)
    return p
end

# Summary dashboard across key series
function summary_dashboard(x_true, y, xmargs, obs_term, dyn_term, total)
    p1 = plot_pos(x_true, y, xmargs)
    μ, _ = _means_stds(xmargs)
    p2 = plot(title = "Velocity (inferred)")
    plot!(p2, getindex!(μ, 2), label = "Inferred velocity", color=:blue)
    p3 = plot_free_energy_terms(obs_term, dyn_term, total)
    return plot(p1, p2, p3, layout=(3,1), size=(950,900))
end

# Plot residuals (observed - inferred projections) for available dims
function plot_residuals(x_true::Vector{<:AbstractVector}, y::Vector{<:AbstractVector}, xmargs, B::AbstractMatrix)
    n = length(y)
    μ, _ = _means_stds(xmargs)
    μy = [B * μ[t] for t in 1:n]
    res = [y[t] .- μy[t] for t in 1:n]
    p = plot(title = "Observation residuals", xlabel = "time", ylabel = "residual")
    for d in 1:length(first(y))
        plot!(p, getindex!(res, d), label = "res[$d]")
    end
    return p
end

# Plot absolute errors for each state dimension
function plot_errors(x_true::Vector{<:AbstractVector}, xmargs)
    μ, _ = _means_stds(xmargs)
    e = [[abs(μ[t][k] - x_true[t][k]) for t in 1:length(xmargs)] for k in 1:3]
    p = plot(title = "Absolute errors", xlabel = "time", ylabel = "|error|")
    colors = [:blue, :green, :orange]
    labels = ["pos", "vel", "acc"]
    for k in 1:3
        plot!(p, e[k], label = labels[k], color = colors[k])
    end
    return p
end

# Scatter of inferred mean vs true per dimension
function plot_scatter_true_vs_inferred(x_true::Vector{<:AbstractVector}, xmargs)
    μ, _ = _means_stds(xmargs)
    p = plot(layout=(1,3), size=(1000,300))
    labels = ["pos", "vel", "acc"]
    for k in 1:3
        pk = plot(title = "$(labels[k]): inferred vs true", legend=false, xlabel="true", ylabel="inferred")
        scatter!(pk, getindex!(x_true, k), getindex!(μ, k), ms=3, alpha=0.7)
        plot!(pk, getindex!(x_true, k), getindex!(x_true, k), color=:black, lw=1)
        plot!(p[k], pk)
    end
    return p
end

# RMSE per dimension
function plot_rmse(x_true::Vector{<:AbstractVector}, xmargs)
    μ, _ = _means_stds(xmargs)
    rmse = [sqrt(mean([ (μ[t][k] - x_true[t][k])^2 for t in 1:length(xmargs) ])) for k in 1:3]
    bar(["pos","vel","acc"], rmse; title="RMSE per state", ylabel="RMSE", legend=false)
end

# Coverage at 95% (μ ± 1.96σ)
function plot_coverage(x_true::Vector{<:AbstractVector}, xmargs)
    μ, σ = _means_stds(xmargs)
    coverage = Float64[]
    for k in 1:3
        inside = [ abs(x_true[t][k] - μ[t][k]) <= 1.96 * σ[t][k] for t in 1:length(xmargs) ]
        push!(coverage, mean(inside))
    end
    bar(["pos","vel","acc"], coverage; title="95% CI coverage", ylabel="fraction", ylim=(0,1), legend=false)
end

# Coverage over time (states)
function plot_state_coverage_time(x_true::Vector{<:AbstractVector}, xmargs)
    μ, σ = _means_stds(xmargs)
    p = plot(layout=(3,1), size=(900,800))
    labels = ["pos", "vel", "acc"]
    for k in 1:3
        inside = [ abs(x_true[t][k] - μ[t][k]) <= 1.96 * σ[t][k] for t in 1:length(xmargs) ]
        plot!(p[k], inside; title="Coverage over time: $(labels[k])", ylim=(0,1), legend=false)
    end
    return p
end

# Residual histogram for observed dims
function plot_residual_hist(y::Vector{<:AbstractVector}, xmargs, B::AbstractMatrix)
    n = length(y)
    μ, _ = _means_stds(xmargs)
    μy = [B * μ[t] for t in 1:n]
    res = [y[t] .- μy[t] for t in 1:n]
    layout = (1, length(first(y)))
    p = plot(layout=layout, size=(320*length(first(y)), 300))
    for d in 1:length(first(y))
        rd = getindex!(res, d)
        plot!(p[d], histogram(rd; bins=30, normalize=true, alpha=0.6), title="Residuals dim $d", legend=false)
    end
    return p
end

# Posterior predictive fit for y with ribbons
function plot_y_fit(y::Vector{<:AbstractVector}, xmargs, B::AbstractMatrix, R::AbstractMatrix)
    n = length(y)
    μ, σ = _means_stds(xmargs)
    ny = length(first(y))
    p = plot(layout=(ny,1), size=(900, 350*ny))
    for d in 1:ny
        μy = [ (B * μ[t])[d] for t in 1:n ]
        Σy = [ B * Diagonal(σ[t].^2) * B' for t in 1:n ]
        σy = [ sqrt(Σy[t][d,d] + R[d,d]) for t in 1:n ]
        pd = plot(title = "y[$d] fit", legend=:bottomright)
        scatter!(pd, 1:n, getindex!(y, d), ms=2, alpha=0.6, color=:red, label="obs")
        plot!(pd, μy, ribbon=σy, fillalpha=0.3, color=:blue, label="pred")
        plot!(p[d], pd)
    end
    return p
end

# Standardized residuals histogram
function plot_stdres_hist(y::Vector{<:AbstractVector}, xmargs, B::AbstractMatrix, R::AbstractMatrix)
    n = length(y)
    μ, σ = _means_stds(xmargs)
    ny = length(first(y))
    p = plot(layout=(ny,1), size=(900, 300*ny))
    for d in 1:ny
        res = [ y[t][d] - (B*μ[t])[d] for t in 1:n ]
        Σy = [ B * Diagonal(σ[t].^2) * B' for t in 1:n ]
        σy = [ sqrt(Σy[t][d,d] + R[d,d]) for t in 1:n ]
        z = [ res[t] / σy[t] for t in 1:n ]
        plot!(p[d], histogram(z; bins=30, normalize=true, alpha=0.6), title="Std residuals dim $d", legend=false)
    end
    return p
end

# QQ plot of standardized residuals
function plot_stdres_qq(y::Vector{<:AbstractVector}, xmargs, B::AbstractMatrix, R::AbstractMatrix)
    n = length(y)
    μ, σ = _means_stds(xmargs)
    ny = length(first(y))
    p = plot(layout=(ny,1), size=(900, 300*ny))
    for d in 1:ny
        res = [ y[t][d] - (B*μ[t])[d] for t in 1:n ]
        Σy = [ B * Diagonal(σ[t].^2) * B' for t in 1:n ]
        σy = [ sqrt(Σy[t][d,d] + R[d,d]) for t in 1:n ]
        z = sort([ res[t] / σy[t] for t in 1:n ])
        q = [quantile(Normal(), (i-0.5)/n) for i in 1:n]
        scatter!(p[d], q, z; ms=2, alpha=0.7, label="stdres")
        plot!(p[d], q, q; color=:black, lw=1, label="y=x", title="QQ stdres dim $d")
    end
    return p
end

# Autocorrelation of standardized residuals
function _acf(x::AbstractVector, maxlag::Int)
    μ = mean(x); σ = std(x)
    σ == 0 && return zeros(maxlag)
    ac = zeros(maxlag)
    for l in 1:maxlag
        x1 = view(x, 1:length(x)-l)
        x2 = view(x, 1+l:length(x))
        ac[l] = cor(x1, x2)
    end
    return ac
end

function plot_stdres_acf(y::Vector{<:AbstractVector}, xmargs, B::AbstractMatrix, R::AbstractMatrix; maxlag::Int=30)
    n = length(y)
    μ, σ = _means_stds(xmargs)
    ny = length(first(y))
    p = plot(layout=(ny,1), size=(900, 300*ny))
    for d in 1:ny
        res = [ y[t][d] - (B*μ[t])[d] for t in 1:n ]
        Σy = [ B * Diagonal(σ[t].^2) * B' for t in 1:n ]
        σy = [ sqrt(Σy[t][d,d] + R[d,d]) for t in 1:n ]
        z = [ res[t] / σy[t] for t in 1:n ]
        ac = _acf(z, maxlag)
        bar!(p[d], 1:maxlag, ac; legend=false, title="Stdres ACF dim $d")
        hline!(p[d], [0.0]; color=:black, lw=1, label="")
    end
    return p
end

# FE cumulative sum
plot_fe_cumsum(total::AbstractVector) = plot(cumsum(total); title="Cumulative free energy", xlabel="time", ylabel="cumsum FE")

# Per-time MSE of states
function plot_mse_time(x_true::Vector{<:AbstractVector}, xmargs)
    μ, _ = _means_stds(xmargs)
    n = length(xmargs)
    mse = [[ (μ[t][k] - x_true[t][k])^2 for t in 1:n ] for k in 1:3]
    p = plot(layout=(3,1), size=(900,900))
    labels = ["pos","vel","acc"]
    colors = [:blue,:green,:orange]
    for k in 1:3
        plot!(p[k], mse[k]; color=colors[k], label="mse $(labels[k])", title="MSE over time: $(labels[k])")
    end
    return p
end

# Simple position animation
function make_animation_pos(x_true::Vector{<:AbstractVector}, y::Vector{<:AbstractVector}, xmargs, path::AbstractString)
    n = length(x_true)
    μ, σ = _means_stds(xmargs)
    anim = @animate for t in 1:n
        p = plot(title = "Position inference (t=$(t))", legend=:topright)
        plot!(p, getindex!(x_true[1:t], 1), label = "True position", color = :black)
        scatter!(p, 1:t, getindex!(y[1:t], 1), ms = 2, alpha = 0.6, label = "Observations", color = :red)
        plot!(p, getindex!(μ[1:t], 1), ribbon = getindex!(σ[1:t], 1), fillalpha = 0.3, label = "Inferred position", color = :blue)
    end
    gif(anim, path; fps = 30)
end

# States animation (pos, vel, acc) inferred vs true
function make_animation_states(x_true::Vector{<:AbstractVector}, xmargs, path::AbstractString)
    n = length(x_true)
    μ, σ = _means_stds(xmargs)
    anim = @animate for t in 1:n
        p1 = plot(title = "Position", legend=false)
        plot!(p1, getindex!(x_true[1:t], 1), color=:black)
        plot!(p1, getindex!(μ[1:t], 1), ribbon = getindex!(σ[1:t], 1), color=:blue, fillalpha=0.3)
        p2 = plot(title = "Velocity", legend=false)
        plot!(p2, getindex!(x_true[1:t], 2), color=:black)
        plot!(p2, getindex!(μ[1:t], 2), ribbon = getindex!(σ[1:t], 2), color=:green, fillalpha=0.3)
        p3 = plot(title = "Acceleration", legend=false)
        plot!(p3, getindex!(x_true[1:t], 3), color=:black)
        plot!(p3, getindex!(μ[1:t], 3), ribbon = getindex!(σ[1:t], 3), color=:orange, fillalpha=0.3)
        plot(p1, p2, p3, layout=(3,1), size=(900,900))
    end
    gif(anim, path; fps = 20)
end

end # module
