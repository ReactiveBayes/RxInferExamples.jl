module GCViz

using Plots, Statistics, LinearAlgebra, Distributions

export plot_pos, plot_pos_vel, plot_free_energy_terms, summary_dashboard_all,
       plot_states, plot_residuals, plot_errors, plot_scatter_true_vs_inferred,
       make_animation_pos, make_animation_states,
       plot_rmse, plot_coverage, plot_residual_hist, plot_fe_cumsum,
       plot_y_fit, plot_stdres_hist, plot_stdres_qq, plot_stdres_acf, plot_mse_time, plot_state_coverage_time,
       plot_fe_iterations, plot_stdres_time, plot_derivative_consistency

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

# Number of state dimensions from marginals
_num_state_dims(xmargs) = length(mean(xmargs[1]))

# Labels for state dimensions
function _dim_label(k::Int)
    if k == 1
        return "Position"
    elseif k == 2
        return "Velocity"
    elseif k == 3
        return "Acceleration"
    else
        return "x^($(k-1))"
    end
end

function plot_pos(x_true::Vector{<:AbstractVector}, y::Vector{<:AbstractVector}, xmargs)
    n = length(x_true)
    px = plot(title = "Position (true, obs, inferred)")
    plot!(px, getindex.(x_true, 1), label = "True position", color = :black)
    scatter!(px, 1:n, getindex!(y, 1), ms = 2, alpha = 0.6, label = "Observations", color = :red)
    μ, σ = _means_stds(xmargs)
    plot!(px, getindex!(μ, 1), ribbon = getindex!(σ, 1), fillalpha = 0.3, label = "Inferred position", color = :blue)
    xlims!(px, 1, n)
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
    xlims!(p2, 1, length(x_true))
    return plot(p1, p2, layout = (2,1), size=(900,600))
end

# Plot all states with credible intervals
function plot_states(x_true::Vector{<:AbstractVector}, xmargs)
    μ, σ = _means_stds(xmargs)
    K = min(length(μ[1]), length(x_true[1]))
    rows = K
    grid = []
    for k in 1:K
        pk = plot(title = _dim_label(k), legend=:bottomright)
        plot!(pk, getindex!(x_true, k), label = "True", color=:black)
        plot!(pk, getindex!(μ, k), ribbon = getindex!(σ, k), label = "Inferred", color=:blue, fillalpha=0.3)
        xlims!(pk, 1, length(x_true))
        push!(grid, pk)
    end
    return plot(grid..., layout=(rows,1), size=(950,300*rows))
end

# Plot per-time free energy contributions
function plot_free_energy_terms(obs_term::AbstractVector, dyn_term::AbstractVector, total::AbstractVector)
    p = plot(title = "Per-time free energy terms", xlabel = "time", ylabel = "value")
    plot!(p, obs_term, label = "Obs term", color = :purple)
    plot!(p, dyn_term, label = "Dyn term", color = :orange)
    plot!(p, total, label = "Total", color = :green, lw=2)
    return p
end

# Free energy over iterations (exact ELBO reported by RxInfer)
function plot_fe_iterations(iter_fe::AbstractVector)
    iters = 1:length(iter_fe)
    p = plot(iters, iter_fe; lw=2, marker=:circle, ms=3, xlabel="iteration", ylabel="free energy",
             title="Free energy over iterations", legend=false)
    hline!(p, [iter_fe[end]]; color=:gray, lw=1, label="")
    return p
end

# Summary dashboard across key series
function summary_dashboard_all(x_true, y, xmargs, obs_term, dyn_term, total)
    μ, σ = _means_stds(xmargs)
    K = min(length(μ[1]), length(x_true[1]))
    # Build state panels
    state_panels = Vector{Any}(undef, K)
    for k in 1:K
        pk = plot(title = _dim_label(k), legend=:bottomright)
        plot!(pk, getindex!(x_true, k), label = "True", color=:black)
        plot!(pk, getindex!(μ, k), ribbon = getindex!(σ, k), label = "Inferred", color=:blue, fillalpha=0.3)
        xlims!(pk, 1, length(x_true))
        state_panels[k] = pk
    end
    # FE panel
    pfe = plot_free_energy_terms(obs_term, dyn_term, total)
    # Assemble grid: K state rows + 1 FE row
    return plot(state_panels..., pfe, layout=(K+1,1), size=(950, 300*(K+1)))
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
    n = min(length(x_true), length(xmargs))
    K = min(length(μ[1]), length(x_true[1]))
    e = [[abs(μ[t][k] - x_true[t][k]) for t in 1:n] for k in 1:K]
    p = plot(title = "Absolute errors", xlabel = "time", ylabel = "|error|")
    for k in 1:K
        plot!(p, e[k], label = _dim_label(k))
    end
    return p
end

# Scatter of inferred mean vs true per dimension
function plot_scatter_true_vs_inferred(x_true::Vector{<:AbstractVector}, xmargs)
    μ, _ = _means_stds(xmargs)
    n = min(length(x_true), length(xmargs))
    K = min(length(μ[1]), length(x_true[1]))
    p = plot(layout=(1,K), size=(330*K,300))
    for k in 1:K
        scatter!(p,
                 getindex!(x_true[1:n], k),
                 getindex!(μ[1:n], k);
                 ms=3, alpha=0.7, subplot=k, label="")
        plot!(p,
              getindex!(x_true[1:n], k),
              getindex!(x_true[1:n], k);
              color=:black, lw=1, subplot=k, label="")
        plot!(p; title="$(_dim_label(k)): inferred vs true", xlabel="true", ylabel="inferred", legend=false, subplot=k)
    end
    return p
end

# RMSE per dimension
function plot_rmse(x_true::Vector{<:AbstractVector}, xmargs)
    μ, _ = _means_stds(xmargs)
    n = min(length(x_true), length(xmargs))
    K = min(length(μ[1]), length(x_true[1]))
    rmse = [sqrt(mean([ (μ[t][k] - x_true[t][k])^2 for t in 1:n ])) for k in 1:K]
    bar([_dim_label(k) for k in 1:K], rmse; title="RMSE per state", ylabel="RMSE", legend=false)
end

# Coverage at 95% (μ ± 1.96σ)
function plot_coverage(x_true::Vector{<:AbstractVector}, xmargs)
    μ, σ = _means_stds(xmargs)
    n = min(length(x_true), length(xmargs))
    K = min(length(μ[1]), length(x_true[1]))
    coverage = [ mean([ abs(x_true[t][k] - μ[t][k]) <= 1.96 * σ[t][k] for t in 1:n ]) for k in 1:K ]
    bar([_dim_label(k) for k in 1:K], coverage; title="95% CI coverage", ylabel="fraction", ylim=(0,1), legend=false)
end

# Coverage over time (states)
function plot_state_coverage_time(x_true::Vector{<:AbstractVector}, xmargs)
    μ, σ = _means_stds(xmargs)
    n = min(length(x_true), length(xmargs))
    K = min(length(μ[1]), length(x_true[1]))
    p = plot(layout=(K,1), size=(900,260*K))
    for k in 1:K
        inside = [ abs(x_true[t][k] - μ[t][k]) <= 1.96 * σ[t][k] for t in 1:n ]
        plot!(p[k], inside; title="Coverage over time: $(_dim_label(k))", ylim=(0,1), legend=false)
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
        # Use histogram! to add series directly to the subplot
        histogram!(p, rd; bins=30, normalize=true, alpha=0.6, subplot=d, legend=false)
        plot!(p; title="Residuals dim $d", subplot=d)
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
        scatter!(p, 1:n, getindex!(y, d); ms=2, alpha=0.6, color=:red, label="obs", subplot=d)
        plot!(p, μy; ribbon=σy, fillalpha=0.3, color=:blue, label="pred", subplot=d)
        plot!(p; title="y[$d] fit", legend=:bottomright, subplot=d)
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
        histogram!(p, z; bins=30, normalize=true, alpha=0.6, subplot=d, legend=false)
        plot!(p; title="Std residuals dim $d", subplot=d)
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
    n = min(length(x_true), length(xmargs))
    K = min(length(μ[1]), length(x_true[1]))
    mse = [[ (μ[t][k] - x_true[t][k])^2 for t in 1:n ] for k in 1:K]
    p = plot(layout=(K,1), size=(900,260*K))
    for k in 1:K
        plot!(p[k], mse[k]; label="mse $(_dim_label(k))", title="MSE over time: $(_dim_label(k))")
    end
    return p
end

# Simple position animation
function make_animation_pos(x_true::Vector{<:AbstractVector}, y::Vector{<:AbstractVector}, xmargs, path::AbstractString; step::Int=1, fps::Int=30, maxframes::Int=typemax(Int))
    n = length(x_true)
    μ, σ = _means_stds(xmargs)
    frames = 0
    anim = @animate for t in 1:step:n
        frames += 1
        frames > maxframes && break
        p = plot(title = "Position inference (t=$(t))", legend=:topright)
        plot!(p, getindex!(x_true[1:t], 1), label = "True position", color = :black)
        scatter!(p, 1:t, getindex!(y[1:t], 1), ms = 2, alpha = 0.6, label = "Observations", color = :red)
        plot!(p, getindex!(μ[1:t], 1), ribbon = getindex!(σ[1:t], 1), fillalpha = 0.3, label = "Inferred position", color = :blue)
    end
    gif(anim, path; fps = fps)
end

# States animation (pos, vel, acc) inferred vs true
function make_animation_states(x_true::Vector{<:AbstractVector}, xmargs, path::AbstractString; step::Int=1, fps::Int=20, maxframes::Int=typemax(Int))
    n = length(x_true)
    μ, σ = _means_stds(xmargs)
    frames = 0
    anim = @animate for t in 1:step:n
        frames += 1
        frames > maxframes && break
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
    gif(anim, path; fps = fps)
end

# Standardized residuals over time
function plot_stdres_time(y::Vector{<:AbstractVector}, xmargs, B::AbstractMatrix, R::AbstractMatrix)
    n = length(y)
    μ, σ = _means_stds(xmargs)
    ny = length(first(y))
    p = plot(layout=(ny,1), size=(900, 300*ny))
    for d in 1:ny
        res = [ y[t][d] - (B*μ[t])[d] for t in 1:n ]
        Σy = [ B * Diagonal(σ[t].^2) * B' for t in 1:n ]
        σy = [ sqrt(Σy[t][d,d] + R[d,d]) for t in 1:n ]
        z = [ res[t] / σy[t] for t in 1:n ]
        plot!(p[d], z; label="stdres[$d]", title="Std residuals over time dim $d")
        hline!(p[d], [0.0]; color=:black, lw=1, label="")
    end
    return p
end

# Consistency of generalized coordinates: finite-difference checks
function plot_derivative_consistency(xmargs, dt::Real)
    μ, _ = _means_stds(xmargs)
    n = length(xmargs)
    K = length(μ[1])
    panels = Int((K >= 2) + (K >= 3))
    panels == 0 && return plot(title="Derivative consistency (requires K ≥ 2)")
    p = plot(layout=(panels,1), size=(900, 300*panels))
    row = 1
    if K >= 2
        dμ_pos = [ (μ[t+1][1] - μ[t][1]) / dt for t in 1:n-1 ]
        vel_inferred = getindex.(μ[1:n-1], 2)
        vel_err = [abs(dμ_pos[t] - vel_inferred[t]) for t in 1:n-1]
        plot!(p[row], vel_err; title="|d(pos)/dt - vel|", ylabel="mismatch")
        row += 1
    end
    if K >= 3
        dμ_vel = [ (μ[t+1][2] - μ[t][2]) / dt for t in 1:n-1 ]
        acc_inferred = getindex.(μ[1:n-1], 3)
        acc_err = [abs(dμ_vel[t] - acc_inferred[t]) for t in 1:n-1]
        plot!(p[row], acc_err; title="|d(vel)/dt - acc|", ylabel="mismatch", xlabel="time")
    end
    return p
end

end # module
