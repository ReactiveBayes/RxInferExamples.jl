module Viz

using Plots
using Statistics
using Logging
using Distributions
using Printf

export plot_hidden_states, plot_free_energy, plot_param_posteriors, make_hidden_states_animation,
       plot_state_errors, plot_residuals, plot_variance_trajectories,
       plot_residual_acf, plot_residual_qq, plot_coverage,
       make_residuals_animation

function plot_hidden_states(z_true::AbstractVector, x_true::AbstractVector, y_obs::AbstractVector, z_est, x_est; title_suffix = "")
    n = length(y_obs)
    pz = plot(title = "Hidden States Z" * (isempty(title_suffix) ? "" : " — " * title_suffix))
    px = plot(title = "Hidden States X" * (isempty(title_suffix) ? "" : " — " * title_suffix))
    plot!(pz, 1:n, z_true, label = "z", color = :orange)
    plot!(pz, 1:n, mean.(z_est), ribbon = std.(z_est), label = "ẑ", color = :teal)
    plot!(px, 1:n, x_true, label = "x", color = :green)
    plot!(px, 1:n, mean.(x_est), ribbon = std.(x_est), label = "x̂", color = :violet)
    scatter!(px, 1:n, y_obs, label = "y", color = :red, ms = 2, alpha = 0.2)
    return plot(pz, px, layout = @layout([ a; b ]))
end

function plot_free_energy(free_energy_history; label = "Bethe Free Energy")
    return plot(free_energy_history, label = label)
end

function plot_param_posteriors(q_κ, q_ω; real_k = nothing, real_w = nothing)
    range_w = range(-1, 0.5, length = 1000)
    range_k = range(0, 2, length = 1000)
    pw = plot(title = "Marginal q(ω)")
    plot!(pw, range_w, x -> pdf(q_ω, x), fillalpha = 0.3, fillrange = 0, label = "Posterior q(ω)", c = 3, legend_position = (0.1, 0.95), legendfontsize = 9)
    if real_w !== nothing
        vline!(pw, [real_w], label = "Real ω")
    end
    xlabel!(pw, "ω")

    pk = plot(title = "Marginal q(κ)")
    plot!(pk, range_k, x -> pdf(q_κ, x), fillalpha = 0.3, fillrange = 0, label = "Posterior q(κ)", c = 3, legend_position = (0.1, 0.95), legendfontsize = 9)
    if real_k !== nothing
        vline!(pk, [real_k], label = "Real κ")
    end
    xlabel!(pk, "κ")

    return plot(pk, pw, layout = @layout([ a; b ]))
end

function make_hidden_states_animation(z_true::AbstractVector, x_true::AbstractVector, y_obs::AbstractVector, z_est, x_est; fname::AbstractString, mp4_fname::Union{Nothing,AbstractString}=nothing, title_suffix = "", max_frames::Int = 120, fps::Int = 24)
    n = length(y_obs)
    step = max(1, ceil(Int, n / max_frames))
    anim = @animate for t in 1:step:n
        pz = plot(title = "Z up to t=$(t)" * (isempty(title_suffix) ? "" : " — " * title_suffix))
        plot!(pz, 1:t, z_true[1:t], label = "z", color = :orange)
        plot!(pz, 1:t, mean.(z_est[1:t]), ribbon = std.(z_est[1:t]), label = "ẑ", color = :teal)

        px = plot(title = "X up to t=$(t)" * (isempty(title_suffix) ? "" : " — " * title_suffix))
        plot!(px, 1:t, x_true[1:t], label = "x", color = :green)
        plot!(px, 1:t, mean.(x_est[1:t]), ribbon = std.(x_est[1:t]), label = "x̂", color = :violet)
        scatter!(px, 1:t, y_obs[1:t], label = "y", color = :red, ms = 2, alpha = 0.2)

        plot(pz, px, layout = @layout([ a; b ]))
    end
    @info "Saving GIF animation" fname
    gif(anim, fname, fps = fps)
    if mp4_fname !== nothing
        try
            @info "Saving MP4 animation" mp4_fname
            mp4(anim, mp4_fname, fps = fps)
        catch err
            @warn "MP4 generation failed (ffmpeg)" error = err
        end
    end
    return fname
end

function make_residuals_animation(y_obs::AbstractVector, x_est; fname::AbstractString, mp4_fname::Union{Nothing,AbstractString}=nothing, max_frames::Int = 120, fps::Int = 24)
    n = length(y_obs)
    step = max(1, ceil(Int, n / max_frames))
    anim = @animate for t in 1:step:n
        r = y_obs[1:t] .- mean.(x_est[1:t])
        p1 = plot(1:t, r, title = "Residuals over time (1..t=$(t))", label = "r")
        p2 = histogram(r, bins = min(40, max(10, round(Int, sqrt(t)))), normalize = true, title = "Residuals hist (1..t=$(t))", label = false)
        plot(p1, p2, layout = @layout([ a; b ]))
    end
    @info "Saving Residuals GIF" fname
    gif(anim, fname, fps = fps)
    if mp4_fname !== nothing
        try
            @info "Saving Residuals MP4" mp4_fname
            mp4(anim, mp4_fname, fps = fps)
        catch err
            @warn "Residuals MP4 generation failed (ffmpeg)" error = err
        end
    end
    return fname
end

function plot_state_errors(z_true, x_true, z_est, x_est)
    ez = z_true .- mean.(z_est)
    ex = x_true .- mean.(x_est)
    p1 = plot(ez, title = "Error z - mean(ẑ)", label = "e_z")
    p2 = plot(ex, title = "Error x - mean(x̂)", label = "e_x")
    return plot(p1, p2, layout = @layout([ a; b ]))
end

function plot_residuals(y_obs, x_est)
    r = y_obs .- mean.(x_est)
    p1 = plot(r, title = "Residuals y - mean(x̂)", label = "r")
    p2 = histogram(r, bins = 40, normalize = true, title = "Residuals histogram", label = false)
    return plot(p1, p2, layout = @layout([ a; b ]))
end

function plot_variance_trajectories(z_est, x_est)
    p1 = plot(std.(z_est) .^ 2, title = "Var ẑ over time", label = "var ẑ")
    p2 = plot(std.(x_est) .^ 2, title = "Var x̂ over time", label = "var x̂")
    return plot(p1, p2, layout = @layout([ a; b ]))
end

function _acf(values::AbstractVector{<:Real}, maxlag::Int)
    n = length(values)
    μ = mean(values)
    v = values .- μ
    denom = sum(abs2, v)
    acf = zeros(Float64, maxlag + 1)
    for lag in 0:maxlag
        num = sum(v[1:n-lag] .* v[1+lag:n])
        acf[lag + 1] = denom ≈ 0 ? 0.0 : num / denom
    end
    return acf
end

function plot_residual_acf(residuals::AbstractVector{<:Real}; maxlag::Int = 50)
    a = _acf(residuals, maxlag)
    lags = 0:maxlag
    p = bar(lags, a, title = "Residual ACF", label = "acf")
    hline!(p, [0.0], color = :black, label = false)
    return p
end

function plot_residual_qq(residuals::AbstractVector{<:Real})
    n = length(residuals)
    if n == 0
        return plot(title = "QQ plot (no data)")
    end
    r = (residuals .- mean(residuals)) ./ (std(residuals) + eps())
    sorted_r = sort(r)
    probs = ((1:n) .- 0.5) ./ n
    q_theory = quantile.(Ref(Normal(0, 1)), probs)
    p = scatter(q_theory, sorted_r, label = "quantiles", title = "QQ plot (residuals vs Normal)", xlabel = "Theoretical", ylabel = "Empirical")
    plot!(p, q_theory, q_theory, label = "y=x", color = :red)
    return p
end

function plot_coverage(z_true::AbstractVector{<:Real}, x_true::AbstractVector{<:Real}, z_est, x_est; alpha = 0.05)
    zμ = mean.(z_est)
    zσ = std.(z_est)
    xμ = mean.(x_est)
    xσ = std.(x_est)
    z_lower = zμ .- quantile(Normal(), 1 - alpha/2) .* zσ
    z_upper = zμ .+ quantile(Normal(), 1 - alpha/2) .* zσ
    x_lower = xμ .- quantile(Normal(), 1 - alpha/2) .* xσ
    x_upper = xμ .+ quantile(Normal(), 1 - alpha/2) .* xσ
    z_cov = (z_true .>= z_lower) .& (z_true .<= z_upper)
    x_cov = (x_true .>= x_lower) .& (x_true .<= x_upper)
    pz = plot(z_cov .* 1.0, title = @sprintf("z coverage (%.1f%%)", 100 * mean(z_cov)), label = "covered(1)/not(0)")
    px = plot(x_cov .* 1.0, title = @sprintf("x coverage (%.1f%%)", 100 * mean(x_cov)), label = "covered(1)/not(0)")
    return plot(pz, px, layout = @layout([ a; b ]))
end

end # module


