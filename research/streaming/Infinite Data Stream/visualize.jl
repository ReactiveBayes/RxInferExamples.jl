module InfiniteDataStreamViz

using Plots

export plot_hidden_and_obs, plot_estimates, save_gif,
       animate_estimates, plot_tau, plot_overlay_means,
       plot_scatter_static_vs_realtime, plot_residuals,
       animate_free_energy, animate_composed_estimates_fe,
       animate_overlay_means,
       plot_fe_comparison, animate_comparison_static_vs_realtime,
       plot_tau_comparison

plot_hidden_and_obs(history::AbstractVector{<:Real}, observations::AbstractVector{<:Real}; size=(1000,300)) = begin
    p = plot(size=size)
    plot!(p, 1:length(history), history, label="Hidden signal")
    scatter!(p, 1:length(observations), observations; ms=4, alpha=0.7, label="Observation")
end

plot_estimates(μ::AbstractVector{<:Real}, σ2::AbstractVector{<:Real}, history::AbstractVector{<:Real}, observations::AbstractVector{<:Real}; upto::Int, size=(1000,300)) = begin
    p = plot(1:upto, μ[1:upto]; ribbon=σ2[1:upto], label="Estimation")
    plot!(p, history[1:upto]; label="Real states")
    scatter!(p, observations[1:upto]; ms=2, label="Observations")
    xlims!(p, (1, upto))
    plot(p; size=size, legend=:bottomright)
end

save_gif(anim, path::AbstractString; fps::Int=24) = gif(anim, path; fps=fps, show_msg=false)

animate_estimates(μ::AbstractVector{<:Real}, σ2::AbstractVector{<:Real}, history::AbstractVector{<:Real}, observations::AbstractVector{<:Real}; stride::Int=5, size=(1000,300)) = @animate for i in 1:stride:length(μ)
    plot_estimates(μ, σ2, history, observations; upto=i, size=size)
end

animate_free_energy(free_energy::AbstractVector{<:Real}; stride::Int=5, size=(800,300)) = @animate for i in 1:stride:length(free_energy)
    plot(free_energy[1:i]; label="Bethe Free Energy (avg)", xlabel="t", size=size)
end

animate_composed_estimates_fe(μ::AbstractVector{<:Real}, σ2::AbstractVector{<:Real}, history::AbstractVector{<:Real}, observations::AbstractVector{<:Real}, free_energy::AbstractVector{<:Real}; stride::Int=5) = @animate for i in 1:stride:min(length(μ), length(free_energy))
    p1 = plot_estimates(μ, σ2, history, observations; upto=i, size=(1000,300))
    xlims!(p1, (1, i))
    p2 = plot(free_energy[1:i]; label="Bethe Free Energy (avg)", xlabel="t", size=(1000,300))
    xlims!(p2, (1, i))
    plot(p1, p2; layout=(2,1), link=:x, size=(1000,620))
end

animate_overlay_means(truth::AbstractVector{<:Real}, μ_static::AbstractVector{<:Real}, μ_rt::AbstractVector{<:Real}; stride::Int=5) = @animate for i in 1:stride:min(length(truth), min(length(μ_static), length(μ_rt)))
    plot_overlay_means(truth, μ_static, μ_rt; upto=i)
end

plot_tau(tau_mean::AbstractVector{<:Real}; label::AbstractString="E[τ]", xlabel::AbstractString="t", ylabel::AbstractString="precision", size=(800,300)) = begin
    plot(tau_mean; label=label, xlabel=xlabel, ylabel=ylabel, size=size)
end

# Side-by-side comparisons for static vs realtime

"""
    plot_fe_comparison(fe_static, fe_rt; upto=min(length(fe_static), length(fe_rt)))

Return a 2-panel plot with Bethe Free Energy time series for static vs realtime.
Includes legends and captions.
"""
plot_fe_comparison(fe_static::AbstractVector{<:Real}, fe_rt::AbstractVector{<:Real}; upto::Int=min(length(fe_static), length(fe_rt)), size=(1000,300)) = begin
    p = plot(fe_static[1:upto]; label="Static: Bethe Free Energy (avg)", xlabel="t", size=size, color=:blue)
    plot!(p, fe_rt[1:upto]; label="Realtime: Bethe Free Energy (avg)", color=:orange)
    title!(p, "Bethe Free Energy comparison")
    plot(p; legend=:topright)
end

"""
    animate_comparison_static_vs_realtime(truth, μ_static, σ2_static, μ_rt, σ2_rt, fe_static, fe_rt; stride=5)

Animated side-by-side visualization: top row overlays truth, static μ, realtime μ; bottom row compares FE.
"""
animate_comparison_static_vs_realtime(truth::AbstractVector{<:Real},
                                      μ_static::AbstractVector{<:Real}, σ2_static::AbstractVector{<:Real},
                                      μ_rt::AbstractVector{<:Real}, σ2_rt::AbstractVector{<:Real},
                                      fe_static::AbstractVector{<:Real}, fe_rt::AbstractVector{<:Real};
                                      stride::Int=5) = @animate for i in 1:stride:min(length(truth), length(μ_static), length(μ_rt), length(fe_static), length(fe_rt))
    p_overlay = plot(truth[1:i]; label="truth", color=:black, size=(1000,300))
    plot!(p_overlay, μ_static[1:i]; ribbon=σ2_static[1:i], label="static μ ± σ", color=:blue)
    plot!(p_overlay, μ_rt[1:i]; ribbon=σ2_rt[1:i], label="realtime μ ± σ", color=:orange)
    xlims!(p_overlay, (1, i))
    title!(p_overlay, "State estimates")

    p_fe = plot(fe_static[1:i]; label="static FE", xlabel="t", size=(1000,300), color=:blue)
    plot!(p_fe, fe_rt[1:i]; label="realtime FE", color=:orange)
    xlims!(p_fe, (1, i))
    title!(p_fe, "Bethe Free Energy (avg)")
    plot(p_overlay, p_fe; layout=(2,1), link=:x, size=(1000,620))
end

"""
    plot_tau_comparison(tau_static, tau_rt; upto=min(length(tau_static), length(tau_rt)))

Compare expected precision τ over time for static vs realtime.
"""
plot_tau_comparison(tau_static::AbstractVector{<:Real}, tau_rt::AbstractVector{<:Real}; upto::Int=min(length(tau_static), length(tau_rt)), size=(1000,300)) = begin
    p = plot(tau_static[1:upto]; label="static E[τ]", xlabel="t", ylabel="precision (τ)", size=size, color=:blue)
    plot!(p, tau_rt[1:upto]; label="realtime E[τ]", color=:orange)
    title!(p, "Observation precision τ (inverse variance)")
    plot(p; legend=:topright)
end

plot_overlay_means(truth::AbstractVector{<:Real}, μ_static::AbstractVector{<:Real}, μ_rt::AbstractVector{<:Real}; upto::Int=min(length(truth), min(length(μ_static), length(μ_rt))), size=(1000,300)) = begin
    p = plot(truth[1:upto]; label="truth", color=:black)
    plot!(p, μ_static[1:upto]; label="static μ", color=:blue)
    plot!(p, μ_rt[1:upto]; label="realtime μ", color=:orange)
    plot(p; size=size, legend=:bottomright)
end

plot_scatter_static_vs_realtime(μ_static::AbstractVector{<:Real}, μ_rt::AbstractVector{<:Real}; upto::Int=min(length(μ_static), length(μ_rt)), size=(600,600)) = begin
    p = scatter(μ_static[1:upto], μ_rt[1:upto]; ms=3, alpha=0.7, label="points", xlabel="static μ", ylabel="realtime μ")
    plot!(p, [minimum(μ_static[1:upto]); maximum(μ_static[1:upto])], [minimum(μ_static[1:upto]); maximum(μ_static[1:upto])]; label="y=x", color=:gray, lw=1.5)
    plot(p; size=size, legend=:bottomright)
end

plot_residuals(truth::AbstractVector{<:Real}, μ::AbstractVector{<:Real}; upto::Int=min(length(truth), length(μ)), size=(1000,300)) = begin
    r = truth[1:upto] .- μ[1:upto]
    p = plot(r; label="residual", xlabel="t", ylabel="truth - mean", size=size)
    hline!(p, [0.0]; color=:gray, lw=1, label="0")
    p
end

end # module

