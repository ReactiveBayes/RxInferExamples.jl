module InfiniteDataStreamViz

using Plots

export plot_hidden_and_obs, plot_estimates, save_gif,
       animate_estimates, plot_tau, plot_overlay_means,
       plot_scatter_static_vs_realtime, plot_residuals,
       animate_free_energy, animate_composed_estimates_fe,
       animate_overlay_means

plot_hidden_and_obs(history::AbstractVector{<:Real}, observations::AbstractVector{<:Real}; size=(1000,300)) = begin
    p = plot(size=size)
    plot!(p, 1:length(history), history, label="Hidden signal")
    scatter!(p, 1:length(observations), observations; ms=4, alpha=0.7, label="Observation")
end

plot_estimates(μ::AbstractVector{<:Real}, σ2::AbstractVector{<:Real}, history::AbstractVector{<:Real}, observations::AbstractVector{<:Real}; upto::Int, size=(1000,300)) = begin
    p = plot(1:upto, μ[1:upto]; ribbon=σ2[1:upto], label="Estimation")
    plot!(p, history[1:upto]; label="Real states")
    scatter!(p, observations[1:upto]; ms=2, label="Observations")
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
    p1 = plot_estimates(μ, σ2, history, observations; upto=i, size=(900,300))
    p2 = plot(free_energy[1:i]; label="Bethe Free Energy (avg)", xlabel="t", size=(700,300))
    plot(p1, p2; layout=(1,2), size=(1600,320))
end

animate_overlay_means(truth::AbstractVector{<:Real}, μ_static::AbstractVector{<:Real}, μ_rt::AbstractVector{<:Real}; stride::Int=5) = @animate for i in 1:stride:min(length(truth), min(length(μ_static), length(μ_rt)))
    plot_overlay_means(truth, μ_static, μ_rt; upto=i)
end

plot_tau(tau_mean::AbstractVector{<:Real}; label::AbstractString="E[τ]", xlabel::AbstractString="t", ylabel::AbstractString="precision", size=(800,300)) = begin
    plot(tau_mean; label=label, xlabel=xlabel, ylabel=ylabel, size=size)
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

