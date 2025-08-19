module InfiniteDataStreamViz

using Plots

export plot_hidden_and_obs, plot_estimates, save_gif

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

end # module

