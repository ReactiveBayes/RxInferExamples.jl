### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ a8856ac8-f2c1-4567-bdcb-bb3458fd5365
begin
using Pkg
Pkg.activate(".")
using GLMakie
using FileIO
using PlutoUI
include("utils.jl")
using Colors
end

# ╔═╡ e0c0e102-bad7-11f0-1ee1-f38838459240
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(80px, 5%);
    	padding-right: max(80px, 5%);
	}
</style>
"""

# ╔═╡ 8ca6bb25-2256-4055-92a8-fadaf52f0028
function plot_posterior(
    results,
    i::Int,
    dim::Int,
    symb::Symbol;
    DPI::Int = 300,
    t_start::Int = 1,
    t_end::Union{Int, Nothing} = nothing,
    show_cov::Bool = true,
    ref::Union{Nothing, AbstractVector} = nothing,
    ref_label::String = "reference",
    ref_color = :red
)
    means = results["$(string(symb))_means"]
    covs  = results["$(string(symb))_covs"]

    D, T_ai, N_ai = size(means)
    @assert 1 ≤ dim ≤ D "dim out of range (1:$D)"
    @assert 1 ≤ i ≤ N_ai "i out of range (1:$N_ai)"

    t_end === nothing && (t_end = T_ai)
    @assert 1 ≤ t_start ≤ t_end ≤ T_ai "Invalid t_start/t_end range."
    timesteps = t_start:t_end

    μs = means[dim, timesteps, i]
    σs = sqrt.([covs[dim, dim, t, i] for t in timesteps])

    # Check reference validity
    if ref !== nothing
        @assert length(ref) == length(timesteps) "Reference vector must have same length as timesteps ($length(timesteps))"
    end

    # === Plot ===
    fig = Figure(size = (4 * DPI, DPI))
    ax = Axis(fig[1, 1],
        title = "Posterior over $symb (dim = $dim, i = $i)",
        xlabel = "Planning step (t)",
        ylabel = "Value"
    )

    # Mean and uncertainty
    lines!(ax, timesteps, μs, color = (:blue, 0.9), linewidth = 2)
	scatter!(ax, timesteps, μs, color = :blue, markersize = 5, label = "mean")
    if show_cov
        band!(ax, timesteps, μs .- σs, μs .+ σs, color = (:skyblue, 0.4), label = "±1σ")
    end

    # Optional reference line
    if ref !== nothing
        lines!(ax, timesteps, ref, color = ref_color, linestyle = :dash, linewidth = 2, label = ref_label)
    end

    axislegend(ax)
    return fig
end


# ╔═╡ 0923065d-e846-4d8e-9589-1d3f9044bb87
function plot_posterior_predictive(
    results,
    varname::Symbol,
    dim::Int;                    # which dimension to plot
    DPI::Int = 300,
    k_start::Int = 1,
    k_end::Union{Int, Nothing} = nothing,
    show_cov::Bool = true,
    show_target::Bool = false
)
    # === Construct variable names ===
    obs_key     = varname === :x ? "agent_x" : string(varname)
    mean_key    = string(varname, "_means")
    cov_key     = string(varname, "_covs")
    target_key  = "x_target"

    # === Extract data ===
    x_true   = results[obs_key]                   # (D, N)
    x_means  = results[mean_key][:, 2, :]         # predictive means
    x_covs   = results[cov_key][:, :, 2, :]       # predictive covariances
    x_target = haskey(results, target_key) ? results[target_key] : nothing

    D, N = size(x_true)
    @assert 1 ≤ dim ≤ D "dim out of range (1:$D)"

    if k_end === nothing
        k_end = N - 1
    end
    @assert 1 ≤ k_start ≤ k_end < N "Invalid k_start/k_end range."
    timesteps = k_start:k_end

    # === Extract data for selected dim ===
    μs = x_means[dim, timesteps]
    x_next = x_true[dim, timesteps .+ 1]

    # === Create figure ===
    fig = Figure(size = (4 * DPI, DPI))
    ax = Axis(fig[1, 1],
        title = "Predictive posterior vs. true obs (var = $(varname), dim = $dim)",
        xlabel = "Time step (t)",
        ylabel = "Value"
    )

    # Predictive mean ±1σ
    if show_cov
        σs = sqrt.([x_covs[dim, dim, t] for t in timesteps])
        band!(ax, timesteps, μs .- σs, μs .+ σs, color = (:skyblue, 0.4))
    end
    lines!(ax, timesteps, μs, color = (:blue, 0.9), linewidth = 2)
	scatter!(ax, timesteps, μs, color = :blue, markersize = 5, label = "predictive mean")

    # True next observation
    scatter!(ax, timesteps, x_next, color = :red, markersize = 5, label = "true next obs")

    # Optional target line
    if show_target && varname == :x && x_target !== nothing
        hlines!(ax, [x_target[dim]], color = :green, linestyle = :dash, linewidth = 2, label = "target")
    end

    axislegend(ax)
    return fig
end


# ╔═╡ fc02fbea-402f-4d0d-9ed3-833010e8ce38
DPI=300

# ╔═╡ f94444fe-2126-4448-a157-a249d02933dc
begin
DIR_RESULTS = "results"
filenames = filter(f -> endswith(f, ".jld2"), readdir(DIR_RESULTS; join=true))
@bind data Select(
    filenames,
    default = isempty(filenames) ? nothing : first(filenames),
)
end

# ╔═╡ 926a2720-2d3c-4598-a04b-834ee0fa38ea
results = load(data)

# ╔═╡ 9ae8074f-9fc7-468f-9590-185b72120e29
T_ai, N_ai = size(results["agent_f"])

# ╔═╡ 3d7c7676-95ee-4b5b-8fa5-a5750c123b18
Fa, Ff, Fg, height = create_physics(
    engine_force_limit = results["engine_force_limit"],
    friction_coefficient = results["friction_coefficient"]
)

# ╔═╡ 752fb6b9-17da-4aaf-b4d9-1ca041c4e0f9
function plot_interaction(results; k_start::Int=1, DPI::Int=300, markersize::Int=20)
	
	fig = Figure(size = (4*DPI, DPI))  # increase height for third plot
	ax1 = Axis(fig[1, 1], title = "Active inference results", xlabel = "x", ylabel = "y")
	lines!(ax1, results["valley_x"], results["valley_y"], color = :black)
	scatter!(ax1, [results["x_target"][1]], [height(results["x_target"][1])], color = :red, label = "goal", markersize=markersize)
	scatter!(ax1, [results["agent_x"][1, k_start]], [height(results["agent_x"][1, k_start])], color = :green, label = "car", markersize=markersize)
	alphas = 0.5 .* exp.(-range(0, 1, length=T_ai))
	colors = RGBA.(0.4, 0.0, 0.4, alphas)
	scatter!(ax1, results["agent_f"][:, k_start], height.(results["agent_f"][:, k_start]), label = "pred_y", color = colors, markersize=markersize)

	xlims!(ax1, minimum(results["agent_f"][:, k_start])-0.1, maximum(results["agent_f"][:, k_start])+0.1)

	axislegend(ax1, position=:lt)
	fig
end

# ╔═╡ ff9135a8-ee6d-4778-8ffa-1dcb7d7bcdc3
function g(s_t_min)
	s_t = similar(s_t_min) # Next state
	s_t[2] = s_t_min[2] + Fg(s_t_min[1]) + Ff(s_t_min[2]) # Update velocity
	s_t[1] = s_t_min[1] + s_t[2] # Update position
	return s_t
end

# ╔═╡ 305cb7e7-2a44-46c1-befd-20208b6a7911
begin
results["u_h_k"] = zeros(results["var_dims"][:u_h_k], N_ai)
results["u_h_k"][2,:] .= Fa.(results["agent_a"])
results["u_h_k"]
end

# ╔═╡ 6e42c3a7-bde2-47db-8117-34746ff76f39
 results["s_g_k"] = hcat([g(results["agent_x"][:, t]) for t in 1:size(results["agent_x"], 2)]...)

# ╔═╡ 2d678e4c-f785-45e3-a1e6-8b0025da2d8a
results["s"] = results["u_h_k"] + results["s_g_k"]

# ╔═╡ 5839947b-42ba-4225-aa1f-046ca0c49abd
results["x"] = results["s"]

# ╔═╡ fc7fcc33-a390-4e92-adeb-c10d0d9ef031
@bind show_cov Select([true, false], default=false)

# ╔═╡ 36d53fce-bde7-424b-a802-64f53251d4c2
@bind k_start PlutoUI.Slider(1:N_ai, default=1, show_value=true)

# ╔═╡ a06d4f31-e728-45a0-8b7c-e4303609265a
@bind k_step PlutoUI.Slider(1:N_ai-k_start-1, default=N_ai-k_start-1, show_value=true)

# ╔═╡ 8c324a52-b8e9-4fd1-979b-ceea8e9143f7
k_end = k_start + k_step

# ╔═╡ b0677384-72c6-4827-97f8-3dee80f0b2bf
@bind t_start PlutoUI.Slider(1:N_ai, default=1, show_value=true)

# ╔═╡ 3a702c8b-9293-44cd-ae90-adc79053d13a
@bind t_step PlutoUI.Slider(1:N_ai-t_start, default=T_ai-t_start, show_value=true)

# ╔═╡ 4750b48a-5593-47df-b3c5-aa725ce19488
t_end = t_start + t_step

# ╔═╡ 0f75bb84-5866-4649-b91a-084bbbe2f6b1
@bind dim_u PlutoUI.Slider(1:results["var_dims"][:u], show_value=true)

# ╔═╡ 1ce67fb9-3d7e-4e7a-9409-aeeac71f0468
plot_posterior(results, k_start, dim_u, :u, t_start=t_start, t_end=t_end, show_cov=show_cov)

# ╔═╡ 0c8869b1-df32-4df2-8ef6-292221c4525e
@bind dim_u_h_k PlutoUI.Slider(1:results["var_dims"][:u_h_k], show_value=true)

# ╔═╡ 7a4554f8-f520-4e86-ae7d-dc096d24ba9f
plot_posterior(results, k_start, dim_u_h_k, :u_h_k, t_start=t_start, t_end=t_end, show_cov=show_cov)

# ╔═╡ 0b815006-dfef-4837-a427-35fca4c42d83
plot_posterior_predictive(results, :u_h_k, dim_u_h_k, k_start=k_start, k_end=k_end, show_cov=show_cov)

# ╔═╡ 7bee6910-16eb-4ce6-8448-29d87f34be4a
@bind dim_s_g_k PlutoUI.Slider(1:results["var_dims"][:s_g_k], show_value=true)

# ╔═╡ 331bc8e3-1807-4aba-87ce-49efe8d32410
plot_posterior(results, k_start, dim_s_g_k, :s_g_k, t_start=t_start, t_end=t_end, show_cov=show_cov)

# ╔═╡ ee01ab14-0346-4942-8623-4a2790df380b
plot_posterior_predictive(results, :s_g_k, dim_s_g_k, k_start=k_start, k_end=k_end, show_cov=show_cov)

# ╔═╡ df876e44-dc34-43f5-84f1-d69b627e1a08
@bind dim_u_s_sum PlutoUI.Slider(1:results["var_dims"][:u_s_sum], show_value=true)

# ╔═╡ 3c29e5ea-593f-409b-a51c-f19242e0ebfc
plot_posterior(results, k_start, dim_u_s_sum, :u_s_sum, t_start=t_start, t_end=t_end, show_cov=show_cov)

# ╔═╡ b52a7533-1c7e-43a0-bdbe-8dd74ba60243
@bind dim_s PlutoUI.Slider(1:results["var_dims"][:s], show_value=true)

# ╔═╡ 7cd4a5fc-c7a1-4636-af8b-d82a80e120fe
plot_posterior(results, k_start, dim_s, :s, t_start=t_start, t_end=t_end, show_cov=show_cov)

# ╔═╡ 7ee4fd22-3f99-435b-8dc7-d8db5bec2813
plot_posterior_predictive(results, :s, dim_s, k_start=k_start, k_end=k_end, show_cov=show_cov)

# ╔═╡ def4a2c0-8649-4b6a-910e-36147af66936
@bind dim_x PlutoUI.Slider(1:results["var_dims"][:x], show_value=true)

# ╔═╡ 4ebe0716-de8b-4de3-a1e6-ce91257a213f
begin
	dims = Dict(k => 0 for k in keys(results["var_dims"]))
	dims[:u] = dim_u
	dims[:u_h_k] = dim_u_h_k
	dims[:s_g_k] = dim_s_g_k
	dims[:u_s_sum] = dim_u_s_sum
	dims[:s] = dim_s
	dims[:x] = dim_x
	dims
end

# ╔═╡ 21a6e91d-2400-4513-89c1-fa982d99d334
plot_posterior(results, k_start, dim_x, :x, t_start=t_start, t_end=t_end, show_cov=show_cov)

# ╔═╡ 77f0f51b-b47d-45fe-ac8b-acc5849a446a
plot_posterior_predictive(results, :x, dim_x, k_start=k_start, k_end=k_end, show_cov=show_cov, show_target=true)

# ╔═╡ 694c4ad4-c165-4761-8e16-6b0361e53309
results["s"]

# ╔═╡ 01a9370b-5280-48e8-8dfd-d2aa8ec5322a
results["agent_x"]

# ╔═╡ 2359e6b1-fa56-4da5-8bbe-8540955801fc
plot_interaction(results, k_start=k_start)

# ╔═╡ 52615ec8-098b-440f-89c8-b7e976634f6d
results["agent_x"][1, k_start]

# ╔═╡ Cell order:
# ╠═e0c0e102-bad7-11f0-1ee1-f38838459240
# ╠═a8856ac8-f2c1-4567-bdcb-bb3458fd5365
# ╠═8ca6bb25-2256-4055-92a8-fadaf52f0028
# ╠═0923065d-e846-4d8e-9589-1d3f9044bb87
# ╠═752fb6b9-17da-4aaf-b4d9-1ca041c4e0f9
# ╠═ff9135a8-ee6d-4778-8ffa-1dcb7d7bcdc3
# ╠═fc02fbea-402f-4d0d-9ed3-833010e8ce38
# ╠═f94444fe-2126-4448-a157-a249d02933dc
# ╠═926a2720-2d3c-4598-a04b-834ee0fa38ea
# ╠═4ebe0716-de8b-4de3-a1e6-ce91257a213f
# ╠═9ae8074f-9fc7-468f-9590-185b72120e29
# ╠═3d7c7676-95ee-4b5b-8fa5-a5750c123b18
# ╠═305cb7e7-2a44-46c1-befd-20208b6a7911
# ╠═6e42c3a7-bde2-47db-8117-34746ff76f39
# ╠═2d678e4c-f785-45e3-a1e6-8b0025da2d8a
# ╠═5839947b-42ba-4225-aa1f-046ca0c49abd
# ╠═fc7fcc33-a390-4e92-adeb-c10d0d9ef031
# ╠═36d53fce-bde7-424b-a802-64f53251d4c2
# ╠═a06d4f31-e728-45a0-8b7c-e4303609265a
# ╠═8c324a52-b8e9-4fd1-979b-ceea8e9143f7
# ╠═b0677384-72c6-4827-97f8-3dee80f0b2bf
# ╠═3a702c8b-9293-44cd-ae90-adc79053d13a
# ╠═4750b48a-5593-47df-b3c5-aa725ce19488
# ╠═0f75bb84-5866-4649-b91a-084bbbe2f6b1
# ╠═1ce67fb9-3d7e-4e7a-9409-aeeac71f0468
# ╠═0c8869b1-df32-4df2-8ef6-292221c4525e
# ╠═7a4554f8-f520-4e86-ae7d-dc096d24ba9f
# ╠═0b815006-dfef-4837-a427-35fca4c42d83
# ╠═7bee6910-16eb-4ce6-8448-29d87f34be4a
# ╠═331bc8e3-1807-4aba-87ce-49efe8d32410
# ╠═ee01ab14-0346-4942-8623-4a2790df380b
# ╠═df876e44-dc34-43f5-84f1-d69b627e1a08
# ╠═3c29e5ea-593f-409b-a51c-f19242e0ebfc
# ╠═b52a7533-1c7e-43a0-bdbe-8dd74ba60243
# ╠═7cd4a5fc-c7a1-4636-af8b-d82a80e120fe
# ╠═7ee4fd22-3f99-435b-8dc7-d8db5bec2813
# ╠═def4a2c0-8649-4b6a-910e-36147af66936
# ╠═21a6e91d-2400-4513-89c1-fa982d99d334
# ╠═77f0f51b-b47d-45fe-ac8b-acc5849a446a
# ╠═694c4ad4-c165-4761-8e16-6b0361e53309
# ╠═01a9370b-5280-48e8-8dfd-d2aa8ec5322a
# ╠═2359e6b1-fa56-4da5-8bbe-8540955801fc
# ╠═52615ec8-098b-440f-89c8-b7e976634f6d
