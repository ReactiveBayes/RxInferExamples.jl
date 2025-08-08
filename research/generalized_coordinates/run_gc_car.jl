ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

using Pkg
Pkg.activate(@__DIR__)

using RxInfer, Random, LinearAlgebra, Distributions, StableRNGs, Plots

# Resolve modules locally to avoid global redefinitions when included multiple times
function _resolve_modules()
    try
        push!(LOAD_PATH, @__DIR__)
        @eval using GeneralizedCoordinatesExamples
        return (
            GeneralizedCoordinatesExamples.GCUtils,
            GeneralizedCoordinatesExamples.GCModel,
            GeneralizedCoordinatesExamples.GCViz,
        )
    catch
        include(joinpath(@__DIR__, "src", "GCUtils.jl")); @eval using .GCUtils
        include(joinpath(@__DIR__, "src", "GCModel.jl")); @eval using .GCModel
        include(joinpath(@__DIR__, "src", "GCViz.jl")); @eval using .GCViz
        return (GCUtils, GCModel, GCViz)
    end
end

let (GCUtils, GCModel, GCViz) = _resolve_modules()
    # NOTE: Per requirements, no algorithmic fallbacks are permitted; run RxInfer end-to-end.

    # Config
    rng = StableRNG(42)
    n   = 2000
    dt  = 0.1
    σ_a = 0.25
    σ_obs_pos = 0.5
    σ_obs_vel = NaN  # set to e.g. 0.7 to also observe velocity

    # Outputs directory
    outdir = joinpath(@__DIR__, "outputs")
    isdir(outdir) || mkpath(outdir)

    # Generate data
    x_true, y = GCUtils.generate_gc_car_data(rng, n, dt; σ_a=σ_a, σ_obs_pos=σ_obs_pos, σ_obs_vel=σ_obs_vel)

    # Build model matrices
    A, _, Qd = GCUtils.constant_acceleration_ABQ(dt; σ_a=σ_a)
    Q = Matrix(Qd)
    B = isnan(σ_obs_vel) ? [1.0 0.0 0.0] : [1.0 0.0 0.0; 0.0 1.0 0.0]
    R = isnan(σ_obs_vel) ? Matrix(Diagonal([σ_obs_pos^2])) : Matrix(Diagonal([σ_obs_pos^2, σ_obs_vel^2]))

    x0_mean = Float64[0.0, 0.0, 0.0]
    x0_cov  = Matrix(Diagonal(fill(100.0, 3)))

    # Initialization for RxInfer v4
    init = @initialization begin
        q(x) = MvNormalMeanCovariance(x0_mean, x0_cov)
    end

    # Inference (RxInfer only) using history collection
    result = infer(
        model = GCModel.gc_car_model(A=A, B=B, Q=Q, R=R, x0_mean=x0_mean, x0_cov=x0_cov),
        data  = (y = y,),
        constraints = GCModel.make_constraints(),
        initialization = init,
        keephistory = n,
        historyvars = (x = KeepLast(),),
        iterations = 100,
        autostart = true,
        free_energy = true,
        allow_node_contraction = false,
        options = (limit_stack_depth = 100,),
        warn = false,
        catch_exception = false,
        showprogress = false,
    )

    xmarginals_any = result.posteriors[:x]
    xmarginals = xmarginals_any isa Vector{<:Vector} ? (all(length.(xmarginals_any) .== 1) ? getindex.(xmarginals_any, 1) : reduce(vcat, xmarginals_any)) : xmarginals_any

    # Per-time FE terms (approximate, Gaussian)
    fe = GCUtils.free_energy_timeseries(y, xmarginals, A, B, Q, R, x0_mean, x0_cov)
    println("Final total FE (approx): ", fe.total[end])

    # Save posterior means/vars
    open(joinpath(outdir, "gc_posterior_summary.csv"), "w") do io
        header = "t,μ_pos,μ_vel,μ_acc,σ2_pos,σ2_vel,σ2_acc"
        println(io, header)
        for t in 1:n
            μt = mean(xmarginals[t])
            Σraw = try
                cov(xmarginals[t])
            catch
                var(xmarginals[t])
            end
            vdiag = Σraw isa AbstractVector ? Σraw : diag(Σraw)
            println(io, string(t, ",", μt[1], ",", μt[2], ",", μt[3], ",", vdiag[1], ",", vdiag[2], ",", vdiag[3]))
        end
    end

    # Save RxInfer free energy trajectory (per-iteration)
    try
        fe_path = joinpath(outdir, "rxinfer_free_energy.csv")
        open(fe_path, "w") do io
            println(io, "iteration,free_energy,delta,relative_delta")
            prev = nothing
            for (i, F) in enumerate(result.free_energy)
                if prev === nothing
                    println(io, string(i, ",", F, ",,,"))
                else
                    d = F - prev
                    rd = prev == 0 ? NaN : d / abs(prev)
                    println(io, string(i, ",", F, ",", d, ",", rd))
                end
                prev = F
            end
        end
    catch err
        @warn "Saving rxinfer_free_energy.csv failed" err
    end

    # Plots
    if isnan(σ_obs_vel)
        fig_pos = GCViz.plot_pos(x_true, y, xmarginals)
        savefig(fig_pos, joinpath(outdir, "gc_pos.png"))
    else
        fig_pv = GCViz.plot_pos_vel(x_true, y, xmarginals)
        savefig(fig_pv, joinpath(outdir, "gc_pos_vel.png"))
    end

    fig_fe = GCViz.plot_free_energy_terms(fe.obs_term, fe.dyn_term, fe.total)
    savefig(fig_fe, joinpath(outdir, "gc_free_energy_terms.png"))

    fig_dash = GCViz.summary_dashboard(x_true, y, xmarginals, fe.obs_term, fe.dyn_term, fe.total)
    savefig(fig_dash, joinpath(outdir, "gc_dashboard.png"))

    # Additional visualizations
    savefig(GCViz.plot_states(x_true, xmarginals), joinpath(outdir, "gc_states.png"))
    savefig(GCViz.plot_residuals(x_true, y, xmarginals, B), joinpath(outdir, "gc_residuals.png"))
    savefig(GCViz.plot_errors(x_true, xmarginals), joinpath(outdir, "gc_errors.png"))
    savefig(GCViz.plot_scatter_true_vs_inferred(x_true, xmarginals), joinpath(outdir, "gc_scatter_true_vs_inferred.png"))
    savefig(GCViz.plot_rmse(x_true, xmarginals), joinpath(outdir, "gc_rmse.png"))
    savefig(GCViz.plot_coverage(x_true, xmarginals), joinpath(outdir, "gc_coverage.png"))
    savefig(GCViz.plot_residual_hist(y, xmarginals, B), joinpath(outdir, "gc_residual_hist.png"))
    savefig(GCViz.plot_fe_cumsum(fe.total), joinpath(outdir, "gc_fe_cumsum.png"))
    # Free energy across iterations (exact)
    try
        savefig(GCViz.plot_fe_iterations(result.free_energy), joinpath(outdir, "gc_fe_iterations.png"))
    catch err
        @warn "Plotting gc_fe_iterations.png failed" err
    end

    savefig(GCViz.plot_y_fit(y, xmarginals, B, R), joinpath(outdir, "gc_y_fit.png"))
    savefig(GCViz.plot_stdres_hist(y, xmarginals, B, R), joinpath(outdir, "gc_stdres_hist.png"))
    savefig(GCViz.plot_stdres_qq(y, xmarginals, B, R), joinpath(outdir, "gc_stdres_qq.png"))
    savefig(GCViz.plot_stdres_acf(y, xmarginals, B, R), joinpath(outdir, "gc_stdres_acf.png"))
    savefig(GCViz.plot_mse_time(x_true, xmarginals), joinpath(outdir, "gc_mse_time.png"))
    savefig(GCViz.plot_state_coverage_time(x_true, xmarginals), joinpath(outdir, "gc_state_coverage_time.png"))

    # Save CSV report (time, obs_term, prior_term, dyn_term, total)
    open(joinpath(outdir, "gc_free_energy_timeseries.csv"), "w") do io
        println(io, "t,obs_term,prior_term,dyn_term,total")
        for t in 1:n
            println(io, string(t, ",", fe.obs_term[t], ",", fe.prior_term[t], ",", fe.dyn_term[t], ",", fe.total[t]))
        end
    end

    # Optional per-dimension observation contributions if available
    try
        if length(first(y)) >= 1
            open(joinpath(outdir, "gc_free_energy_obs_dim_terms.csv"), "w") do io
                nd = length(first(y))
                header = join(["t"; ["obs_dim_" * string(d) for d in 1:nd]], ",")
                println(io, header)
                for t in 1:n
                    row = join([string(t); string.(fe.obs_dim_terms[t])], ",")
                    println(io, row)
                end
            end
        end
    catch err
        @warn "Saving gc_free_energy_obs_dim_terms.csv failed" err
    end

    # Animations
    try
        if isnan(σ_obs_vel)
            GCViz.make_animation_pos(x_true, y, xmarginals, joinpath(outdir, "gc_position_animation.gif"))
        end
        # Use make_animation_states when environment supports long GIF generation
        GCViz.make_animation_states(x_true, xmarginals, joinpath(outdir, "gc_states_animation.gif"))
    catch err
        @warn "Animation failed" err
    end
end
