ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

using Pkg
Pkg.activate(@__DIR__)

using RxInfer, Random, LinearAlgebra, Distributions, StableRNGs, Plots

function _resolve_modules()
    try
        push!(LOAD_PATH, @__DIR__)
        @eval using GeneralizedCoordinatesExamples
        return (
            GeneralizedCoordinatesExamples.GCUtils,
            GeneralizedCoordinatesExamples.GCModel,
            GeneralizedCoordinatesExamples.GCViz,
            GeneralizedCoordinatesExamples.GCConfig,
            GeneralizedCoordinatesExamples.GCGenerators,
            GeneralizedCoordinatesExamples.GCReport,
        )
    catch
        include(joinpath(@__DIR__, "src", "GCUtils.jl")); @eval using .GCUtils
        include(joinpath(@__DIR__, "src", "GCModel.jl")); @eval using .GCModel
        include(joinpath(@__DIR__, "src", "GCViz.jl")); @eval using .GCViz
        include(joinpath(@__DIR__, "src", "GCConfig.jl")); @eval using .GCConfig
        include(joinpath(@__DIR__, "src", "GCGenerators.jl")); @eval using .GCGenerators
        include(joinpath(@__DIR__, "src", "GCReport.jl")); @eval using .GCReport
        return (GCUtils, GCModel, GCViz, GCConfig, GCGenerators, GCReport)
    end
end

let (GCUtils, GCModel, GCViz, GCConfig, GCGenerators, GCReport) = _resolve_modules()
    rng = StableRNG(7)
    outroot = joinpath(@__DIR__, "outputs")
    isdir(outroot) || mkpath(outroot)

    scenarios = [
        GCConfig.ScenarioConfig("pos_only_constant_5k", 5000, 0.1, 0.25, 0.5, NaN, :constant_accel, Dict{Symbol,Any}()),
        GCConfig.ScenarioConfig("pos_only_sin_f02_5k", 5000, 0.05, 0.1, 0.3, NaN, :sinusoid, Dict(:freq=>0.2, :amp=>5.0)),
        GCConfig.ScenarioConfig("pos_only_sin_mixed_5k", 5000, 0.04, 0.15, 0.3, NaN, :sinusoid_mixed, Dict(:freqs=>[0.1,0.35,0.6], :amps=>[2.0,1.0,0.8])),
        GCConfig.ScenarioConfig("pos_only_poly3_5k", 5000, 0.05, 0.2, 0.4, NaN, :poly, Dict(:degree=>3, :coeffs=>[0.0, 0.5, -0.01, 0.0001])),
        GCConfig.ScenarioConfig("pos_only_poly5_5k", 5000, 0.05, 0.25, 0.4, NaN, :poly, Dict(:degree=>5, :coeffs=>[0.0, 0.4, -0.01, 0.0001, -1e-6, 5e-9])),
        GCConfig.ScenarioConfig("pos_only_trend_osc_5k", 5000, 0.05, 0.2, 0.4, NaN, :trend_plus_osc, Dict(:degree=>2, :coeffs=>[0.0, 0.2, 0.0], :freq=>0.15, :amp=>1.5)),
    ]
    run_cfg = GCConfig.default_run_config()

    for scen in scenarios
        x_true, y, A, Q = GCGenerators.generate_scenario(rng, scen)
        B = isnan(scen.σ_obs_vel) ? [1.0 0.0 0.0] : [1.0 0.0 0.0; 0.0 1.0 0.0]
        R = isnan(scen.σ_obs_vel) ? Matrix(Diagonal([scen.σ_obs_pos^2])) : Matrix(Diagonal([scen.σ_obs_pos^2, scen.σ_obs_vel^2]))
        x0_mean = Float64[0.0, 0.0, 0.0]
        x0_cov  = Matrix(Diagonal(fill(100.0, 3)))

        init = @initialization begin
            q(x) = MvNormalMeanCovariance(x0_mean, x0_cov)
        end

        result = infer(
            model = GCModel.gc_car_model(A=A, B=B, Q=Q, R=R, x0_mean=x0_mean, x0_cov=x0_cov),
            data  = (y = y,),
            constraints = GCModel.make_constraints(),
            initialization = init,
            keephistory = scen.n,
            historyvars = (x = KeepLast(),),
            iterations = run_cfg.iterations,
            autostart = run_cfg.autostart,
            free_energy = run_cfg.free_energy,
            allow_node_contraction = false,
            options = (limit_stack_depth = 100,),
            warn = false,
            catch_exception = false,
            showprogress = false,
        )

        xm_any = result.posteriors[:x]
        xm = xm_any isa Vector{<:Vector} ? (all(length.(xm_any) .== 1) ? getindex.(xm_any, 1) : reduce(vcat, xm_any)) : xm_any
        fe = GCUtils.free_energy_timeseries(y, xm, A, B, Q, R, x0_mean, x0_cov)

        outdir = joinpath(outroot, scen.name)
        isdir(outdir) || mkpath(outdir)

        # Plots (all the same set as single run)
        savefig(GCViz.plot_states(x_true, xm), joinpath(outdir, "gc_states.png"))
        savefig(GCViz.plot_residuals(x_true, y, xm, B), joinpath(outdir, "gc_residuals.png"))
        savefig(GCViz.plot_errors(x_true, xm), joinpath(outdir, "gc_errors.png"))
        savefig(GCViz.plot_scatter_true_vs_inferred(x_true, xm), joinpath(outdir, "gc_scatter_true_vs_inferred.png"))
        savefig(GCViz.plot_rmse(x_true, xm), joinpath(outdir, "gc_rmse.png"))
        savefig(GCViz.plot_coverage(x_true, xm), joinpath(outdir, "gc_coverage.png"))
        savefig(GCViz.plot_residual_hist(y, xm, B), joinpath(outdir, "gc_residual_hist.png"))
        savefig(GCViz.plot_fe_cumsum(fe.total), joinpath(outdir, "gc_fe_cumsum.png"))
        savefig(GCViz.plot_y_fit(y, xm, B, R), joinpath(outdir, "gc_y_fit.png"))
        savefig(GCViz.plot_stdres_hist(y, xm, B, R), joinpath(outdir, "gc_stdres_hist.png"))
        savefig(GCViz.plot_stdres_qq(y, xm, B, R), joinpath(outdir, "gc_stdres_qq.png"))
        savefig(GCViz.plot_stdres_acf(y, xm, B, R), joinpath(outdir, "gc_stdres_acf.png"))
        savefig(GCViz.plot_mse_time(x_true, xm), joinpath(outdir, "gc_mse_time.png"))
        savefig(GCViz.plot_state_coverage_time(x_true, xm), joinpath(outdir, "gc_state_coverage_time.png"))
        savefig(GCViz.summary_dashboard(x_true, y, xm, fe.obs_term, fe.dyn_term, fe.total), joinpath(outdir, "gc_dashboard.png"))
        # Plot FE over steps (use per-time total FE)
        savefig(GCViz.plot_fe_iterations(fe.total), joinpath(outdir, "gc_fe_iterations.png"))
        savefig(GCViz.plot_stdres_time(y, xm, B, R), joinpath(outdir, "gc_stdres_time.png"))
        savefig(GCViz.plot_derivative_consistency(xm, scen.dt), joinpath(outdir, "gc_derivative_consistency.png"))

        # CSVs
        open(joinpath(outdir, "rxinfer_free_energy.csv"), "w") do io
            println(io, "iteration,free_energy")
            for (i,F) in enumerate(result.free_energy)
                println(io, string(i, ",", F))
            end
        end
        open(joinpath(outdir, "gc_free_energy_timeseries.csv"), "w") do io
            println(io, "t,obs_term,prior_term,dyn_term,total")
            for t in 1:scen.n
                println(io, string(t, ",", fe.obs_term[t], ",", fe.prior_term[t], ",", fe.dyn_term[t], ",", fe.total[t]))
            end
        end
        # Metrics: RMSE and 95% coverage per state
        n = scen.n
        μ = [mean(xm[t]) for t in 1:n]
        σ2 = Vector{Vector{Float64}}(undef, n)
        for t in 1:n
            Σraw = try
                cov(xm[t])
            catch
                var(xm[t])
            end
            σ2[t] = Σraw isa AbstractVector ? collect(Σraw) : diag(Σraw)
        end
        rmse = [ sqrt(mean([ (μ[t][k] - x_true[t][k])^2 for t in 1:n ])) for k in 1:3 ]
        coverage = [ mean([ abs(x_true[t][k] - μ[t][k]) <= 1.96 * sqrt(σ2[t][k]) for t in 1:n ]) for k in 1:3 ]
        open(joinpath(outdir, "metrics.csv"), "w") do io
            println(io, "metric,pos,vel,acc")
            println(io, "rmse,", join(string.(rmse), ","))
            println(io, "coverage95,", join(string.(coverage), ","))
        end
        open(joinpath(outdir, "scenario_config.toml"), "w") do io
            println(io, "name = \"" * scen.name * "\"")
            println(io, "n = ", scen.n)
            println(io, "dt = ", scen.dt)
            println(io, "sigma_a = ", scen.σ_a)
            println(io, "sigma_obs_pos = ", scen.σ_obs_pos)
            println(io, "sigma_obs_vel = \"" * string(scen.σ_obs_vel) * "\"")
            println(io, "generator = \"" * String(Symbol(scen.generator)) * "\"")
        end

        # Markdown report
        GCReport.write_markdown_report(outdir, scen; extra=Dict(
            :metrics => Dict(:rmse=>rmse, :coverage=>coverage),
            :fe_iters => fe.total,
        ))
    end
end


