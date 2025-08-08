ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

using Pkg
Pkg.activate(@__DIR__)

using RxInfer, Random, LinearAlgebra, Distributions, StableRNGs, Plots, Statistics
ENV["GKSwstype"] = "100"  # headless plots

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

    # Keep a simple and complex/oscillatory scenarios
    n_each = 2000
    base_scenarios = [
        GCConfig.ScenarioConfig("pos_only_constant_$(n_each)", n_each, 0.1, 0.25, 0.5, NaN, :constant_accel, Dict{Symbol,Any}()),
        GCConfig.ScenarioConfig("pos_only_sin_f02_$(n_each)", n_each, 0.05, 0.1, 0.3, NaN, :sinusoid, Dict(:freq=>0.2, :amp=>5.0)),
        GCConfig.ScenarioConfig("pos_only_piecewise_mixed_$(n_each)", n_each, 0.05, 0.7, 0.4, NaN, :piecewise_mixed, Dict(
            :segment_length=>500,
            :segments=>[
                Dict(:coeffs=>[0.0, 0.15, 0.0, 6e-4, 0.0, -1e-6], :freqs=>[0.05, 0.12],       :amps=>[6.0, 3.5], :phases=>[0.0, π/3]),
                Dict(:coeffs=>[0.0, -0.1, 0.0, -5e-4, 0.0, 1e-6], :freqs=>[0.4, 0.8, 1.2],    :amps=>[5.0, 3.0, 2.0], :phases=>[π/6, π/2, 2π/3]),
                Dict(:coeffs=>[0.0, 0.05, 0.0, 3e-4, 0.0, -8e-7], :freqs=>[0.2, 0.33, 0.5],   :amps=>[4.0, 2.5, 1.5], :phases=>[π/4, 3π/4, π/8]),
                Dict(:coeffs=>[0.0, 0.0, 0.0, -2e-4, 0.0, 5e-7],  :freqs=>[0.1, 0.25, 0.65],  :amps=>[7.0, 3.0, 2.0], :phases=>[π/2, π/5, 7π/8])
            ]
        )),
    ]
    orders = 1:8
    run_cfg = GCConfig.default_run_config()

    for K in orders
        scenarios = [GCConfig.ScenarioConfig(s.name, s.n, s.dt, K, s.σ_a, s.σ_obs_pos, s.σ_obs_vel, s.generator, s.generator_kwargs) for s in base_scenarios]
        order_root = joinpath(outroot, "order_" * string(K))
        isdir(order_root) || mkpath(order_root)
        for scen in scenarios
        x_true, y, A, Q = GCGenerators.generate_scenario(rng, scen)
        K = scen.order
        B = if isnan(scen.σ_obs_vel)
            h = zeros(Float64, 1, K); h[1,1] = 1.0; h
        else
            h = zeros(Float64, 2, K); h[1,1] = 1.0; h[2,2] = 1.0; h
        end
        R = isnan(scen.σ_obs_vel) ? Matrix(Diagonal([scen.σ_obs_pos^2])) : Matrix(Diagonal([scen.σ_obs_pos^2, scen.σ_obs_vel^2]))
        x0_mean = zeros(K)
        x0_cov  = Matrix(Diagonal(fill(100.0, K)))

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

        outdir = joinpath(order_root, scen.name)
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
        savefig(GCViz.summary_dashboard_all(x_true, y, xm, fe.obs_term, fe.dyn_term, fe.total), joinpath(outdir, "gc_dashboard.png"))
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
        D = min(length(first(x_true)), K)
        rmse = [ sqrt(mean([ (μ[t][k] - x_true[t][k])^2 for t in 1:n ])) for k in 1:D ]
        coverage = [ mean([ abs(x_true[t][k] - μ[t][k]) <= 1.96 * sqrt(σ2[t][k]) for t in 1:n ]) for k in 1:D ]
        corr_mean = [
            try
                cor([x_true[t][k] for t in 1:n], [μ[t][k] for t in 1:n])
            catch
                NaN
            end for k in 1:D
        ]
        open(joinpath(outdir, "metrics.csv"), "w") do io
            header = join(vcat(["metric"], ["dim_"*string(k) for k in 1:D]), ",")
            println(io, header)
            println(io, "rmse,", join(string.(rmse), ","))
            println(io, "coverage95,", join(string.(coverage), ","))
            println(io, "corr_mean,", join(string.(corr_mean), ","))
        end
        # Dump per-time true state, posterior mean and variance, and observations
        open(joinpath(outdir, "x_true.csv"), "w") do io
            println(io, join(vcat(["t"], ["dim_"*string(k) for k in 1:D]), ","))
            for t in 1:n
                println(io, join(vcat([string(t)], [string(x_true[t][k]) for k in 1:D]), ","))
            end
        end
        open(joinpath(outdir, "post_mean.csv"), "w") do io
            println(io, join(vcat(["t"], ["dim_"*string(k) for k in 1:D]), ","))
            for t in 1:n
                println(io, join(vcat([string(t)], [string(μ[t][k]) for k in 1:D]), ","))
            end
        end
        open(joinpath(outdir, "post_var.csv"), "w") do io
            println(io, join(vcat(["t"], ["dim_"*string(k) for k in 1:D]), ","))
            for t in 1:n
                println(io, join(vcat([string(t)], [string(σ2[t][k]) for k in 1:D]), ","))
            end
        end
        # Observations
        obs_dims = length(y[1])
        open(joinpath(outdir, "y.csv"), "w") do io
            println(io, join(vcat(["t"], ["obs_"*string(k) for k in 1:obs_dims]), ","))
            for t in 1:n
                println(io, join(vcat([string(t)], [string(y[t][k]) for k in 1:obs_dims]), ","))
            end
        end
        open(joinpath(outdir, "scenario_config.toml"), "w") do io
            println(io, "name = \"" * scen.name * "\"")
            println(io, "n = ", scen.n)
            println(io, "dt = ", scen.dt)
            println(io, "sigma_a = ", scen.σ_a)
            println(io, "sigma_obs_pos = ", scen.σ_obs_pos)
            println(io, "sigma_obs_vel = \"" * string(scen.σ_obs_vel) * "\"")
            println(io, "generator = \"" * String(Symbol(scen.generator)) * "\"")
            println(io, "order = ", scen.order)
        end

        # Markdown report
        GCReport.write_markdown_report(outdir, scen; extra=Dict(
            :metrics => Dict(:rmse=>rmse, :coverage=>coverage),
            :fe_iters => fe.total,
        ))
        end
    end
end


