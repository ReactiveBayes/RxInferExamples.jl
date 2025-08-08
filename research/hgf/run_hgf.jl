#!/usr/bin/env julia

using Logging
using Plots
using Statistics
using Dates
using Printf
using JSON

include("HGF.jl")
using .HGF

function main()
    # Ensure headless plotting works on servers/CI
    ENV["GKSwstype"] = "100"
    logger = ConsoleLogger(stderr, Logging.Info)
    global_logger(logger)

    params = default_hgf_params()

    z, x, y = generate_data(params)

    filter_result = run_filter(y, params.real_k, params.real_w, params.z_variance, params.y_variance)
    z_hist = filter_result.history[:z_next]
    x_hist = filter_result.history[:x_next]

    p1 = plot_hidden_states(z, x, y, z_hist, x_hist; title_suffix = "Filtering")
    p2 = plot_free_energy(filter_result.free_energy_history)

    smoothing_result = run_smoothing(y, params.z_variance, params.y_variance)
    z_post = smoothing_result.posteriors[:z]
    x_post = smoothing_result.posteriors[:x]
    p3 = plot_hidden_states(z, x, y, z_post, x_post; title_suffix = "Smoothing")
    p4 = plot_free_energy(smoothing_result.free_energy)
    q_κ = smoothing_result.posteriors[:κ]
    q_ω = smoothing_result.posteriors[:ω]
    p5 = plot_param_posteriors(q_κ, q_ω; real_k = params.real_k, real_w = params.real_w)

    # Diagnostics
    p_err = HGF.Viz.plot_state_errors(z, x, z_hist, x_hist)
    p_res = HGF.Viz.plot_residuals(y, x_hist)
    p_var = HGF.Viz.plot_variance_trajectories(z_hist, x_hist)
    residual_series = y .- mean.(x_hist)
    p_acf = HGF.Viz.plot_residual_acf(residual_series; maxlag = 60)
    p_qq = HGF.Viz.plot_residual_qq(residual_series)
    p_cov = HGF.Viz.plot_coverage(z, x, z_hist, x_hist; alpha = 0.05)

    # Save outputs to top-level results directory
    project_root = normpath(joinpath(@__DIR__, "..", ".."))
    outdir = normpath(joinpath(project_root, "results", string(Dates.format(now(), "yyyy-mm-dd_HHMMSS")), "HGF"))
    isdir(outdir) || mkpath(outdir)
    savefig(p1, joinpath(outdir, "hidden_states_filtering.png"))
    savefig(p2, joinpath(outdir, "free_energy_filtering.png"))
    savefig(p3, joinpath(outdir, "hidden_states_smoothing.png"))
    savefig(p4, joinpath(outdir, "free_energy_smoothing.png"))
    savefig(p5, joinpath(outdir, "param_posteriors.png"))
    savefig(p_err, joinpath(outdir, "state_errors.png"))
    savefig(p_res, joinpath(outdir, "residuals.png"))
    savefig(p_var, joinpath(outdir, "variance_trajectories.png"))
    savefig(p_acf, joinpath(outdir, "residuals_acf.png"))
    savefig(p_qq, joinpath(outdir, "residuals_qq.png"))
    savefig(p_cov, joinpath(outdir, "coverage.png"))

    # Default GIF and MP4 outputs
    try
        HGF.Viz.make_hidden_states_animation(
            z, x, y, z_hist, x_hist;
            fname = joinpath(outdir, "hidden_states_filtering.gif"),
            mp4_fname = joinpath(outdir, "hidden_states_filtering.mp4"),
            title_suffix = "Filtering", max_frames = 180, fps = 24)
        HGF.Viz.make_hidden_states_animation(
            z, x, y, z_post, x_post;
            fname = joinpath(outdir, "hidden_states_smoothing.gif"),
            mp4_fname = joinpath(outdir, "hidden_states_smoothing.mp4"),
            title_suffix = "Smoothing", max_frames = 180, fps = 24)
        HGF.Viz.make_residuals_animation(
            y, x_hist;
            fname = joinpath(outdir, "residuals.gif"),
            mp4_fname = joinpath(outdir, "residuals.mp4"),
            max_frames = 180, fps = 24)
    catch err
        @warn "Animation generation failed" error = err
    end

    # Write summary file
    open(joinpath(outdir, "summary.txt"), "w") do io
        @printf(io, "Approximate value of κ: %.6f\n", mean(q_κ))
        @printf(io, "Approximate value of ω: %.6f\n", mean(q_ω))
    end

    # Structured JSON report
    report = Dict(
        "params" => Dict(
            "seed" => params.seed,
            "real_k" => params.real_k,
            "real_w" => params.real_w,
            "z_variance" => params.z_variance,
            "y_variance" => params.y_variance,
            "n" => params.n,
        ),
        "filtering" => Dict(
            "free_energy" => filter_result.free_energy_history,
            "z_mean_final" => mean(last(z_hist)),
            "x_mean_final" => mean(last(x_hist)),
        ),
        "smoothing" => Dict(
            "free_energy" => smoothing_result.free_energy,
            "k_mean" => mean(q_κ),
            "w_mean" => mean(q_ω),
        ),
        "metrics" => Dict(
            "mse_z" => mean(abs2, z .- mean.(z_post)),
            "mse_x" => mean(abs2, x .- mean.(x_post)),
            "mse_y_residual" => mean(abs2, residual_series),
        ),
        "files" => Dict(
            "hidden_states_filtering_png" => "hidden_states_filtering.png",
            "free_energy_filtering_png" => "free_energy_filtering.png",
            "hidden_states_smoothing_png" => "hidden_states_smoothing.png",
            "free_energy_smoothing_png" => "free_energy_smoothing.png",
            "param_posteriors_png" => "param_posteriors.png",
            "state_errors_png" => "state_errors.png",
            "residuals_png" => "residuals.png",
            "variance_trajectories_png" => "variance_trajectories.png",
            "residuals_acf_png" => "residuals_acf.png",
            "residuals_qq_png" => "residuals_qq.png",
            "coverage_png" => "coverage.png",
            "hidden_states_filtering_gif" => "hidden_states_filtering.gif",
            "hidden_states_smoothing_gif" => "hidden_states_smoothing.gif",
         "hidden_states_filtering_mp4" => "hidden_states_filtering.mp4",
         "hidden_states_smoothing_mp4" => "hidden_states_smoothing.mp4",
         "residuals_gif" => "residuals.gif",
         "residuals_mp4" => "residuals.mp4",
            "summary_txt" => "summary.txt",
        ),
    )
    open(joinpath(outdir, "report.json"), "w") do io
        JSON.print(io, report)
    end

    @info "Saved HGF outputs" outdir
end

main()


