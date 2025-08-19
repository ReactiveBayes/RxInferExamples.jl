using Test
using DelimitedFiles

latest_output_root() = begin
    root = joinpath(@__DIR__, "..", "output")
    isdir(root) || return nothing
    paths = [joinpath(root, name) for name in readdir(root)]
    dirs = filter(isdir, paths)
    isempty(dirs) && return nothing
    sort(dirs; by=basename) |> last
end

project_root = normpath(joinpath(@__DIR__, ".."))
run_script_path = normpath(joinpath(project_root, "run.jl"))

@testset "Infinite Data Stream pipeline" begin
    # Launch using Julia's absolute executable to avoid PATH/ENOENT issues
    julia_exe = Base.julia_cmd()
    cmd = `$julia_exe --project=$(project_root) $(run_script_path)`
    run(setenv(cmd, "GKSwstype" => "100"))

    latest = latest_output_root()
    @test latest !== nothing

    static_dir = joinpath(latest, "static")
    realtime_dir = joinpath(latest, "realtime")
    cmp_dir = joinpath(latest, "comparison")

    @test isfile(joinpath(static_dir, "static_inference.png"))
    @test isfile(joinpath(static_dir, "static_free_energy.csv"))
    @test isfile(joinpath(realtime_dir, "realtime_inference.png"))
    @test isfile(joinpath(realtime_dir, "realtime_posterior_x_current.csv"))
    @test isfile(joinpath(cmp_dir, "metrics.txt"))

    metrics = read(joinpath(cmp_dir, "metrics.txt"), String)
    @test occursin("mse_static=", metrics)
    @test occursin("mse_realtime=", metrics)

    # Extended comparison artifacts
    @test isfile(joinpath(cmp_dir, "means_compare.png"))
    @test isfile(joinpath(cmp_dir, "scatter_static_vs_realtime.png"))
    @test isfile(joinpath(cmp_dir, "residuals_static.png"))
    @test isfile(joinpath(cmp_dir, "residuals_realtime.png"))

    # Numerical equivalence (within tolerance) between static and realtime means
    μs = readdlm(joinpath(static_dir, "static_posterior_x_current.csv"))[:, 1]
    μr = readdlm(joinpath(realtime_dir, "realtime_posterior_x_current.csv"))[:, 1]
    upto = min(length(μs), length(μr))
    @test isapprox(μs[1:upto], μr[1:upto]; atol=1e-8, rtol=1e-8)
end

