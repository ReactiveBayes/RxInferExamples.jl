using Test
using DelimitedFiles

latest_output_root() = begin
    root = joinpath(@__DIR__, "..", "output")
    isdir(root) || return nothing
    d = sort(filter(isdir, joinpath(root, name) for name in readdir(root)); by=basename)
    isempty(d) ? nothing : last(d)
end

project_root = normpath(joinpath(@__DIR__, ".."))
run_script_path = normpath(joinpath(project_root, "run.jl"))

@testset "Infinite Data Stream pipeline" begin
    # Build a Julia command that evaluates include with a raw string path
    julia_cmd = `julia --project=$(project_root) --eval $("ENV[\\\"GKSwstype\\\"]=\\\"100\\\"; include(raw\\\"$(run_script_path)\\\")")`
    run(julia_cmd)

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
end

