using Test
using DelimitedFiles

project_root = normpath(joinpath(@__DIR__, ".."))

@testset "free energy per-timestep (static)" begin
    # Run static branch via main runner (cheap when artifacts already exist)
    julia_exe = Base.julia_cmd()
    cmd = `$julia_exe --project=$(project_root) $(joinpath(project_root, "run.jl"))`
    run(setenv(cmd, "GKSwstype" => "100"))

    # Find most recent output
    root = joinpath(project_root, "output")
    d = sort(filter(isdir, joinpath(root, n) for n in readdir(root)); by=basename)
    latest = last(d)
    static_dir = joinpath(latest, "static")
    fe = readdlm(joinpath(static_dir, "static_free_energy.csv"))[:]
    obs = readdlm(joinpath(static_dir, "static_observations.csv"))[:]
    @test length(fe) == length(obs)
    @test all(isfinite, fe)
end


