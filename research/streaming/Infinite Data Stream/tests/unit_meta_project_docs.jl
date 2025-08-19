using Test

@testset "meta and docs presence" begin
    root = normpath(joinpath(@__DIR__, ".."))
    @test isfile(joinpath(root, "Project.toml"))
    @test isfile(joinpath(root, "Manifest.toml"))
    @test isfile(joinpath(root, "meta.jl"))
    @test isfile(joinpath(root, "README.md"))
    @test isdir(joinpath(root, "docs"))
end


