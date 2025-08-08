using Test
using VisualizationMethods

@testset "scan and render" begin
    models = scan_models()
    @test length(models) > 0
    first_model = first(models)
    mer = model_to_mermaid(first_model.file, first_model.name)
    @test occursin("graph TD", mer)
    dot = model_to_dot(first_model.file, first_model.name)
    @test occursin("digraph G", dot)
    assets = render_graph_assets(first_model.file, first_model.name)
    @test isfile(assets.mmd)
    @test isfile(assets.dot)
end
