using Test
using RxInfer
using RxInferLLM

@testset "LLM modules basic" begin
    @test isa(RxInferLLM.CONFIG, NamedTuple)
    @test haskey(RxInferLLM.CONFIG, :llm_model)

    model = RxInferLLM.language_mixture_model(c = RxInferLLM.CONFIG.observations,
        context₁ = "c1",
        context₂ = "c2",
        task₁ = RxInferLLM.CONFIG.prior_task,
        task₂ = RxInferLLM.CONFIG.prior_task,
        likelihood_task = RxInferLLM.CONFIG.likelihood_task)
    @test occursin("ModelGenerator", string(typeof(model)))

    # If OPENAI_KEY is present run a quick smoke inference; otherwise skip.
    if RxInferLLM.CONFIG.HAS_OPENAI_KEY
        @testset "LLM smoke" begin
            # run a tiny inference (single iteration) to ensure connectivity
            res = RxInferLLM.run_inference(RxInferLLM.CONFIG)
            @test !isnothing(res)
        end
    else
        @info "Skipping LLM smoke test because OPENAI_KEY is not set"
    end
end


