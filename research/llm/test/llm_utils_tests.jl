using Test
using RxInferLLM

@testset "LLM utility methods" begin
    if RxInferLLM.has_openai_key()
        s = RxInferLLM.LLMUtils.sentiment_score("I really like this library.")
        @test haskey(s, "score") && haskey(s, "label")
        @test 0.0 <= s["score"] <= 1.0

        topics = RxInferLLM.LLMUtils.topic_extraction("Bayesian inference, probabilistic programming, variational methods")
        @test isa(topics, Array)

        expl = RxInferLLM.LLMUtils.explainability("I had a bad experience.")
        @test isa(expl, String)
    else
        @info "Skipping LLM utils tests because OPENAI_KEY is not set"
    end
end


