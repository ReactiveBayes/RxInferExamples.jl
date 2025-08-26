using Test
using RxInferLLM

@testset "LLM flexible calls" begin
    if RxInferLLM.has_openai_key()
        # temperature variations
        m1 = RxInferLLM.call_to_llm(RxInferLLM.openai_key(), RxInferLLM.CONFIG.llm_model, [Dict("role"=>"user","content"=>"Say hi")]; temperature=0.0)
        m2 = RxInferLLM.call_to_llm(RxInferLLM.openai_key(), RxInferLLM.CONFIG.llm_model, [Dict("role"=>"user","content"=>"Say hi")]; temperature=0.9)
        @test m1 !== nothing && m2 !== nothing

        # system prompt override
        messages = [Dict("role"=>"system","content"=>"Respond in JSON {\"greeting\":<string>}"), Dict("role"=>"user","content"=>"Greet me")]
        r = RxInferLLM.call_to_llm(RxInferLLM.openai_key(), RxInferLLM.CONFIG.llm_model, messages; response_format=Dict("type"=>"json_schema","json_schema"=>Dict("name"=>"greet","schema"=>Dict("type"=>"object","properties"=>Dict("greeting"=>Dict("type"=>"string")),"required"=>["greeting"]))))
        parsed = try JSON.parse(r.response[:choices][1][:message][:content]) catch; nothing end
        @test parsed !== nothing && haskey(parsed, "greeting")

        # prose output
        messages = [Dict("role"=>"user","content"=>"Explain Bayesian inference in one sentence.")]
        r2 = RxInferLLM.call_to_llm(RxInferLLM.openai_key(), RxInferLLM.CONFIG.llm_model, messages; temperature=0.2)
        @test isa(r2.response[:choices][1][:message][:content], String)
    else
        @info "Skipping flexible LLM calls tests because OPENAI_KEY is not set"
    end
end


