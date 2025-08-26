using Test
using JSON
using OpenAI
using RxInferLLM

@testset "LLM integration" begin
    if RxInferLLM.has_openai_key()
        messages = [
            Dict("role"=>"system",
                 "content"=>"You are an assistant that outputs a JSON object with keys 'analysis', 'mean', 'variance'."),
            Dict("role"=>"user",
                 "content"=>"Please read the following short text and return a JSON object with keys: analysis (string), mean (0-10), variance (0.1-100). Text: 'I love RxInfer, it's very useful.'")
        ]

        response_schema = Dict(
            "type" => "json_schema",
            "json_schema" => Dict(
                "name" => "normal_estimate",
                "schema" => Dict(
                    "type" => "object",
                    "properties" => Dict(
                        "analysis" => Dict("type" => "string"),
                        "mean" => Dict("type" => "number", "minimum" => 0, "maximum" => 10),
                        "variance" => Dict("type" => "number", "minimum" => 0.1, "maximum" => 100)
                    ),
                    "required" => ["analysis", "mean", "variance"],
                    "additionalProperties" => false
                )
            )
        )

        r = create_chat(RxInferLLM.openai_key(), RxInferLLM.CONFIG.llm_model, messages; response_format = response_schema)
        obj = JSON.parse(r.response[:choices][1][:message][:content])

        @test all(haskey(obj, k) for k in ["analysis", "mean", "variance"]) == true
        @test isa(obj["analysis"], String)

        # mean/variance may be strings or numbers; coerce
        mean_v = obj["mean"]
        var_v = obj["variance"]
        if isa(mean_v, String)
            mean_v = try parse(Float64, mean_v) catch; NaN end
        end
        if isa(var_v, String)
            var_v = try parse(Float64, var_v) catch; NaN end
        end

        @test isfinite(mean_v) && 0.0 <= mean_v <= 10.0
        @test isfinite(var_v) && 0.1 <= var_v <= 100.0
    else
        @info "Skipping LLM integration test because OPENAI_KEY is not set"
    end
end


