using Test
using JSON
using RxInferLLM

@testset "LLM logged responses validation" begin
    if RxInferLLM.has_openai_key()
        path = get(ENV, "LLM_LOG_PATH", "output/llm_responses.jsonl")
        @test isfile(path) |> identity

        # read last 10 entries
        lines = readlines(path)
        @test !isempty(lines)

        for ln in Iterators.take(reverse(lines), 10)
            entry = try JSON.parse(ln) catch e; @test false; continue end
            @test haskey(entry, "timestamp") && haskey(entry, "model") && haskey(entry, "content")

            # content should be parseable JSON from assistant
            content = entry["content"]
            parsed = try JSON.parse(content) catch e; nothing end
            @test !(parsed === nothing)

            # validate shape
            @test all(haskey(parsed, k) for k in ["analysis", "mean", "variance"]) == true

            # numeric coercion
            mean_v = parsed["mean"]
            var_v = parsed["variance"]
            if isa(mean_v, String)
                mean_v = try parse(Float64, mean_v) catch; NaN end
            end
            if isa(var_v, String)
                var_v = try parse(Float64, var_v) catch; NaN end
            end

            @test isfinite(mean_v) && 0.0 <= mean_v <= 10.0
            @test isfinite(var_v) && 0.0 < var_v <= 100.0
        end
    else
        @info "Skipping logged responses validation because OPENAI_KEY is not set"
    end
end


