using RxInfer, JSON, OpenAI, Dates

# Path to append raw LLM responses (can be overridden via ENV)
const LLM_LOG_PATH = get(ENV, "LLM_LOG_PATH", "output/llm_responses.jsonl")
const LLM_REQUEST_LOG_PATH = get(ENV, "LLM_REQUEST_LOG_PATH", "output/llm_requests.jsonl")

function call_to_llm(secret::String, model::String, messages; response_format=nothing, retries::Int=2, kwargs...)
    # Retry wrapper around create_chat; logs the assistant content (safe to log, avoid secrets)
    last_err = nothing
    for attempt in 1:(retries + 1)
        start_ns = time_ns()
        try
            # log request
            try
                isdir("output") || mkdir("output")
                open(LLM_REQUEST_LOG_PATH, "a") do io
                    reqentry = Dict("timestamp" => string(Dates.now()), "model" => model, "attempt" => attempt, "messages" => messages, "kwargs" => Dict(kwargs))
                    println(io, JSON.json(reqentry))
                end
            catch _
            end

            if response_format === nothing
                r = create_chat(secret, model, messages; kwargs...)
            else
                r = create_chat(secret, model, messages; response_format = response_format, kwargs...)
            end
            duration_ms = (time_ns() - start_ns) / 1e6

            # ensure output dir and write response metadata
            try
                isdir("output") || mkdir("output")
                open(LLM_LOG_PATH, "a") do io
                    entry = Dict(
                        "timestamp" => string(Dates.now()),
                        "model" => model,
                        "attempt" => attempt,
                        "duration_ms" => duration_ms,
                        "status" => "ok",
                        "content" => r.response[:choices][1][:message][:content]
                    )
                    println(io, JSON.json(entry))
                end
            catch _
                # logging should not break primary flow
            end

            return r
        catch e
            last_err = e
            # record error
            try
                isdir("output") || mkdir("output")
                open(LLM_LOG_PATH, "a") do io
                    entry = Dict(
                        "timestamp" => string(Dates.now()),
                        "model" => model,
                        "attempt" => attempt,
                        "duration_ms" => (time_ns() - start_ns) / 1e6,
                        "status" => "error",
                        "error" => string(e)
                    )
                    println(io, JSON.json(entry))
                end
            catch _
            end
            if attempt <= retries
                sleep(1)
                continue
            else
                rethrow(e)
            end
        end
    end
    rethrow(last_err)
end

@rule LLMPrior(:b, Marginalisation) (q_c::PointMass{<:String}, q_t::PointMass{<:String}) = begin
    messages = [
        Dict("role" => "system",
             "content" => """
                You are an expert analyst who maps contextual cues to a
                Normal(mean, variance) distribution.

                • Think step-by-step internally.
                • **Only** output a JSON object that conforms to the schema below.
                • Do not wrap the JSON in markdown fences or add extra keys.
            """),

        Dict("role" => "assistant",
             "content" => """
                ## CONTEXT
                $(q_c.point)
            """),

        Dict("role" => "user",
             "content" => """
                ## TASK
                $(q_t.point)

                Using the context above, infer a Normal distribution and return:
                  "analysis"  – brief rationale (≤ 100 words)
                  "mean"      – number in [0, 10]
                  "variance"  – number in [1, 100]
            """)
    ]

    response_schema = Dict(
        "type" => "json_schema",
        "json_schema" => Dict(
            "name"   => "normal_estimate",
            "schema" => Dict(
                "type"       => "object",
                "properties" => Dict(
                    "analysis" => Dict("type" => "string"),
                    "mean"     => Dict("type" => "number", "minimum" => 0, "maximum" => 10),
                    "variance" => Dict("type" => "number", "minimum" => 1, "maximum" => 100)
                ),
                "required" => ["analysis", "mean", "variance"],
                "additionalProperties" => false
            )
        )
    )

    r = call_to_llm(CONFIG.secret_key, CONFIG.llm_model, messages; response_format = response_schema)
    obj = JSON.parse(r.response[:choices][1][:message][:content])

    return NormalMeanVariance(obj["mean"], obj["variance"])
end

@rule LLMObservation(:b, Marginalisation) (q_out::PointMass{<:String}, q_t::PointMass{<:String}) = begin
    messages = [
        Dict("role" => "system",
             "content" => """
                You are **LLMObservation**, a senior evaluator who maps a text to
                a Normal(mean, variance) distribution.

                • Think step-by-step internally, but **only** output a JSON object
                  that conforms to the provided schema.
                • Do not wrap the JSON in markdown fences or add extra keys.
            """),

        Dict("role" => "assistant",
             "content" => """
                ## TEXT
                $(q_out.point)
            """),

        Dict("role" => "user",
             "content" => """
                ## TASK
                $(q_t.point)

                Using the text above, infer a Gaussian distribution.
                Return a JSON object with keys:
                  "analysis"  – ≤ 100 words explaining your reasoning
                  "mean"      – number in [0, 10]
                  "variance"  – number in [0.1, 100]
            """)
    ]

    response_schema = Dict(
        "type" => "json_schema",
        "json_schema" => Dict(
            "name"   => "normal_estimate",
            "schema" => Dict(
                "type"       => "object",
                "properties" => Dict(
                    "analysis" => Dict("type" => "string"),
                    "mean"     => Dict("type" => "number", "minimum" => 0, "maximum" => 10),
                    "variance" => Dict("type" => "number", "minimum" => 0.1, "maximum" => 100)
                ),
                "required" => ["analysis", "mean", "variance"],
                "additionalProperties" => false
            )
        )
    )

    r = call_to_llm(CONFIG.secret_key, CONFIG.llm_model, messages; response_format = response_schema)
    obj = JSON.parse(r.response[:choices][1][:message][:content])

    return NormalMeanVariance(obj["mean"], obj["variance"])
end

