module Rules

using RxInfer, JSON, OpenAI
import ..Config: CONFIG
using Nodes: LLMPrior, LLMObservation

"""Rules that call the LLM and convert responses into NormalMeanVariance."""

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

    r = create_chat(CONFIG.secret_key, CONFIG.llm_model, messages; response_format = response_schema)
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

    r = create_chat(CONFIG.secret_key, CONFIG.llm_model, messages; response_format = response_schema)
    obj = JSON.parse(r.response[:choices][1][:message][:content])

    return NormalMeanVariance(obj["mean"], obj["variance"])
end

end # module

export


