module LLMUtils

using JSON
using RxInferLLM: openai_key, CONFIG
using Dates

export sentiment_score, topic_extraction, explainability

function sentiment_score(text::String)
    messages = [
        Dict("role"=>"system","content"=>"You are a sentiment analyst."),
        Dict("role"=>"user","content"=>"Analyze sentiment and return JSON {\"score\": number(0-1), \"label\": string}\nText: \"$text\"")
    ]
    r = RxInferLLM.call_to_llm(openai_key(), CONFIG.llm_model, messages; response_format = Dict("type"=>"json_schema","json_schema"=>Dict("name"=>"sentiment","schema"=>Dict("type"=>"object","properties"=>Dict("score"=>Dict("type"=>"number","minimum"=>0,"maximum"=>1),"label"=>Dict("type"=>"string")),"required"=>["score","label"]))))
    obj = JSON.parse(r.response[:choices][1][:message][:content])
    return obj
end

function topic_extraction(text::String)
    messages = [Dict("role"=>"system","content"=>"You are a topic extractor."), Dict("role"=>"user","content"=>"Extract up to 5 topics as JSON array of strings from:\n$text")]
    r = RxInferLLM.call_to_llm(openai_key(), CONFIG.llm_model, messages)
    # Try parse as JSON array
    parsed = try JSON.parse(r.response[:choices][1][:message][:content]) catch; split(r.response[:choices][1][:message][:content], '\n') end
    return parsed
end

function explainability(text::String)
    messages = [Dict("role"=>"system","content"=>"You are an explainability assistant."), Dict("role"=>"user","content"=>"Provide a short (<=100 words) explanation of why this text is positive/negative: $text")]
    r = RxInferLLM.call_to_llm(openai_key(), CONFIG.llm_model, messages)
    return r.response[:choices][1][:message][:content]
end

end # module


