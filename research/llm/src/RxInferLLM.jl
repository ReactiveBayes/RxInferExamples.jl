


module RxInferLLM

include("Config.jl")
include("Nodes.jl")
include("Rules.jl")
include("Inference.jl")

include("Plotting.jl")
include("LLMUtils.jl")

export main, CONFIG, run_inference, language_mixture_model, make_plots

function main()
    println("Running RxInferLLM.main with model=", CONFIG.llm_model)
    results = run_inference(CONFIG)
    make_plots(results, CONFIG)
    return results
end

end # module


