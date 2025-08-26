#!/usr/bin/env julia
"""Entry point for running the RxInfer LLM example.

Configuration is at the top; edit `CONFIG` to change behavior.
"""
module RunRxInferLLM

using RxInfer, OpenAI, JSON, Plots
using .Config
using .Nodes
using .Rules
using .Inference
using .Plotting

const CONFIG = Config.CONFIG

function main()
    println("Running RxInfer LLM example with model=", CONFIG.llm_model)

    results = run_inference(CONFIG)

    make_plots(results, CONFIG)

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module

export main


