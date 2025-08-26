#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using RxInferLLM

function main()
    println("Running RxInferLLM example...")
    if !RxInferLLM.has_openai_key()
        println("ERROR: OPENAI_KEY not set. Set it in the environment or use a .env file.")
        println("See .env.example and scripts/run_with_key.sh for secure usage.")
        return
    end

    results = RxInferLLM.main()
    # Save a simple summary to output/
    isdir("output") || mkdir("output")
    open("output/inference_summary.txt", "w") do io
        println(io, "Inference completed. Iterations = ", RxInferLLM.CONFIG.n_iterations)
        try
            if hasproperty(results, :posteriors)
                println(io, "Posterior keys: ", collect(keys(results.posteriors)))
            else
                println(io, "Result type: ", typeof(results))
            end
        catch e
            println(io, "Could not enumerate result keys: ", e)
        end
    end
    println("Saved summary to output/inference_summary.txt")
end

main()


