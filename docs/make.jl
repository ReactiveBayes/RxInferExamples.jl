using Documenter, Weave

makedocs(
    clean=true,
    sitename="RxInfer.jl Examples",
    pages=[
        "Main page" => "index.md"
    ],
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true",
        example_size_threshold=200 * 1024,
        size_threshold_warn=200 * 1024,
    )
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo="github.com/ReactiveBayes/RxInferExamples.jl.git",
        devbranch="main",
        forcepush=true
    )
end