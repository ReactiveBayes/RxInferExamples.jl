using Documenter, Weave

# Function to generate pages structure from examples directory
function generate_pages()
    examples_dir = joinpath(@__DIR__, "src", "examples")
    !isdir(examples_dir) && return ["Main page" => "index.md"]

    # Base pages with main index
    pages = Any["Home" => "index.md"]

    # Helper function to create nested page structure
    function process_directory(dir, prefix="")
        items = []
        
        # Process all markdown files in the current directory
        for entry in readdir(dir)
            path = joinpath(dir, entry)
            
            if isfile(path) && endswith(entry, ".md") && entry != "index.md"
                # Get relative path from src directory
                rel_path = relpath(path, joinpath(@__DIR__, "src"))
                # Remove .md extension for the page title
                title = replace(basename(entry), r"\.md$" => "")
                push!(items, title => rel_path)
            elseif isdir(path) && !startswith(entry, "_")
                # Process subdirectory (skip directories starting with _)
                subdir_items = process_directory(path, joinpath(prefix, entry))
                if !isempty(subdir_items)
                    push!(items, entry => subdir_items)
                end
            end
        end
        
        return items
    end

    # Add all example pages
    append!(pages, process_directory(examples_dir))

    return pages
end

makedocs(
    clean=true,
    sitename="RxInfer.jl Examples",
    pages=generate_pages(),
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