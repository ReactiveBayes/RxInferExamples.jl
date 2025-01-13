using Documenter, Weave

# Function to collect metadata from all examples
function collect_examples_metadata()
    examples_dir = joinpath(@__DIR__, "src", "examples")
    !isdir(examples_dir) && return []

    metadata = []
    
    # Walk through all directories
    for (root, _, files) in walkdir(examples_dir)
        # Check if there's a meta.jl file
        meta_path = joinpath(root, "meta.jl")
        md_files = filter(f -> endswith(f, ".md") && f != "index.md", files)
        
        for md_file in md_files
            example_path = relpath(joinpath(root, md_file), joinpath(@__DIR__, "src"))
            meta = if isfile(meta_path)
                include(meta_path)
            else
                continue
            end
            
            push!(metadata, (
                path = example_path,
                title = meta.title,
                description = meta.description,
                tags = meta.tags
            ))
        end
    end
    
    return metadata
end

# Function to generate the list of examples page
function generate_examples_list()
    metadata = collect_examples_metadata()
    
    # Group examples by their primary tag (first tag)
    categories = Dict()
    for meta in metadata
        category = meta.tags[1]
        push!(get!(categories, category, []), meta)
    end
    
    # Generate markdown content
    open(joinpath(@__DIR__, "src", "list_of_examples.md"), "w") do io
        write(io, """
        # List of Examples

        This page contains a comprehensive list of all available examples in the RxInfer.jl Examples collection.
        Each example includes a brief description and relevant tags to help you find what you're looking for.

        """)
        
        # Write examples by category
        for category in sort(collect(keys(categories)))
            write(io, """
            ## $(titlecase(category))
            
            """)
            
            for meta in categories[category]
                tags_str = join(["[$(tag)]" for tag in meta.tags[2:end]], " ")
                write(io, """
                ### [$(meta.title)]($(meta.path))
                
                $(meta.description)
                
                **Tags:** $(tags_str)
                
                ---
                
                """)
            end
        end
    end
end

# Generate the examples list before building docs
generate_examples_list()

# Function to generate pages structure from examples directory
function generate_pages()
    examples_dir = joinpath(@__DIR__, "src", "examples")
    !isdir(examples_dir) && return ["Main page" => "index.md"]

    # Base pages with main index and list of examples
    pages = Any[
        "Home" => "index.md",
        "List of Examples" => "list_of_examples.md"
    ]

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