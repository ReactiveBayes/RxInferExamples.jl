using Pkg

# Ensure Weave is available
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Weave

# Define build directories
const BUILD_DIR = joinpath(@__DIR__, "build")
const FIG_DIR = joinpath(BUILD_DIR, "figures")

# Create build directories if they don't exist
mkpath(BUILD_DIR)
mkpath(FIG_DIR)

# Find all notebook files in the examples directory
notebook_files = String[]
for (root, dirs, files) in walkdir(@__DIR__)
    for file in files
        if endswith(file, ".ipynb")
            push!(notebook_files, relpath(joinpath(root, file), @__DIR__))
        end
    end
end

# Process each notebook
for notebook in notebook_files
    input_path = joinpath(@__DIR__, notebook)
    output_filename = replace(notebook, ".ipynb" => ".md")
    output_path = joinpath(BUILD_DIR, output_filename)
    
    @info "Processing $notebook..."
    try
        weave(input_path;
            out_path=output_path,
            doctype="github",
            fig_path=FIG_DIR
        )
        @info "Successfully processed $notebook -> $output_filename"
    catch e
        @error "Failed to process $notebook" exception=(e, catch_backtrace())
    end
end

@info "All notebooks processed. Output in $BUILD_DIR"
