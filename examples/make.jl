using Pkg

# Ensure Weave is available
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Distributed
using Base.Filesystem

# Parse command line arguments
const FILTER = length(ARGS) > 0 ? ARGS[1] : nothing

# Add worker processes if none exist
if nworkers() == 1
    addprocs(max(1, Sys.CPU_THREADS ÷ 2))
end

# Load Weave and Pkg on all workers
@everywhere using Weave
@everywhere using Pkg

# Define build directories
const BUILD_DIR = abspath(joinpath(@__DIR__, "..", "docs", "src", "examples"))
const CACHE_DIR = joinpath(BUILD_DIR, "_cache")

# Create build directories if dont exist
mkpath(BUILD_DIR)
mkpath(CACHE_DIR)

@info """
Build directories:
BUILD_DIR: $BUILD_DIR
CACHE_DIR: $CACHE_DIR
"""

# Function to copy files with exceptions
@everywhere function copy_files(src_dir, dst_dir; exclude = [])
    # Create destination directory if it doesn't exist
    mkpath(dst_dir)
    
    # Copy all files except those in exclude list
    for item in readdir(src_dir; join=true)
        basename(item) in exclude && continue
        
        if isfile(item)
            dst = joinpath(dst_dir, basename(item))
            @info "Copying file: $(basename(item))"
            cp(item, dst; force=true)
        elseif isdir(item)
            # Recursively copy subdirectories
            dst_subdir = joinpath(dst_dir, basename(item))
            copy_files(item, dst_subdir; exclude)
        end
    end
end

# Function to check markdown for error blocks
@everywhere function has_error_blocks(md_path)
    !isfile(md_path) && return true
    mdtext = read(md_path, String)
    
    erroridx = findnext("```\nError:", mdtext, 1)
    if !isnothing(erroridx)
        # Extract error context window
        errwindow = 500
        errstart = reduce((idx, _) -> max(firstindex(mdtext), prevind(mdtext, idx)), 1:errwindow; init = first(erroridx))
        errend = reduce((idx, _) -> min(lastindex(mdtext), nextind(mdtext, idx)), 1:errwindow; init = last(erroridx))
        
        # Log the error with context
        @warn """
        Error block found in $md_path
        Error context:
        
        $(mdtext[errstart:errend])
        """
        return true
    end
    return false
end

# Function to fix image paths in markdown
# The problem is that Weave generates absolute image paths 
# but the Documenter wants relative paths
# this function finds absolute path in the string content 
# of the file (verbatim) and replaces it with relative paths 
@everywhere function fix_image_paths(md_path)
    content = read(md_path, String)
    
    # Pattern to match markdown image syntax with absolute paths
    # Matches both ![alt](path) and ![](path) formats
    pattern = r"!\[(.*?)\]\((/.*?/docs/src/examples/[^\)]+)\)"
    
    # Replace absolute paths with relative ones
    new_content = content
    for m in eachmatch(pattern, content)
        alt_text, abs_path = m.captures
        
        # Get the relative path from the markdown file to the image
        md_dir = dirname(md_path)
        img_path = normpath(abs_path)
        rel_path = relpath(img_path, md_dir)
        
        @info "Converting path in $(basename(md_path)):" abs_path => rel_path
        
        # Replace this specific match
        old_img = "![$(alt_text)]($(abs_path))"
        new_img = "![$(alt_text)]($(rel_path))"
        new_content = replace(new_content, old_img => new_img)
    end
    
    # Write back only if changes were made
    if content != new_content
        write(md_path, new_content)
    end
end

# Function to process a single notebook
@everywhere function process_notebook(notebook_path, build_dir, cache_dir)
    # Get the notebook's directory and activate its environment
    notebook_dir = dirname(notebook_path)
    
    # Setup paths
    input_path = abspath(notebook_path)
    rel_path = relpath(notebook_path, @__DIR__)
    output_path = joinpath(build_dir, replace(rel_path, ".ipynb" => ".md"))
    output_dir = dirname(output_path)
    build_input_path = joinpath(build_dir, rel_path)
    
    @info """
    Notebook directory: $notebook_dir
    Files in the notebook directory: $(readdir(notebook_dir))
    Output directory: $output_dir
    """
    
    # Create directories if dont exist
    mkpath(output_dir)

    # Copy all the files (except for `Manifest.toml`) from the notebook directory 
    # to the build directory
    copy_files(notebook_dir, output_dir; exclude = ["Manifest.toml"])
    
    @info """
    After copying:
    Files in output_dir: $(readdir(output_dir))
    """
    
    # Create notebook-specific cache directory
    notebook_cache_dir = joinpath(cache_dir, dirname(rel_path))
    mkpath(notebook_cache_dir)
    
    # Change to output directory before activating
    cd(output_dir)
    @info "Working directory changed to: $(pwd())"

    # Create a new module for this notebook
    mod = Module()
    @info "Created an anonymous module for execution of $notebook_dir"
    
    # Activate the project in the current directory
    Core.eval(mod, quote
        using Pkg
        Pkg.activate(".")
        Pkg.instantiate()
    end)
    
    @info "Processing $rel_path..."
    try
        weave(build_input_path;
            out_path=output_path,
            doctype="github",
            fig_path=output_dir,
            cache=:all,
            cache_path=notebook_cache_dir,
            mod=mod
        )
        
        # Fix any absolute image paths in the generated markdown
        fix_image_paths(output_path)
        
        # Read the existing content
        content = read(output_path, String)

        CONTRIBUTING_NOTE = """
        !!! note "Contributing"
            This example was automatically generated from a Jupyter notebook in the [RxInferExamples.jl](https://github.com/ReactiveBayes/RxInferExamples.jl) repository.

            We welcome and encourage contributions! You can help by:
            - Improving this example
            - Creating new examples 
            - Reporting issues or bugs
            - Suggesting enhancements

            Visit our [GitHub repository](https://github.com/ReactiveBayes/RxInferExamples.jl) to get started.
            Together we can make [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl) even better! 💪
        """
        
        # Write the contribution note at the beginning
        write(output_path, 
        """
        $CONTRIBUTING_NOTE
        ---
        $content
        ---
        $CONTRIBUTING_NOTE
        """)
        
        @info "Successfully processed $rel_path"
        return output_path
    catch e
        @error "Failed to process $rel_path" exception=(e, catch_backtrace())
        return nothing
    end
end

# Find all notebook files in the examples directory
notebook_files = String[]
for (root, dirs, files) in walkdir(@__DIR__)
    # Skip .ipynb_checkpoints directories
    filter!(d -> d != ".ipynb_checkpoints", dirs)
    
    for file in files
        if endswith(file, ".ipynb")
            push!(notebook_files, relpath(joinpath(root, file), @__DIR__))
        end
    end
end

@info """
Found notebooks:
$(join(["  • " * notebook for notebook in sort(notebook_files)], "\n"))
"""

# Filter notebooks if a pattern is provided
if !isnothing(FILTER)
    original_count = length(notebook_files)
    notebook_files = filter(notebook_files) do notebook
        # Only match against the notebook name, not the full path
        notebook_name = basename(notebook)
        occursin(lowercase(FILTER), lowercase(notebook_name))
    end
    filtered_count = length(notebook_files)
    if filtered_count == 0
        @error "No notebooks found matching pattern: $FILTER"
        exit(1)
    end
    @info """
    Filtered notebooks ($filtered_count / $original_count match pattern '$FILTER'):
    $(join(["  • " * notebook for notebook in sort(notebook_files)], "\n"))
    """
end

# Process notebooks in parallel
results = pmap(notebook -> (notebook, process_notebook(
    joinpath(@__DIR__, notebook),
    BUILD_DIR,
    CACHE_DIR
)), notebook_files)

is_processing_failed(output_path) = isnothing(output_path) || has_error_blocks(output_path)

# Check for errors in output files
final_results = [(notebook, output_path, is_processing_failed(output_path)) for (notebook, output_path) in results]

# Split results into successful and failed
successful_notebooks = [notebook for (notebook, _, failed) in final_results if !failed]
failed_notebooks = [notebook for (notebook, _, failed) in final_results if failed]

@info """
Processing Report:
Total notebooks: $(length(final_results))
Successful: $(length(successful_notebooks))
Failed: $(length(failed_notebooks))

$(isempty(successful_notebooks) ? "" : "Successfully processed notebooks:\n" * join(["  • " * notebook for notebook in successful_notebooks], "\n"))
$(isempty(failed_notebooks) ? "" : "Failed notebooks:\n" * join(["  • " * notebook for notebook in failed_notebooks], "\n"))
"""

if !isempty(failed_notebooks)
    @warn """
    Some notebooks failed to process. This might be due to:
    1. Actual errors in the notebooks
    2. Cached results causing issues

    Try running 'make clean' to clear all build artifacts and cache,
    then run the build again. If the error persists, check the notebook contents.
    """
end

if isempty(failed_notebooks)
    @info """
    All notebooks processed successfully! 🎉
    
    Next steps:
    Run `make docs` to generate the documentation website.
    """
end

# Exit with error if any notebooks failed
exit(isempty(failed_notebooks) ? 0 : 1)
