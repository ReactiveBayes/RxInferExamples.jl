using Pkg

# Ensure Weave is available
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Weave
using Distributed

# Parse command line arguments
const FILTER = length(ARGS) > 0 ? ARGS[1] : nothing

# Add worker processes if none exist
if nworkers() == 1
    addprocs(max(1, Sys.CPU_THREADS ÷ 2))
end

# Load Weave on all workers
@everywhere using Weave
@everywhere using Pkg

# Define build directories
const BUILD_DIR = abspath(joinpath(@__DIR__, "..", "docs", "src", "examples"))
const FIG_DIR   = joinpath(BUILD_DIR, "figures")
const CACHE_DIR = joinpath(BUILD_DIR, "_cache")

# Create build directories
mkpath(BUILD_DIR)
mkpath(FIG_DIR)
mkpath(CACHE_DIR)

@info """
Build directories:
BUILD_DIR: $BUILD_DIR
FIG_DIR: $FIG_DIR
CACHE_DIR: $CACHE_DIR
"""

# Function to copy auxiliary files
@everywhere function copy_auxiliary_files(src_dir, dst_dir)
    # Create destination directory if it doesn't exist
    mkpath(dst_dir)
    
    # Copy all files except .ipynb files and Manifest.toml
    for item in readdir(src_dir; join=true)
        basename(item) == "Manifest.toml" && continue
        
        if isfile(item)
            dst = joinpath(dst_dir, basename(item))
            @info "Copying auxiliary file: $(basename(item))"
            cp(item, dst; force=true)
        elseif isdir(item)
            # Recursively copy subdirectories
            dst_subdir = joinpath(dst_dir, basename(item))
            copy_auxiliary_files(item, dst_subdir)
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

# Function to process a single notebook
@everywhere function process_notebook(notebook_path, build_dir, fig_dir, cache_dir)
    # Get the notebook's directory and activate its environment
    notebook_dir = dirname(notebook_path)
    
    # Create a new module for this notebook
    mod = Module()
    
    # Setup paths
    input_path = abspath(notebook_path)
    rel_path = relpath(notebook_path, @__DIR__)
    output_path = joinpath(build_dir, replace(rel_path, ".ipynb" => ".md"))
    output_dir = dirname(output_path)
    build_input_path = joinpath(build_dir, rel_path)
    
    @info """
    Path information:
    notebook_dir: $notebook_dir
    output_dir: $output_dir
    Current working directory (before): $(pwd())
    Files in notebook_dir: $(readdir(notebook_dir))
    """
    
    # Create directories and copy files
    mkpath(output_dir)
    copy_auxiliary_files(notebook_dir, output_dir)
    
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
            fig_path=fig_dir,
            cache=:all,
            cache_path=notebook_cache_dir,
            mod=mod
        )
        # Read the existing content
        content = read(output_path, String)
        
        # Write the contribution note at the beginning
        write(output_path, """
        !!! note "Contributing"
            This example was automatically generated from a Jupyter notebook. 
            You can find the source code in the [RxInferExamples.jl](https://github.com/ReactiveBayes/RxInferExamples.jl) repository.
            We welcome contributions! If you'd like to:
            - Fix or improve this example
            - Add a new example
            - Report an issue
            
            Visit our [GitHub repository](https://github.com/ReactiveBayes/RxInferExamples.jl) to get started.
            Your contributions help make RxInfer.jl better for everyone!

        ---

        $content""")
        
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
    @info "Filtered notebooks: $filtered_count / $original_count match pattern '$FILTER'"
end

# Process notebooks in parallel
results = pmap(notebook -> (notebook, process_notebook(
    joinpath(@__DIR__, notebook),
    BUILD_DIR,
    FIG_DIR,
    CACHE_DIR
)), notebook_files)

# Check for errors in output files
final_results = [(notebook, output_path, 
    if output_path === nothing
        false  # Weave failed
    else
        !has_error_blocks(output_path)  # Check for error blocks
    end
) for (notebook, output_path) in results]

# Split results
successful_notebooks = [notebook for (notebook, _, success) in final_results if success]
failed_notebooks = [notebook for (notebook, _, success) in final_results if !success]

@info """
Processing Report:
Total notebooks: $(length(final_results))
Successful: $(length(successful_notebooks))
Failed: $(length(failed_notebooks))

$(isempty(successful_notebooks) ? "" : "\nSuccessfully processed notebooks:\n" * join(["  • " * notebook for notebook in successful_notebooks], "\n"))
$(isempty(failed_notebooks) ? "" : "\nFailed notebooks:\n" * join(["  • " * notebook for notebook in failed_notebooks], "\n"))
"""

if !isempty(failed_notebooks)
    @info """
    Some notebooks failed to process. This might be due to:
    1. Actual errors in the notebooks
    2. Cached results causing issues

    Try running 'make clean' to clear all build artifacts and cache,
    then run the build again. If the error persists, check the notebook contents.
    """
end

# Exit with error if any notebooks failed
exit(isempty(failed_notebooks) ? 0 : 1)