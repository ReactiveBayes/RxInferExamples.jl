using Pkg

# Ensure Weave is available
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Weave
using Distributed

# Add worker processes if none exist
if nworkers() == 1
    addprocs(max(1, Sys.CPU_THREADS ÷ 2))
end

# Load Weave on all workers
@everywhere using Weave
@everywhere using Pkg

# Define build directories
const BUILD_DIR = joinpath(@__DIR__, "build")
const FIG_DIR   = joinpath(BUILD_DIR, "figures")
const CACHE_DIR = joinpath(@__DIR__, "cache")

# Clear previous build
rm(BUILD_DIR; force=true, recursive=true)

# Create build directories
mkpath(BUILD_DIR)
mkpath(FIG_DIR)
mkpath(CACHE_DIR)

# Function to check markdown for error blocks
function has_error_blocks(md_path)
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
    
    # Activate and instantiate the project in the notebook's directory within the module
    Core.eval(mod, quote
        using Pkg
        Pkg.activate($notebook_dir)
        Pkg.instantiate()
    end)
    
    input_path = abspath(notebook_path)
    rel_path = relpath(notebook_path, dirname(build_dir))
    output_path = joinpath(build_dir, replace(rel_path, ".ipynb" => ".md"))
    
    # Create notebook-specific cache directory
    notebook_cache_dir = joinpath(cache_dir, dirname(rel_path))
    mkpath(notebook_cache_dir)
    
    # Ensure output directory exists
    mkpath(dirname(output_path))
    
    @info "Processing $rel_path..."
    try
        weave(input_path;
            out_path=output_path,
            doctype="github",
            fig_path=fig_dir,
            cache=:all,
            cache_path=notebook_cache_dir,
            mod=mod
        )
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
    for file in files
        if endswith(file, ".ipynb")
            push!(notebook_files, relpath(joinpath(root, file), @__DIR__))
        end
    end
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

Successfully processed notebooks:
$(join(["  • " * notebook for notebook in successful_notebooks], "\n"))

Failed notebooks:
$(join(["  • " * notebook for notebook in failed_notebooks], "\n"))
"""

# Exit with error if any notebooks failed
exit(isempty(failed_notebooks) ? 0 : 1)
