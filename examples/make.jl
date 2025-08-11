using Pkg
using Distributed
using Base.Filesystem
using ArgParse

# Parse command line arguments
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--use-dev"
        action = :store_true
        help = "Use local development version of RxInfer"
        "--use-cache"
        action = :store_true
        help = "Use cached results for notebooks"
        "--rxinfer-path"
        help = "Path to RxInfer.jl repository (overrides default ../RxInfer.jl)"
        "--strict-env"
        action = :store_true
        help = "Fail if required environment variables are missing (use for CI)"
        "filter"
        help = "Filter pattern for notebook names"
        required = false
    end

    return parse_args(s)
end

const ARGS = parse_commandline()
const FILTER = get(ARGS, "filter", nothing)
const USE_DEV = ARGS["use-dev"]
const USE_CACHE = ARGS["use-cache"]
const STRICT_ENV = ARGS["strict-env"]
const CUSTOM_RXINFER_PATH = get(ARGS, "rxinfer-path", nothing)

const RXINFER_PATH = if USE_DEV
    # Try to find local RxInfer development version
    rxinfer_dev = if !isnothing(CUSTOM_RXINFER_PATH)
        CUSTOM_RXINFER_PATH
    else
        joinpath(dirname(@__DIR__), "..", "RxInfer.jl")
    end

    if isdir(rxinfer_dev)
        @info "Using local development version of RxInfer from $(rxinfer_dev)"
        abspath(rxinfer_dev)
    else
        @error """
        Could not find local RxInfer development version at $(rxinfer_dev).
        Please either:
        1. Clone RxInfer.jl repository next to RxInferExamples.jl
        2. Specify path with --rxinfer-path
        3. Remove --use-dev flag to use released version
        """
        exit(-1)
    end
else
    nothing
end

# Activate and setup environment
Pkg.activate(@__DIR__)

Pkg.instantiate()

using Base.Filesystem

# Add worker processes if none exist
if nworkers() == 1
    addprocs(max(1, Sys.CPU_THREADS ÷ 2))
end

# Load Weave and Pkg on all workers
@everywhere using Weave
@everywhere using Pkg

# Define build directories
const SOURCE_DIR = abspath(joinpath(@__DIR__, "..", "docs", "src"))
const BUILD_DIR = abspath(joinpath(SOURCE_DIR, "categories"))
const CACHE_DIR = joinpath(BUILD_DIR, "_cache")

# Create build directories if dont exist
mkpath(BUILD_DIR)
mkpath(CACHE_DIR)

@info """
Build directories:
SOURCE_DIR: $SOURCE_DIR
BUILD_DIR: $BUILD_DIR
CACHE_DIR: $CACHE_DIR
"""

# Function to copy files with exceptions
@everywhere function copy_files(src_dir, dst_dir; exclude=[])
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

# Function to check if required environment variables are available
@everywhere function check_env_requirements(meta)
    if !hasfield(typeof(meta), :env_required) || isnothing(meta.env_required)
        return true, nothing
    end

    missing_vars = String[]
    for var in meta.env_required
        if isempty(get(ENV, var, ""))
            push!(missing_vars, var)
        end
    end

    if !isempty(missing_vars)
        return false, missing_vars
    end

    return true, nothing
end

# Function to check markdown for error blocks
@everywhere function has_error_blocks(md_path)
    !isfile(md_path) && return true
    mdtext = read(md_path, String)

    erroridx = findnext("```\nError:", mdtext, 1)
    if !isnothing(erroridx)
        # Extract error context window
        errwindow = 500
        errstart = reduce((idx, _) -> max(firstindex(mdtext), prevind(mdtext, idx)), 1:errwindow; init=first(erroridx))
        errend = reduce((idx, _) -> min(lastindex(mdtext), nextind(mdtext, idx)), 1:errwindow; init=last(erroridx))

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
    pattern = r"!\[(.*?)\]\((/.*?/docs/src/categories/[^\)]+)\)"

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

# Function to replace hidden block markers with HTML details tags
@everywhere function replace_hidden_blocks(content)
    # Pattern to match code blocks with hidden block markers
    # This regex captures: 
    # - The opening code fence (```julia)
    # - The hidden block start marker and its summary text
    # - The code content
    # - The hidden block end marker
    # - The closing code fence (```)
    pattern = r"(```[a-z]*\n)### EXAMPLE_HIDDEN_BLOCK_START\((.*?)\) ###(.*?)### EXAMPLE_HIDDEN_BLOCK_END ###\n(```)"s

    # Replace the matched pattern with HTML details tags around the code block
    new_content = replace(content, pattern => s -> begin
        matches = match(pattern, s)

        code_fence_open = matches.captures[1]  # ```julia\n
        summary_text = matches.captures[2]     # The summary text
        code_content = matches.captures[3]     # The code between markers
        code_fence_close = matches.captures[4] # ```

        # Add a tab character to each line of the code content
        code_content = join(filter(line -> !occursin(r"#\s*hide", line), map(line -> "\t" * line, eachline(IOBuffer(code_content)))), "\n")

        "!!! details \"Hidden block of $(summary_text) - click to expand\"\n\t$(code_fence_open)$(code_content)\n\t$(code_fence_close)"
    end)

    return new_content
end

# Function to process a single notebook
@everywhere function process_notebook(notebook_path, build_dir, cache_dir, rxinfer_path=nothing)
    # Get the notebook's directory and activate its environment
    notebook_dir = dirname(notebook_path)
    notebook_name = basename(notebook_path)

    # Setup paths
    rel_path = relpath(notebook_path, @__DIR__)
    output_path = joinpath(build_dir, replace(lowercase(dirname(rel_path)), " " => "_"), "index.md")
    output_dir = dirname(output_path)
    build_input_path = joinpath(output_dir, notebook_name)

    @info """
    Notebook directory: $notebook_dir
    Files in the notebook directory: $(readdir(notebook_dir))
    Output directory: $output_dir
    """

    # Create directories if dont exist
    mkpath(output_dir)

    # Copy all the files (except for `Manifest.toml`) from the notebook directory 
    # to the build directory
    copy_files(notebook_dir, output_dir; exclude=["Manifest.toml"])

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
    end)

    if !isnothing(rxinfer_path)
        Core.eval(mod, quote
            @info "Adding development version of RxInfer to notebook environment"
            Pkg.develop(path=$rxinfer_path)
        end)
    end

    Core.eval(mod, quote
        Pkg.instantiate()
    end)

    Core.eval(mod, quote
        envio = open(joinpath($output_dir, "env.log"), "w")
        Pkg.status(io=envio)
        flush(envio)
        close(envio)
    end)

    @info "Processing $rel_path..."
    try
        if !isfile(joinpath(output_dir, "meta.jl"))
            error("The example is missing a meta.jl file. See contributing guide for more information.")
        end

        meta = include(joinpath(output_dir, "meta.jl"))

        # Check environment variable requirements
        env_ok, missing_vars = check_env_requirements(meta)
        if !env_ok
            if $STRICT_ENV
                error("Missing required environment variables: $(join(missing_vars, ", ")). Set STRICT_ENV=false to skip examples with missing env vars.")
            else
                @warn "Skipping $(basename(notebook_path)) - missing required environment variables: $(join(missing_vars, ", "))"
                return nothing
            end
        end

        weave(build_input_path;
            out_path=output_path,
            doctype="github",
            fig_path=output_dir,
            cache=ifelse($USE_CACHE, :all, :refresh),
            cache_path=notebook_cache_dir,
            mod=mod
        )

        # Fix any absolute image paths in the generated markdown
        fix_image_paths(output_path)

        # Read the existing content
        content = read(output_path, String)

        # Replace hidden block markers with HTML details tags
        content = replace_hidden_blocks(content)

        EXAMPLE_METADATA = """
        ```@meta
        EditURL="https://github.com/ReactiveBayes/RxInferExamples.jl"
        Description=\"\"\"
        $(meta.title) with RxInfer.jl
        $(meta.description)
        Check more examples and tutorials at https://examples.rxinfer.com
        \"\"\"
        ```
        """

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

        # Read the environment log content
        env_log_content = read(joinpath(output_dir, "env.log"), String)

        ENVIRONMENT_NOTE = """
        !!! compat "Environment"
            This example was executed in a clean, isolated environment. Below are the exact package versions used:
            
            For reproducibility:
            - Use the same package versions when running locally
            - Report any issues with package compatibility

        ```
        $(env_log_content)
        ```
        """

        # Write the contribution note at the beginning
        write(output_path,
            """
            $EXAMPLE_METADATA
            $CONTRIBUTING_NOTE
            ---
            $content
            ---
            $CONTRIBUTING_NOTE
            ---
            $ENVIRONMENT_NOTE
            """)

        @info "Successfully processed $rel_path"
        return output_path
    catch e
        @error "Failed to process $rel_path" exception = (e, catch_backtrace())
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
        CACHE_DIR,
        RXINFER_PATH
    )), notebook_files)

is_processing_failed(output_path) = isnothing(output_path) || has_error_blocks(output_path)

# Check for errors in output files
final_results = [(notebook, output_path, is_processing_failed(output_path)) for (notebook, output_path) in results]

# Split results into successful, failed, and skipped
successful_notebooks = [notebook for (notebook, output_path, failed) in final_results if !failed && !isnothing(output_path)]
failed_notebooks = [notebook for (notebook, output_path, failed) in final_results if failed && !isnothing(output_path)]
skipped_notebooks = [notebook for (notebook, output_path, _) in final_results if isnothing(output_path)]

@info """
Processing Report:
Total notebooks: $(length(final_results))
Successful: $(length(successful_notebooks))
Failed: $(length(failed_notebooks))
Skipped: $(length(skipped_notebooks))

$(isempty(successful_notebooks) ? "" : "Successfully processed notebooks:\n" * join(["  • " * notebook for notebook in successful_notebooks], "\n"))
$(isempty(failed_notebooks) ? "" : "Failed notebooks:\n" * join(["  • " * notebook for notebook in failed_notebooks], "\n"))
$(isempty(skipped_notebooks) ? "" : "Skipped notebooks:\n" * join(["  • " * notebook for notebook in skipped_notebooks], "\n"))
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

if !isempty(skipped_notebooks)
    @warn """
    Some notebooks were skipped due to missing environment variables.
    To run these examples, set the required environment variables:
    $(join(unique([join(meta.env_required, ", ") for meta in [include(joinpath(@__DIR__, dirname(notebook), "meta.jl")) for notebook in skipped_notebooks] if hasfield(typeof(meta), :env_required) && !isnothing(meta.env_required)]), "\n"))

    Or use --strict-env flag to fail instead of skip.
    """
end

if isempty(failed_notebooks)
    @info """
    Notebooks processed successfully! 🎉 $(isempty(skipped_notebooks) ? "" : "(Some notebooks were skipped due to missing environment variables. See above for details.)")

    Next steps:
    Run `make docs` to generate the documentation website.
    """
end

# Print resulting file structure
function print_dir_structure(dir, prefix="")
    # Get all markdown files in current directory
    md_files = filter(f -> endswith(f, ".md"), readdir(dir))
    subdirs = sort(filter(d -> isdir(joinpath(dir, d)) && !startswith(d, "_"), readdir(dir)))

    @info "$(prefix)📁 $(basename(dir))/"
    # Print current directory if it has markdown files
    if !isempty(md_files)
        for file in sort(md_files)
            @info "$(prefix)  📄 $file"
        end
    end

    # Always process subdirectories if they exist
    for subdir in subdirs
        print_dir_structure(joinpath(dir, subdir), prefix * "  ")
    end
end

@info """
Generated source directory structure:
"""
print_dir_structure(SOURCE_DIR)

# Exit with error if any notebooks failed
exit(isempty(failed_notebooks) ? 0 : 1)
