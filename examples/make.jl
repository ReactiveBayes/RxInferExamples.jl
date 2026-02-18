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
        "--workers"
        help = "Amount of workers to use to build the examples (defaults to Sys.CPU_THREADS)"
        default = Sys.CPU_THREADS
        "filter"
        help = "Filter pattern for notebook names"
        required = false
    end

    return parse_args(s)
end

const ARGS = parse_commandline()
const NWORKERS = get(ARGS, "workers", Sys.CPU_THREADS)
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

## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
ENV["GKSwstype"] = "100"

# Activate and setup environment
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Base.Filesystem

# Add worker processes if none exist
if nworkers() == 1
    addprocs(max(1, NWORKERS))
end

# Load Weave and Pkg on all workers
@everywhere using Weave
@everywhere using Pkg

## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
@everywhere ENV["GKSwstype"] = "100"

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
    # Empty directory at the given path 
    rm(dst_dir, force=true, recursive=true)

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

@everywhere struct ProcessedNotebook
    notebook::String
    notebook_path::String
    output_path::String
    result::Symbol
    time_to_instantiate::Float64
    time_to_build::Float64
    total_time::Float64
end

is_processing_failed(processed::ProcessedNotebook) =
    processed.result !== :skipped &&
    (processed.result === :failed || isnothing(processed.output_path) || has_error_blocks(processed.output_path))
is_processing_skipped(processed::ProcessedNotebook) =
    processed.result === :skipped
is_processing_successful(processed::ProcessedNotebook) =
    !is_processing_skipped(processed) && !is_processing_failed(processed)

@everywhere ns_to_seconds(nanoseconds) = nanoseconds / 10^9

@everywhere function execute_in_julia_subprocess(envdir, command)
    run(`julia --startup-file=no --threads=2,1 --gcthreads=2,1 --project=$envdir -e $command`)
end

# Function to process a single notebook
@everywhere function process_notebook(envdir, notebook, build_dir, cache_dir)::ProcessedNotebook
    # Get the notebook's directory and activate its environment
    notebook_path = joinpath(@__DIR__, notebook)
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

    # The cache strategy is inlined inside of `@everywhere`
    cache_strategy = ifelse($USE_CACHE, :all, :refresh)

    # Change to output directory before activating
    cd(output_dir)
    @info "Working directory changed to: $(pwd())"

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
                return ProcessedNotebook(notebook, notebook_path, output_path, :skipped, 0.0, 0.0, 0.0)
            end
        end

        # Activate the project in the current directory
        time_to_instantiate_begin = time_ns()
        notebook_dependencies = Pkg.activate(notebook_dir) do 
            collect(keys(Pkg.project().dependencies))
        end
        execute_in_julia_subprocess(envdir, quote
            using Pkg
            envio = open(joinpath($output_dir, "env.log"), "w")
            Pkg.status($notebook_dependencies, io=envio)
            flush(envio)
            close(envio)
        end)
        time_to_instantiate_end = time_ns()

        time_to_build_begin = time_ns()
        execute_in_julia_subprocess(envdir, quote
            using Serialization
            using Weave
            weave($build_input_path;
                out_path=$output_path,
                doctype="github",
                fig_path=$output_dir,
                cache=$(QuoteNode(cache_strategy)),
                cache_path=$notebook_cache_dir,
            )
        end)
        time_to_build_end = time_ns()

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
        time_to_instantiate = ns_to_seconds(time_to_instantiate_end - time_to_instantiate_begin)
        time_to_build = ns_to_seconds(time_to_build_end - time_to_build_begin)
        return ProcessedNotebook(
            notebook,
            notebook_path,
            output_path,
            :success,
            time_to_instantiate,
            time_to_build,
            time_to_instantiate + time_to_build
        )
    catch e
        @error "Failed to process $rel_path" exception = (e, catch_backtrace())
        return ProcessedNotebook(notebook, notebook_path, output_path, :failed, 0.0, 0.0, 0.0)
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

notebook_directories = map(f -> joinpath(@__DIR__, dirname(f)), notebook_files)
temporary_environment = mktempdir(cleanup=true)
# We start with the current environment
temporary_environment_dependencies = Pkg.project().dependencies

@info """
Creating temporary environment to run examples from at $temporary_environment.
The environment will be deleted automatically after the process exits.
"""
for notebook_directory in notebook_directories
    new_dependencies = Pkg.activate(notebook_directory) do
        Pkg.project().dependencies
    end
    @info notebook_directory, new_dependencies
    merge!(
        temporary_environment_dependencies,
        new_dependencies
    )
end

Pkg.activate(temporary_environment) do
    Pkg.add(collect(keys(temporary_environment_dependencies)))
    Pkg.update()
    Pkg.precompile()
end

if !isnothing(RXINFER_PATH)
    Pkg.activate(temporary_environment) do
        @info "Adding development version of RxInfer to the temporary environment"
        Pkg.develop(path = RXINFER_PATH)
        Pkg.update()
        Pkg.precompile()
    end
end

# Process notebooks in parallel
results = pmap(
    notebook -> process_notebook(
        temporary_environment,
        notebook,
        BUILD_DIR,
        CACHE_DIR
    ),
    notebook_files
)

Pkg.activate(temporary_environment) do 
    Pkg.rm(collect(keys(temporary_environment_dependencies)))
    Pkg.gc()
end

# Split results into successful, failed, and skipped
successful_results = filter(is_processing_successful, results)
failed_results = filter(is_processing_failed, results)
skipped_results = filter(is_processing_skipped, results)

function prettify_notebook_string(result::ProcessedNotebook)
    return string(
        result.notebook,
        " (total: ", round(result.total_time, digits=2), " seconds, ",
        "instantiate: ", round(result.time_to_instantiate, digits=2), " seconds, ",
        "build: ", round(result.time_to_build, digits=2), " seconds)"
    )
end

@info """
Processing Report:
Total notebooks: $(length(results))
Successful: $(length(successful_results))
Failed: $(length(failed_results))
Skipped: $(length(skipped_results))

$(isempty(successful_results) ? "" : "Successfully processed notebooks:\n" * join(["  • " * prettify_notebook_string(result) for result in successful_results], "\n"))
$(isempty(failed_results) ? "" : "Failed notebooks:\n" * join(["  • " * result.notebook for result in failed_results], "\n"))
$(isempty(skipped_results) ? "" : "Skipped notebooks:\n" * join(["  • " * result.notebook for result in skipped_results], "\n"))
"""

longest_to_execute_show = 10
longest_to_execute_notebooks = Iterators.take(sort(successful_results, by=(r) -> -r.total_time), longest_to_execute_show)

@info """
Performance Report:
Top ($longest_to_execute_show) longest to execute notebooks:
$(join(["  • " * prettify_notebook_string(result) for result in longest_to_execute_notebooks], "\n"))
"""

if !isempty(failed_results)
    @warn """
    Some notebooks failed to process. This might be due to:
    1. Actual errors in the notebooks
    2. Cached results causing issues

    Try running 'make clean' to clear all build artifacts and cache,
    then run the build again. If the error persists, check the notebook contents.
    """
end

if !isempty(skipped_results)
    @warn """
    Some notebooks were skipped due to missing environment variables.
    To run these examples, set the required environment variables:
    $(join(unique([join(meta.env_required, ", ") for meta in [include(joinpath(@__DIR__, dirname(result.notebook), "meta.jl")) for result in skipped_results] if hasfield(typeof(meta), :env_required) && !isnothing(meta.env_required)]), "\n"))

    Or use --strict-env flag to fail instead of skip.
    """
end

if isempty(failed_results)
    @info """
    Notebooks processed successfully! 🎉 $(isempty(skipped_results) ? "" : "(Some notebooks were skipped due to missing environment variables. See above for details.)")

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
exit(isempty(failed_results) ? 0 : 1)
