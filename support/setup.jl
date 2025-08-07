#!/usr/bin/env julia

"""
Setup script for RxInferExamples.jl

This script prepares the environment, installs dependencies, and optionally builds
examples and documentation. It is designed to run non-interactively by default and
fail fast when critical steps fail.

Run from repository root directory.

Options:
  --clean            Clean build cache before setup
  --no-examples      Skip building examples
  --no-docs          Skip building documentation
  --no-preview       Do not start documentation preview server
  --force            Continue on non-critical errors
  --quiet            Reduce output verbosity
  --convert          Run notebook -> script conversion incrementally
  --convert-all      Force conversion of all notebooks (no skip)
  --verify           Verify 1:1 notebook->script mapping after conversion
"""

function parse_args(args)
    opts = Dict{String,Any}(
        "clean" => false,
        "no_examples" => false,
        "no_docs" => false,
        "no_preview" => true, # default non-interactive
        "force" => false,
        "quiet" => false,
        "convert" => false,
        "convert_all" => false,
        "verify" => false,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--clean"
            opts["clean"] = true
        elseif a == "--no-examples"
            opts["no_examples"] = true
        elseif a == "--no-docs"
            opts["no_docs"] = true
        elseif a == "--no-preview"
            opts["no_preview"] = true
        elseif a == "--force"
            opts["force"] = true
        elseif a == "--quiet"
            opts["quiet"] = true
        elseif a == "--convert"
            opts["convert"] = true
        elseif a == "--convert-all"
            opts["convert_all"] = true
        elseif a == "--verify"
            opts["verify"] = true
        else
            error("Unknown option: $(a)")
        end
        i += 1
    end
    return (; clean = opts["clean"], no_examples = opts["no_examples"], no_docs = opts["no_docs"],
             no_preview = opts["no_preview"], force = opts["force"], quiet = opts["quiet"],
             convert = opts["convert"], convert_all = opts["convert_all"], verify = opts["verify"])
end

# Check if we're in the repository root
if !isfile("README.md")
    error("This script must be run from the repository root directory")
end

# Function to execute commands with error handling
function run_command(cmd, message; exit_on_error=false, show_output=true, quiet=false)
    quiet || println("\n== $message ==")
    try
        if show_output
            run(cmd)
        else
            # Capture output but don't display it unless there's an error
            output = IOBuffer()
            error_output = IOBuffer()
            process = run(pipeline(cmd, stdout=output, stderr=error_output), wait=true)
            
            if process.exitcode != 0
                quiet || println("Command failed with exit code $(process.exitcode)")
                quiet || println("Output:")
                quiet || println(String(take!(output)))
                quiet || println("Error:")
                quiet || println(String(take!(error_output)))
                if exit_on_error
                    error("Exiting due to command failure")
                end
            end
        end
        return true
    catch e
        quiet || println("Error executing command: $e")
        if exit_on_error
            error("Exiting due to command failure")
        end
        return false
    end
end

opts = parse_args(ARGS)

opts.quiet || println("Setting up RxInferExamples.jl...")

if opts.clean
    run_command(`make clean`, "Cleaning build cache"; quiet=opts.quiet)
end

# Update Julia via juliaup (non-fatal)
run_command(`juliaup update`, "Updating Julia via juliaup"; quiet=opts.quiet)

# Update default environment packages (non-fatal)
opts.quiet || println("\n== Updating default Julia environment packages ==")
try
    using Pkg
    Pkg.update()
catch e
    opts.quiet || println("Warning: Failed to update default environment: $(e)")
end

# Install required packages
opts.quiet || println("\n== Installing required packages ==")
using Pkg

# Add required packages with their versions specified to avoid compatibility issues
required_packages = [
    "Weave",
    "ImageInTerminal",  # Add missing dependency
    "JSON"  # For notebooks_to_scripts.jl
]

for pkg in required_packages
    opts.quiet || println("Installing $pkg...")
    try
        Pkg.add(pkg)
    catch e
        if opts.force
            opts.quiet || println("Warning: Failed to install $(pkg): $(e)")
        else
            rethrow(e)
        end
    end
end

# Set environment variables to help with graphics issues
ENV["GKSwstype"] = "100"  # Non-interactive backend
ENV["JULIA_COPY_STACKS"] = "1"  # More helpful error messages

# Check if examples project is properly set up
if !isfile("examples/Manifest.toml")
    opts.quiet || println("\n== Setting up examples project environment ==")
    run_command(`julia --project=examples -e 'using Pkg; Pkg.instantiate()'`, 
                "Initializing examples environment"; quiet=opts.quiet)
end

# Update examples environment packages
run_command(`julia --project=examples -e 'using Pkg; Pkg.update(); Pkg.instantiate()'`,
            "Updating examples environment packages"; quiet=opts.quiet)

# Optional conversion step
if opts.convert || opts.convert_all
    opts.quiet || println("\n== Converting notebooks to scripts ==")
    converter = joinpath("support", "notebooks_to_scripts.jl")
    flags = String[]
    if opts.convert_all
        push!(flags, "--force")
    else
        push!(flags, "--skip-existing")
    end
    if opts.verify
        push!(flags, "--verify")
    end
    if opts.quiet
        push!(flags, "--quiet")
    end
    convert_cmd = Cmd(vcat(["julia", converter], flags))
    run_command(convert_cmd, "Converting notebooks"; quiet=opts.quiet)
end

# Build examples with error handling
if !opts.no_examples
    opts.quiet || println("\n== Building examples ==")
    opts.quiet || println("Note: This may take several minutes. Build errors for individual examples will be logged but won't stop the process.")
    examples_success = run_command(`make examples`, "Building examples", exit_on_error=false; quiet=opts.quiet)
    if !examples_success && !opts.force
        error("Building examples failed. Rerun with --force to continue despite errors.")
    end
end

# Build documentation with error handling
if !opts.no_docs
    # Update docs environment packages prior to build
    run_command(`julia --project=docs -e 'using Pkg; Pkg.update(); Pkg.instantiate()'`,
                "Updating docs environment packages"; quiet=opts.quiet)
    opts.quiet || println("\n== Building documentation ==")
    docs_success = run_command(`make docs`, "Building documentation", exit_on_error=false; quiet=opts.quiet)
    if !docs_success && !opts.force
        error("Building documentation failed. Rerun with --force to continue despite errors.")
    end
    if docs_success
        opts.quiet || println("\n✅ Documentation built successfully!")
    end
end

# Ask before starting preview server
if !opts.no_docs && !opts.no_preview
    (opts.quiet) || println("\n== Starting documentation preview ==")
    run(`make preview`)
else
    opts.quiet || println("\nSkipping preview. You can start it later with: make preview")
end

opts.quiet || println("\nSetup complete!")
opts.quiet || println("To view the documentation, run: make preview")
opts.quiet || println("To convert notebooks to Julia scripts, run: julia support/notebooks_to_scripts.jl --skip-existing --verify") 