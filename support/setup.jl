#!/usr/bin/env julia

"""
Setup script for RxInferExamples.jl

This script updates Julia packages, installs required dependencies,
and builds the examples and documentation.
Run this script from the repository root directory.
"""

# Check if we're in the repository root
if !isfile("README.md")
    error("This script must be run from the repository root directory")
end

# Function to execute commands with error handling
function run_command(cmd, message; exit_on_error=false, show_output=true)
    println("\n== $message ==")
    try
        if show_output
            run(cmd)
        else
            # Capture output but don't display it unless there's an error
            output = IOBuffer()
            error_output = IOBuffer()
            process = run(pipeline(cmd, stdout=output, stderr=error_output), wait=true)
            
            if process.exitcode != 0
                println("Command failed with exit code $(process.exitcode)")
                println("Output:")
                println(String(take!(output)))
                println("Error:")
                println(String(take!(error_output)))
                if exit_on_error
                    error("Exiting due to command failure")
                end
            end
        end
        return true
    catch e
        println("Error executing command: $e")
        if exit_on_error
            error("Exiting due to command failure")
        end
        return false
    end
end

println("Setting up RxInferExamples.jl...")

# Check if we need to clean
clean_flag = any(arg -> arg == "--clean", ARGS)
if clean_flag
    println("\n== Cleaning build cache ==")
    run_command(`make clean`, "Cleaning build cache")
end

# Update Julia packages using juliaup
run_command(`juliaup update`, "Updating Julia via juliaup")

# Install required packages
println("\n== Installing required packages ==")
using Pkg

# Add required packages with their versions specified to avoid compatibility issues
required_packages = [
    "Weave",
    "ImageInTerminal",  # Add missing dependency
    "JSON"  # For notebooks_to_scripts.jl
]

for pkg in required_packages
    println("Installing $pkg...")
    Pkg.add(pkg)
end

# Set environment variables to help with graphics issues
ENV["GKSwstype"] = "100"  # Using a non-interactive backend
ENV["JULIA_COPY_STACKS"] = "1"  # More helpful error messages

# Check if examples project is properly set up
if !isfile("examples/Manifest.toml")
    println("\n== Setting up examples project environment ==")
    run_command(`julia --project=examples -e 'using Pkg; Pkg.instantiate()'`, 
                "Initializing examples environment")
end

# Build examples with error handling
println("\n== Building examples ==")
println("Note: This may take several minutes. Build errors for individual examples will be logged but won't stop the process.")

examples_success = run_command(`make examples`, "Building examples", exit_on_error=false)

if !examples_success
    println("\n⚠️  There were errors building some examples.")
    println("You can still proceed with building the documentation, but some examples might be missing.")
    println("To retry building just the examples, run: make clean && make examples")
    
    if !any(arg -> arg == "--force", ARGS)
        println("\nDo you want to continue building the documentation? (y/N)")
        response = lowercase(strip(readline()))
        if response != "y" && response != "yes"
            error("Setup aborted by user")
        end
    end
end

# Build documentation with error handling
println("\n== Building documentation ==")
docs_success = run_command(`make docs`, "Building documentation", exit_on_error=false)

if !docs_success
    println("\n⚠️  There were errors building the documentation.")
    println("You can still preview what was built successfully.")
else
    println("\n✅ Documentation built successfully!")
end

# Ask before starting preview server
if !any(arg -> arg == "--no-preview", ARGS)
    println("\nDo you want to start the documentation preview server? (Y/n)")
    response = lowercase(strip(readline()))
    if response == "" || response == "y" || response == "yes"
        println("\n== Starting documentation preview ==")
        run(`make preview`)
    else
        println("\nSkipping preview. You can start it later with: make preview")
    end
else
    println("\nSkipping preview as requested. You can start it later with: make preview")
end

println("\nSetup complete!")
println("To view the documentation, run: make preview")
println("To convert notebooks to Julia scripts, run: julia support/notebooks_to_scripts.jl") 