#!/usr/bin/env julia

"""
Convert Jupyter notebooks in the examples/ folder to standalone Julia scripts.

This script:
1. Recursively walks through the examples/ directory
2. Finds all Jupyter notebook files (.ipynb)
3. Extracts Julia code from each notebook
4. Creates a parallel scripts/ directory structure with .jl files

Usage: Run this script from the repository root directory
```
julia support/notebooks_to_scripts.jl
```
"""

using JSON
using Base.Filesystem
using Dates  # Add Dates module import

# Ensure we're running from the project root
if !isdir("examples")
    error("This script must be run from the repository root that contains the 'examples' directory")
end

# Create output directory if it doesn't exist
output_dir = "scripts"
mkpath(output_dir)

# Function to extract Julia code from a notebook
function extract_code_from_notebook(notebook_path)
    # Read the notebook file
    notebook_content = JSON.parsefile(notebook_path)
    
    # Extract the cells from the notebook
    cells = get(notebook_content, "cells", [])
    
    code_blocks = String[]
    
    for cell in cells
        # Check if it's a code cell
        if get(cell, "cell_type", "") == "code"
            # Extract the source code
            source = get(cell, "source", [])
            if !isempty(source)
                # Join the lines of code
                code = join(source, "")
                # Skip empty cells or cells with just comments
                if !isempty(strip(code)) && !all(startswith.(split(code, "\n"), "#"))
                    push!(code_blocks, code)
                end
            end
        end
    end
    
    # Create a Julia script with extracted code
    return join(code_blocks, "\n\n")
end

# Function to convert notebook path to script path
function notebook_to_script_path(notebook_path)
    # Replace the base directory
    script_path = replace(notebook_path, "examples/" => "scripts/")
    
    # Replace the file extension
    script_path = replace(script_path, r"\.ipynb$" => ".jl")
    
    return script_path
end

# Function to copy Project.toml and meta.jl if they exist
function copy_project_files(source_dir, target_dir)
    for file in ["Project.toml", "meta.jl"]
        source_file = joinpath(source_dir, file)
        if isfile(source_file)
            target_file = joinpath(target_dir, file)
            cp(source_file, target_file, force=true)
            println("  Copied $file")
        end
    end
end

# Walk through examples directory and convert notebooks
global notebooks_processed = 0  # Declare as global to fix scope warning
println("Converting notebooks to Julia scripts...")

for (root, dirs, files) in walkdir("examples")
    # Skip the root examples directory itself
    if root == "examples"
        continue
    end
    
    # Create corresponding directory in scripts/
    target_dir = replace(root, "examples/" => "scripts/")
    mkpath(target_dir)
    
    # Copy Project.toml and meta.jl if they exist
    copy_project_files(root, target_dir)
    
    # Process all notebook files
    for file in files
        if endswith(file, ".ipynb")
            notebook_path = joinpath(root, file)
            script_path = notebook_to_script_path(notebook_path)
            
            # Extract code and write to script file
            try
                println("Converting $(notebook_path)")
                code = extract_code_from_notebook(notebook_path)
                
                # Add header with source notebook information
                header = """
                # This file was automatically generated from $(notebook_path)
                # by $(basename(@__FILE__)) at $(Dates.now())
                #
                # Source notebook: $(file)
                
                """
                
                # Write the code to the script file
                mkpath(dirname(script_path))
                open(script_path, "w") do io
                    write(io, header * code)
                end
                
                global notebooks_processed += 1  # Use global keyword for assignment
            catch e
                println("  Error processing $notebook_path: $e")
            end
        end
    end
end

println("Conversion complete. Processed $notebooks_processed notebooks.")
println("Standalone Julia scripts are available in the '$output_dir' directory.") 