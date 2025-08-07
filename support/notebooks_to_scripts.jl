#!/usr/bin/env julia

"""
Convert Jupyter notebooks in the `examples/` folder to standalone Julia scripts in `scripts/`.

Features:
- Recursively walks `examples/`
- Extracts Julia code cells from each `.ipynb`
- Writes mirrored `scripts/` structure with `.jl` files
- Copies `Project.toml` and `meta.jl` alongside generated scripts
- Incremental mode (skip up-to-date outputs)
- Dry-run, filtering, and verification options

Usage:
  julia support/notebooks_to_scripts.jl [options]

Options:
  --dry-run            Show what would be converted without writing files
  --skip-existing      Skip if target `.jl` is newer than or same age as source
  --force              Re-generate scripts even if up-to-date
  --filter SUBSTR      Only convert notebooks whose path contains SUBSTR
  --verify             After conversion, verify 1:1 mapping
  --quiet              Reduce output noise
  --list               List notebooks that would be processed

Exit status: non-zero if any error occurs or verification fails (when --verify)
"""

using JSON
using Base.Filesystem
using Dates

const script_dir = dirname(abspath(@__FILE__))
const repo_root = dirname(script_dir)
const examples_dir = joinpath(repo_root, "examples")
const output_dir = joinpath(repo_root, "scripts")

if !isdir(examples_dir)
    error("Cannot find 'examples' directory in the repository root at $(repo_root)")
end

mkpath(output_dir)

"""
Return a NamedTuple of parsed CLI options.
"""
function parse_args(args::Vector{String})
    options = Dict{String,Any}(
        "dry_run" => false,
        "skip_existing" => false,
        "force" => false,
        "filter" => nothing,
        "verify" => false,
        "quiet" => false,
        "list" => false,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--dry-run"
            options["dry_run"] = true
        elseif a == "--skip-existing"
            options["skip_existing"] = true
        elseif a == "--force"
            options["force"] = true
        elseif a == "--verify"
            options["verify"] = true
        elseif a == "--quiet"
            options["quiet"] = true
        elseif a == "--list"
            options["list"] = true
        elseif a == "--filter"
            i += 1
            i > length(args) && error("--filter requires a value after --filter")
            options["filter"] = args[i]
        else
            error("Unknown option: $(a)")
        end
        i += 1
    end
    return (; dry_run = options["dry_run"],
             skip_existing = options["skip_existing"],
             force = options["force"],
             filter = options["filter"],
             verify = options["verify"],
             quiet = options["quiet"],
             list = options["list"])
end

"""
Extract Julia code from a Jupyter notebook file.
Returns a single string with code blocks separated by blank lines.
"""
function extract_code_from_notebook(notebook_path::AbstractString)::String
    notebook_content = JSON.parsefile(notebook_path)
    cells = get(notebook_content, "cells", [])
    code_blocks = String[]
    for cell in cells
        if get(cell, "cell_type", "") == "code"
            source = get(cell, "source", [])
            if !isempty(source)
                code = join(source, "")
                if !isempty(strip(code)) && !all(startswith.(split(code, "\n"), "#"))
                    push!(code_blocks, code)
                end
            end
        end
    end
    return join(code_blocks, "\n\n")
end

"""
Map a notebook path in `examples/` to its output script path in `scripts/`.
"""
function notebook_to_script_path(notebook_path::AbstractString)
    script_path = replace(notebook_path, examples_dir => output_dir)
    return replace(script_path, r"\.ipynb$" => ".jl")
end

"""
Copy `Project.toml` and `meta.jl` from the source dir to the target dir if present.
"""
function copy_project_files(source_dir::AbstractString, target_dir::AbstractString; quiet::Bool=false)
    for file in ("Project.toml", "meta.jl")
        source_file = joinpath(source_dir, file)
        if isfile(source_file)
            target_file = joinpath(target_dir, file)
            cp(source_file, target_file, force=true)
            quiet || println("  Copied $(file)")
        end
    end
end

"""
Return true if the target script is considered up-to-date with the notebook.
"""
function up_to_date(notebook_path::AbstractString, script_path::AbstractString)::Bool
    return isfile(script_path) && (mtime(script_path) >= mtime(notebook_path))
end

function should_process(nb::AbstractString; filter::Union{Nothing,String})
    return filter === nothing || occursin(filter::String, nb)
end

function conversion_header(notebook_path::AbstractString, source_file::AbstractString)
    return """
# This file was automatically generated from $(notebook_path)
# by $(basename(@__FILE__)) at $(Dates.now())
# Do not edit by hand. Edit the notebook instead.
#
# Source notebook: $(source_file)

"""
end

function main()
    opts = parse_args(ARGS)

    opts.quiet || println("Converting notebooks to Julia scripts...")
    notebooks_processed = 0
    notebooks_skipped = 0
    errors = 0
    planned = String[]

    for (root, _dirs, files) in walkdir(examples_dir)
        root == examples_dir && continue
        target_dir = replace(root, examples_dir => output_dir)
        mkpath(target_dir)
        copy_project_files(root, target_dir; quiet=opts.quiet)
        for file in files
            endswith(file, ".ipynb") || continue
            notebook_path = joinpath(root, file)
            should_process(notebook_path; filter=opts.filter) || continue
            script_path = notebook_to_script_path(notebook_path)
            if opts.skip_existing && !opts.force && up_to_date(notebook_path, script_path)
                notebooks_skipped += 1
                continue
            end
            push!(planned, notebook_path)
            if opts.list
                println(notebook_path)
            end
        end
    end

    if opts.list && opts.dry_run
        return 0
    end

    for notebook_path in planned
        try
            script_path = notebook_to_script_path(notebook_path)
            if !opts.dry_run
                opts.quiet || println("Converting $(notebook_path)")
                code = extract_code_from_notebook(notebook_path)
                if isempty(strip(code))
                    opts.quiet || println("  Skipped (no code cells): $(notebook_path)")
                    notebooks_skipped += 1
                    continue
                end
                header = conversion_header(notebook_path, basename(notebook_path))
                mkpath(dirname(script_path))
                open(script_path, "w") do io
                    write(io, header * code)
                end
            end
            notebooks_processed += 1
        catch e
            println("  Error processing $(notebook_path): $(e)")
            errors += 1
        end
    end

    if opts.verify
        missing = String[]
        for (root, _dirs, files) in walkdir(examples_dir)
            for f in files
                endswith(f, ".ipynb") || continue
                nb = joinpath(root, f)
                opts.filter === nothing || occursin(opts.filter, nb) || continue
                sp = notebook_to_script_path(nb)
                if !isfile(sp)
                    push!(missing, sp)
                end
            end
        end
        if !isempty(missing)
            println("Verification failed. Missing $(length(missing)) scripts:")
            for m in missing
                println("  " * m)
            end
            return 2
        end
    end

    opts.quiet || println("Conversion complete. Processed $(notebooks_processed) notebooks. Skipped $(notebooks_skipped).")
    opts.quiet || println("Standalone Julia scripts are available in the '$(output_dir)' directory.")
    return errors == 0 ? 0 : 1
end

exit(main())