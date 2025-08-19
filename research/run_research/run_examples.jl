#!/usr/bin/env julia

using Logging
using Printf
using Dates
using Pkg

const EMJ_INFO = "ℹ️"; const EMJ_OK = "✅"; const EMJ_WARN = "⚠️"; const EMJ_ERR = "❌";
const EMJ_ROCKET = "🚀"; const EMJ_TOOLS = "🛠️";

"""
run_examples.jl

- Reads an optional YAML config (path passed via --config) to decide which research runs to execute
- Instantiates per-research project environments before running
- Provides consistent, emoji-rich logging

YAML toggles (runs.*):
  - gc_single, gc_suite, gcn_single, gcn_suite, hgf => Bool
Default YAML path: `research/run_research/run_config.yaml` (falls back to `research/hgf/run_config.yaml` if missing).
"""
struct Step
    name::String
    path::String
    project::String
    enabled::Bool
end

function read_config(args)
    cfg_path = nothing
    for (i,a) in enumerate(args)
        if a == "--config" && i < length(args)
            cfg_path = args[i+1]
            break
        end
    end
    return cfg_path
end

yaml_read_bool(line) = lowercase(replace(strip(split(line, ":", limit=2)[2]), '"' => '')) in ("1","true","yes","on")

function load_yaml_config(path::AbstractString)
    cfg = Dict{String,Any}()
    cfg["runs"] = Dict{String,Bool}(
        "gc_single" => true,
        "gc_suite" => true,
        "gcn_single" => true,
        "gcn_suite" => true,
        "hgf" => true,
    )
    if !isfile(path)
        return cfg
    end
    for line in eachline(path)
        l = strip(line)
        isempty(l) && continue
        startswith(l, "#") && continue
        occursin("runs.gc_single:", l) && (cfg["runs"]["gc_single"] = yaml_read_bool(l))
        occursin("runs.gc_suite:", l)  && (cfg["runs"]["gc_suite"]  = yaml_read_bool(l))
        occursin("runs.gcn_single:", l) && (cfg["runs"]["gcn_single"] = yaml_read_bool(l))
        occursin("runs.gcn_suite:", l)  && (cfg["runs"]["gcn_suite"]  = yaml_read_bool(l))
        occursin("runs.hgf:", l)       && (cfg["runs"]["hgf"]       = yaml_read_bool(l))
    end
    return cfg
end

banner(msg) = println("\n$EMJ_ROCKET  $(msg)  $(repeat("=", max(0, 64 - length(msg))))\n")

function run_step(step::Step)
    step.enabled || (println("$EMJ_INFO  Skipping disabled step: $(step.name)" ); return true)
    banner("Running $(step.name)")
    println("$EMJ_INFO  Project: $(step.project)")
    println("$EMJ_INFO  Script : $(step.path)")
    t0 = time()
    try
        run(`julia --project=$(step.project) $(step.path)`)
        println("$EMJ_OK  Completed $(step.name) in $(round(time() - t0, digits=2))s")
        return true
    catch e
        println("$EMJ_ERR  Failed $(step.name): $(e)")
        return false
    end
end

function main()
    cfg_path = read_config(ARGS)
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    # Prefer run_research config by default; accept legacy hgf config if passed explicitly
    default_yaml = joinpath(repo_root, "research", "run_research", "run_config.yaml")
    fallback_yaml = joinpath(repo_root, "research", "hgf", "run_config.yaml")
    yaml_path = coalesce(cfg_path, isfile(default_yaml) ? default_yaml : fallback_yaml)
    cfg = load_yaml_config(yaml_path)

    steps = Step[
        Step("Generalized Coordinates (single)", joinpath(repo_root, "research", "generalized_coordinates", "run_gc_car.jl"), joinpath(repo_root, "research", "generalized_coordinates"), cfg["runs"]["gc_single"]),
        Step("Generalized Coordinates (suite)", joinpath(repo_root, "research", "generalized_coordinates", "run_gc_suite.jl"), joinpath(repo_root, "research", "generalized_coordinates"), cfg["runs"]["gc_suite"]),
        Step("Generalized Coordinates n-order (single)", joinpath(repo_root, "research", "generalized_coordinates_n_order", "run_gc_car.jl"), joinpath(repo_root, "research", "generalized_coordinates_n_order"), cfg["runs"]["gcn_single"]),
        Step("Generalized Coordinates n-order (suite)", joinpath(repo_root, "research", "generalized_coordinates_n_order", "run_gc_suite.jl"), joinpath(repo_root, "research", "generalized_coordinates_n_order"), cfg["runs"]["gcn_suite"]),
        Step("HGF end-to-end", joinpath(repo_root, "research", "hgf", "run_hgf.jl"), joinpath(repo_root, "research", "hgf"), cfg["runs"]["hgf"]),
    ]

    println("$EMJ_TOOLS  Ensuring research project envs are instantiated…")
    for proj in unique(s.project for s in steps)
        if isfile(joinpath(proj, "Project.toml"))
            try
                run(`julia --project=$(proj) -e 'using Pkg; Pkg.instantiate()'`)
            catch e
                println("$EMJ_WARN  Instantiation warning for $(proj): $(e)")
            end
        end
    end

    failures = String[]
    for s in steps
        isfile(s.path) || (println("$EMJ_WARN  Skipping missing script: $(s.path)"); continue)
        ok = run_step(s)
        ok || push!(failures, s.name)
    end

    if isempty(failures)
        println("\n$EMJ_OK  All research runs completed successfully")
        return 0
    else
        println("\n$EMJ_ERR  Some runs failed: ", join(failures, ", "))
        return 1
    end
end

exit(main())

#!/usr/bin/env julia

using Logging
using Printf
using Dates
using Pkg

const EMJ_INFO = "ℹ️"; const EMJ_OK = "✅"; const EMJ_WARN = "⚠️"; const EMJ_ERR = "❌";
const EMJ_ROCKET = "🚀"; const EMJ_TOOLS = "🛠️";

struct Step
    name::String
    path::String
    project::String
    enabled::Bool
end

function read_config(args)
    # parse --config PATH
    cfg_path = nothing
    for (i,a) in enumerate(args)
        if a == "--config" && i < length(args)
            cfg_path = args[i+1]
            break
        end
    end
    return cfg_path
end

function yaml_read_bool(line)
    s = strip(split(line, ":", limit=2)[2])
    s = replace(s, '"' => '')
    return lowercase(s) in ("1","true","yes","on")
end

function load_yaml_config(path::AbstractString)
    cfg = Dict{String,Any}()
    cfg["runs"] = Dict{String,Bool}(
        "gc_single" => true,
        "gc_suite" => true,
        "gcn_single" => true,
        "gcn_suite" => true,
        "hgf" => true,
    )
    if !isfile(path)
        return cfg
    end
    for line in eachline(path)
        l = strip(line)
        isempty(l) && continue
        startswith(l, "#") && continue
        if startswith(l, "runs:")
            # subsequent lines read in second pass
            continue
        elseif occursin("runs.gc_single:", l)
            cfg["runs"]["gc_single"] = yaml_read_bool(l)
        elseif occursin("runs.gc_suite:", l)
            cfg["runs"]["gc_suite"] = yaml_read_bool(l)
        elseif occursin("runs.gcn_single:", l)
            cfg["runs"]["gcn_single"] = yaml_read_bool(l)
        elseif occursin("runs.gcn_suite:", l)
            cfg["runs"]["gcn_suite"] = yaml_read_bool(l)
        elseif occursin("runs.hgf:", l)
            cfg["runs"]["hgf"] = yaml_read_bool(l)
        end
    end
    return cfg
end

function banner(msg)
    println("\n$EMJ_ROCKET  $(msg)  $(repeat("=", max(0, 64 - length(msg))))\n")
end

function run_step(step::Step)
    step.enabled || (println("$EMJ_INFO  Skipping disabled step: $(step.name)"); return true)
    banner("Running $(step.name)")
    println("$EMJ_INFO  Project: $(step.project)")
    println("$EMJ_INFO  Script : $(step.path)")
    t0 = time()
    try
        cmd = `julia --project=$(step.project) $(step.path)`
        run(cmd)
        println("$EMJ_OK  Completed $(step.name) in $(round(time() - t0, digits=2))s")
        return true
    catch e
        println("$EMJ_ERR  Failed $(step.name): $(e)")
        return false
    end
end

function main()
    cfg_path = read_config(ARGS)
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    cfg = load_yaml_config(coalesce(cfg_path, joinpath(repo_root, "research", "hgf", "run_config.yaml")))

    steps = Step[
        Step("Generalized Coordinates (single)", joinpath(repo_root, "research", "generalized_coordinates", "run_gc_car.jl"), joinpath(repo_root, "research", "generalized_coordinates"), cfg["runs"]["gc_single"]),
        Step("Generalized Coordinates (suite)", joinpath(repo_root, "research", "generalized_coordinates", "run_gc_suite.jl"), joinpath(repo_root, "research", "generalized_coordinates"), cfg["runs"]["gc_suite"]),
        Step("Generalized Coordinates n-order (single)", joinpath(repo_root, "research", "generalized_coordinates_n_order", "run_gc_car.jl"), joinpath(repo_root, "research", "generalized_coordinates_n_order"), cfg["runs"]["gcn_single"]),
        Step("Generalized Coordinates n-order (suite)", joinpath(repo_root, "research", "generalized_coordinates_n_order", "run_gc_suite.jl"), joinpath(repo_root, "research", "generalized_coordinates_n_order"), cfg["runs"]["gcn_suite"]),
        Step("HGF end-to-end", joinpath(repo_root, "research", "hgf", "run_hgf.jl"), joinpath(repo_root, "research", "hgf"), cfg["runs"]["hgf"]),
    ]

    println("$EMJ_TOOLS  Ensuring research project envs are instantiated…")
    for proj in unique(s.project for s in steps)
        if isfile(joinpath(proj, "Project.toml"))
            try
                run(`julia --project=$(proj) -e 'using Pkg; Pkg.instantiate()'`)
            catch e
                println("$EMJ_WARN  Instantiation warning for $(proj): $(e)")
            end
        end
    end

    failures = String[]
    for s in steps
        isfile(s.path) || (println("$EMJ_WARN  Skipping missing script: $(s.path)"); continue)
        ok = run_step(s)
        ok || push!(failures, s.name)
    end

    if isempty(failures)
        println("\n$EMJ_OK  All research runs completed successfully")
        return 0
    else
        println("\n$EMJ_ERR  Some runs failed: ", join(failures, ", "))
        return 1
    end
end

exit(main())

#!/usr/bin/env julia

using Logging
using Printf
using Dates
using Pkg

const EMJ_INFO = "ℹ️"; const EMJ_OK = "✅"; const EMJ_WARN = "⚠️"; const EMJ_ERR = "❌";
const EMJ_ROCKET = "🚀"; const EMJ_GEAR = "⚙️"; const EMJ_BOOK = "📚"; const EMJ_NOTE = "📝";
const EMJ_LAPTOP = "💻"; const EMJ_CHART = "📈"; const EMJ_SPARKS = "✨"; const EMJ_TOOLS = "🛠️";

struct Step
    name::String
    path::String
    project::String
end

function banner(msg)
    println("\n$EMJ_ROCKET  $(msg)  $(repeat("=", max(0, 64 - length(msg))))\n")
end

function run_step(step::Step)
    banner("Running $(step.name)")
    println("$EMJ_INFO  Project: $(step.project)")
    println("$EMJ_INFO  Script : $(step.path)")
    t0 = time()
    try
        cmd = `julia --project=$(step.project) $(step.path)`
        run(cmd)
        println("$EMJ_OK  Completed $(step.name) in $(round(time() - t0, digits=2))s")
        return true
    catch e
        println("$EMJ_ERR  Failed $(step.name): $(e)")
        return false
    end
end

function main()
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    steps = Step[
        Step("Generalized Coordinates (single)", joinpath(repo_root, "research", "generalized_coordinates", "run_gc_car.jl"), joinpath(repo_root, "research", "generalized_coordinates")),
        Step("Generalized Coordinates (suite)", joinpath(repo_root, "research", "generalized_coordinates", "run_gc_suite.jl"), joinpath(repo_root, "research", "generalized_coordinates")),
        Step("Generalized Coordinates n-order (single)", joinpath(repo_root, "research", "generalized_coordinates_n_order", "run_gc_car.jl"), joinpath(repo_root, "research", "generalized_coordinates_n_order")),
        Step("Generalized Coordinates n-order (suite)", joinpath(repo_root, "research", "generalized_coordinates_n_order", "run_gc_suite.jl"), joinpath(repo_root, "research", "generalized_coordinates_n_order")),
        Step("HGF end-to-end", joinpath(repo_root, "research", "hgf", "run_hgf.jl"), joinpath(repo_root, "research", "hgf")),
    ]

    println("$EMJ_TOOLS  Ensuring research project envs are instantiated…")
    for proj in unique(s.project for s in steps)
        if isfile(joinpath(proj, "Project.toml"))
            try
                run(`julia --project=$(proj) -e 'using Pkg; Pkg.instantiate()'`)
            catch e
                println("$EMJ_WARN  Instantiation warning for $(proj): $(e)")
            end
        end
    end

    failures = String[]
    for s in steps
        isfile(s.path) || (println("$EMJ_WARN  Skipping missing script: $(s.path)"); continue)
        ok = run_step(s)
        ok || push!(failures, s.name)
    end

    if isempty(failures)
        println("\n$EMJ_OK  All research runs completed successfully")
        return 0
    else
        println("\n$EMJ_ERR  Some runs failed: ", join(failures, ", "))
        return 1
    end
end

exit(main())


