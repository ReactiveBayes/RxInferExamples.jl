#!/usr/bin/env julia

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using VisualizationMethods
using Dates

models = scan_models()
results = visualize_repo_models()

println("[", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"), "] Rendered ", length(results), " models (static).")

# Save original @model snippet into each output dir for comparison
for m in models
    src = read(m.file, String)
    pat = Regex("(?s)@model\\s+function\\s+" * m.name * "\\s*\\([^)]*\\)\\s*.*?\\send")
    mm = match(pat, src)
    subdir = joinpath(dirname(@__DIR__), "outputs", string(splitext(basename(m.file))[1], "__", m.name))
    isdir(subdir) || mkpath(subdir)
    if mm !== nothing
        open(joinpath(subdir, string(m.name, "_source.jl")), "w") do io
            write(io, mm.match)
        end
    end
end
