module GCReport

using Dates
using Printf

export write_markdown_report

function _files_table(outdir::AbstractString)
    entries = filter(x -> x != "." && x != "..", readdir(outdir))
    rows = String[]
    for f in sort(entries)
        path = joinpath(outdir, f)
        sz = isfile(path) ? filesize(path) : 0
        push!(rows, "| `$(f)` | $(isdir(path) ? "dir" : "file") | $(sz) |")
    end
    return join(rows, '\n')
end

"""
write_markdown_report(outdir, scen; extra=Dict())
Write a Markdown report summarizing scenario configuration, metrics, and outputs.
`extra` may include keys like :metrics (Dict), :fe_summary, etc.
"""
function write_markdown_report(outdir::AbstractString, scen; extra=Dict{Symbol,Any}())
    mkpath(outdir)
    md = IOBuffer()
    println(md, "# GC Scenario Report: $(scen.name)")
    println(md, "Generated: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))\n")
    println(md, "## Scenario")
    println(md, "- **n**: $(scen.n)")
    println(md, "- **dt**: $(scen.dt)")
    println(md, "- **σ_a**: $(scen.σ_a)")
    println(md, "- **σ_obs_pos**: $(scen.σ_obs_pos)")
    println(md, "- **σ_obs_vel**: $(scen.σ_obs_vel)")
    println(md, "- **generator**: `$(scen.generator)` with kwargs `$(scen.generator_kwargs)`\n")

    if haskey(extra, :metrics)
        met = extra[:metrics]
        println(md, "## Metrics\n")
        println(md, "| metric | pos | vel | acc |\n|---|---:|---:|---:|")
        @printf(md, "| rmse | %.6f | %.6f | %.6f |\n", met[:rmse]...)
        @printf(md, "| coverage95 | %.3f | %.3f | %.3f |\n\n", met[:coverage]...)
    end

    if haskey(extra, :fe_iters)
        fe = extra[:fe_iters]
        println(md, "## Free energy over iterations")
        println(md, "- iterations: $(length(fe))")
        println(md, @sprintf("- start: %.6f, end: %.6f, delta: %.6f\n", fe[1], fe[end], fe[end]-fe[1]))
    end

    println(md, "## Outputs in `$(outdir)`\n")
    println(md, "| name | type | bytes |\n|---|---|---:|")
    println(md, _files_table(outdir))

    open(joinpath(outdir, "REPORT.md"), "w") do io
        write(io, String(take!(md)))
    end
end

end # module


