module VisualizationMethods

export scan_models, parse_model_signatures, model_to_mermaid, model_to_dot, render_graph_assets, visualize_repo_models

using Dates
using Printf

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const OUTPUT_ROOT = abspath(joinpath(@__DIR__, "..", "outputs"))

"""
scan_models(; root::AbstractString=ROOT) -> Vector{NamedTuple}

Scan the repository for GraphPPL / RxInfer `@model` definitions in `.jl` files.
Returns a vector of named tuples: (file, name, line, signature, call)
"""
function scan_models(; root::AbstractString=ROOT)
    jl_files = String[]
    for (dir, _, files) in walkdir(root)
        occursin("/results/", dir) && continue
        occursin("/outputs/", dir) && continue
        occursin("/.git/", dir) && continue
        occursin("/.ipynb_checkpoints/", dir) && continue
        for f in files
            endswith(f, ".jl") || continue
            push!(jl_files, joinpath(dir, f))
        end
    end

    models = NamedTuple{(:file, :name, :line, :signature, :call), Tuple{String,String,Int,String,String}}[]
    m_pat = r"@model\s+function\s+([\p{L}_][\p{L}\p{N}_]*)\s*\(([^)]*)\)"s
    for path in jl_files
        src = try
            read(path, String)
        catch
            nothing
        end
        src === nothing && continue
        for m in eachmatch(m_pat, src)
            name = String(m.captures[1])
            args = String(m.captures[2])
            first_idx = m.offset
            line = 1 + count(==('\n'), src[1:first_idx])
            signature = "$(name)($(args))"
            push!(models, (file=path, name=name, line=line, signature=signature, call="$(name)()"))
        end
    end
    sort!(models, by = x -> (x.file, x.line))
    return models
end

"""
parse_model_signatures(models) -> Dict{String,Vector{NamedTuple}}
Group found models by source file.
"""
function parse_model_signatures(models)
    by_file = Dict{String, Vector{NamedTuple}}()
    for m in models
        push!(get!(by_file, m.file, NamedTuple[]), m)
    end
    return by_file
end

# Block-aware extraction of a @model function body
function extract_model_body(src::AbstractString, name::AbstractString)
    sig_pat = Regex("@model\\s+function\\s+" * name * "\\s*\\(([^)]*)\\)")
    ms = match(sig_pat, src)
    ms === nothing && return nothing
    start_pos = ms.offset + lastindex(ms.match)
    # Move to next line after signature
    after = src[start_pos:end]
    # Scan line by line counting block depth
    depth = 1
    body_buf = IOBuffer()
    for ln in split(after, '\n')
        s = strip(ln)
        # naive skip of block comments/strings not handled; best-effort
        # increment on block starters and decrement on 'end'
        if occursin(r"\b(function|begin|if|for|while|try|let|struct|mutable\s+struct|macro)\b", s)
            depth += 1
        end
        if s == "end" || occursin(r"\bend\b", s)
            depth -= 1
            if depth == 0
                # reached the function end; stop without writing this 'end'
                break
            end
        end
        write(body_buf, ln, '\n')
    end
    return String(take!(body_buf))
end

"""
model_to_mermaid(file::AbstractString, name::AbstractString) -> String
Produce a Mermaid graph description for the given model by emitting a simple node/edge list
based on assignments within the model body. This is heuristic but works for many examples.
"""
function model_to_mermaid(file::AbstractString, name::AbstractString)
    src = read(file, String)
    body = extract_model_body(src, name)
    body === nothing && return "graph TD\n  A[\"$(name)\"]\n"

    # Patterns: stochastic (~, .~ with optional where { ... }); deterministic (:= or =)
    ident = "[\\p{L}_][\\p{L}\\p{N}_]*"
    stoch_pat = Regex("^\\s*($ident(?:\\[[^\\]]*\\])?)\\s*(?:\\.\\s*)?~\\s*($ident)\\s*\\((.*?)\\)\\s*(?:where\\s*\\{[^\\}]*\\})?", "m")
    det_pat   = Regex("^\\s*($ident(?:\\[[^\\]]*\\])?)\\s*:?=\\s*($ident)\\s*\\((.*?)\\)", "m")

    stoch = collect(eachmatch(stoch_pat, body))
    dets  = collect(eachmatch(det_pat, body))

    nodes = Set{String}()
    edges = Set{Tuple{String,String}}()

    is_ident(t::AbstractString) = occursin(r"^[\p{L}_][\p{L}\p{N}_]*$", String(t))

    process_edge(var, fn, args) = begin
        var = String(var); fn = String(fn); args = String(args)
        # normalize y[i] -> y
        var = replace(var, r"\[.*\]" => "")
        push!(nodes, var)
        push!(nodes, fn)
        push!(edges, (fn, var))
        cleaned = replace(replace(args, '[' => ' '), ']' => ' ')
        for token in split(cleaned, [',',' ',';','\n','\t'])
            t = strip(token)
            isempty(t) && continue
            occursin(r"^[-+*/0-9.]+$", t) && continue
            is_ident(t) || continue
            t == fn && continue
            push!(nodes, t)
            push!(edges, (t, var))
        end
    end

    for s in stoch
        var = strip(s.captures[1])
        dist = strip(s.captures[2])
        args = strip(s.captures[3])
        process_edge(var, dist, args)
    end

    for d in dets
        var = strip(d.captures[1])
        fn  = strip(d.captures[2])
        args = strip(d.captures[3])
        process_edge(var, fn, args)
    end

    # Always run a simple line-based fallback to catch missed forms
    for raw in split(body, '\n')
        line = first(split(raw, '#'))  # strip comments
        l = strip(line)
        isempty(l) && continue
        if occursin("~", l)
            # y .~ Dist(args) or x ~ Dist(args)
            lhs_rhs = split(l, "~"; limit=2)
            length(lhs_rhs) == 2 || continue
            lhs = strip(lhs_rhs[1])
            rhs = strip(lhs_rhs[2])
            lhs = replace(replace(lhs, "."=>""), r"\[.*\]"=>"")
            # function name and args
            mr = match(r"^\s*([^\(]+)\s*\((.*)\)\s*$", rhs)
            mr === nothing && continue
            fname = strip(replace(mr.captures[1], r"[^\p{L}\p{N}_]+"=>""))
            fargs = mr.captures[2]
            isempty(fname) && continue
            process_edge(lhs, fname, fargs)
        elseif occursin(":=", l) || occursin("=", l)
            op = occursin(":=", l) ? ":=" : "="
            lhs_rhs = split(l, op; limit=2)
            length(lhs_rhs) == 2 || continue
            lhs = strip(lhs_rhs[1])
            rhs = strip(lhs_rhs[2])
            lhs = replace(lhs, r"\[.*\]"=>"")
            mr = match(r"^\s*([^\(]+)\s*\((.*)\)\s*$", rhs)
            mr === nothing && continue
            fname = strip(replace(mr.captures[1], r"[^\p{L}\p{N}_]+"=>""))
            fargs = mr.captures[2]
            isempty(fname) && continue
            process_edge(lhs, fname, fargs)
        end
    end

    bufs = IOBuffer()
    println(bufs, "graph TD")
    for n in sort!(collect(nodes))
        println(bufs, @sprintf("  %s[\"%s\"]", replace(n, '.'=> '_', '['=>'_', ']' => '_'), n))
    end
    for (a,b) in sort!(collect(edges))
        a2 = replace(a, '.'=> '_', '['=>'_', ']' => '_')
        b2 = replace(b, '.'=> '_', '['=>'_', ']' => '_')
        println(bufs, "  $(a2) --> $(b2)")
    end
    return String(take!(bufs))
end

"""
model_to_dot(file, name) -> String
Generate DOT from the Mermaid text for compatibility with Graphviz renderers.
"""
function model_to_dot(file::AbstractString, name::AbstractString)
    mer = model_to_mermaid(file, name)
    # Simple conversion: treat as directed graph with labeled nodes
    lines = split(mer, '\n')
    buf = IOBuffer()
    println(buf, "digraph G {")
    for ln in lines
        startswith(ln, "  ") || continue
        if occursin("[\"", ln) && occursin("\"]", ln)
            # node line:   id["label"]
            id = strip(first(split(ln, "[")))
            lab = match(r"\[\"(.*)\"\]", ln)
            if lab !== nothing
                println(buf, @sprintf("  \"%s\" [label=\"%s\"];", id, lab.captures[1]))
            end
        elseif occursin("-->", ln)
            parts = split(strip(ln), "-->")
            length(parts) == 2 || continue
            a = strip(parts[1])
            b = strip(parts[2])
            println(buf, @sprintf("  \"%s\" -> \"%s\";", a, b))
        end
    end
    println(buf, "}")
    return String(take!(buf))
end

"""
render_graph_assets(file, name; outdir=OUTPUT_ROOT) -> NamedTuple
Save Mermaid (`.mmd`) and DOT (`.dot`) and, if `dot` is available, PNG (`.png`).
Outputs are written to a subdirectory named `{basename(file)}__{name}` under `outputs/`.
"""
function render_graph_assets(file::AbstractString, name::AbstractString; outdir::AbstractString=OUTPUT_ROOT)
    isdir(outdir) || mkpath(outdir)
    base = splitext(basename(file))[1]
    subdir = joinpath(outdir, string(base, "__", name))
    isdir(subdir) || mkpath(subdir)

    mer = model_to_mermaid(file, name)
    dot = model_to_dot(file, name)

    mmd_path = joinpath(subdir, "$(name).mmd")
    dot_path = joinpath(subdir, "$(name).dot")
    png_path = joinpath(subdir, "$(name).png")

    open(mmd_path, "w") do io; write(io, mer); end
    open(dot_path, "w") do io; write(io, dot); end

    # Try render with Graphviz if available on PATH
    try
        run(`dot -Tpng $(dot_path) -o $(png_path)`)  # requires Graphviz CLI
    catch err
        # Silently skip PNG if Graphviz is not installed
    end

    return (mmd=mmd_path, dot=dot_path, png=isfile(png_path) ? png_path : nothing, dir=subdir)
end

"""
visualize_repo_models(; root=ROOT, outdir=OUTPUT_ROOT) -> Vector{NamedTuple}
High-level driver: scan, group, and render all found models.
"""
function visualize_repo_models(; root::AbstractString=ROOT, outdir::AbstractString=OUTPUT_ROOT)
    models = scan_models(root=root)
    results = NamedTuple[]
    for m in models
        assets = render_graph_assets(m.file, m.name; outdir=outdir)
        push!(results, merge(m, assets))
    end
    return results
end

end # module
