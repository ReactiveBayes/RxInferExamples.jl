module VisualizationMethodsFFG

export try_render_ffg_for_model, visualize_repo_models_ffg

using RxInfer
using Printf
using GraphViz

const OUTPUT_ROOT = abspath(joinpath(@__DIR__, "..", "outputs"))

"""
try_render_ffg_for_model(file::AbstractString, name::AbstractString; outdir=OUTPUT_ROOT)

Heuristically attempts to build a model instance by calling `name()` with no args.
If successful, creates an FFG via `RxInfer.create_model` and writes a DOT/PNG using GraphViz.
Returns NamedTuple with paths or nothing if failed.
"""
function try_render_ffg_for_model(file::AbstractString, name::AbstractString; outdir::AbstractString=OUTPUT_ROOT)
    isdir(outdir) || mkpath(outdir)
    base = splitext(basename(file))[1]
    subdir = joinpath(outdir, string(base, "__", name))
    isdir(subdir) || mkpath(subdir)

    mod = Module(Symbol("VizFFG_", replace(base, ' '=>'_', '-' =>'_')))
    # Including may fail due to missing optional deps; swallow and skip
    ok = try
        Core.include(mod, file);
        true
    catch
        false
    end
    ok || return (dot_ffg=nothing, png_ffg=nothing)
    if !isdefined(mod, Symbol(name))
        return (dot_ffg=nothing, png_ffg=nothing)
    end

    model_fn = getfield(mod, Symbol(name))
    # Attempt zero-arg generator
    gen = try
        model_fn()
    catch
        return (dot_ffg=nothing, png_ffg=nothing)
    end

    model = try
        RxInfer.getmodel(RxInfer.create_model(gen))
    catch
        return (dot_ffg=nothing, png_ffg=nothing)
    end

    # Export to DOT/PNG using GraphViz.load returning a Graph
    try
        g = GraphViz.load(model, strategy=:simple)
        dot_path = joinpath(subdir, string(name, "_ffg.dot"))
        open(dot_path, "w") do io
            print(io, string(g))
        end
        png_path = joinpath(subdir, string(name, "_ffg.png"))
        try
            run(`dot -Tpng $(dot_path) -o $(png_path)`)  # requires Graphviz CLI
        catch
            png_path = nothing
        end
        return (dot_ffg=dot_path, png_ffg=png_path)
    catch
        return (dot_ffg=nothing, png_ffg=nothing)
    end
end

"""
visualize_repo_models_ffg(models; outdir=OUTPUT_ROOT)

Given the list from VisualizationMethods.scan_models, tries to render FFGs where possible.
"""
function visualize_repo_models_ffg(models; outdir::AbstractString=OUTPUT_ROOT)
    results = NamedTuple[]
    for m in models
        r = try_render_ffg_for_model(m.file, m.name; outdir=outdir)
        push!(results, merge(m, r))
    end
    return results
end

end # module
