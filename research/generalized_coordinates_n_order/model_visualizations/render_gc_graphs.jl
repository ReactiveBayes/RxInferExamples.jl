#!/usr/bin/env julia

import Pkg

# Activate the visualization methods project to reuse its utilities
visualization_project = abspath(joinpath(@__DIR__, "..", "..", "visualization_methods"))
Pkg.activate(visualization_project)

using VisualizationMethods
using Dates

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

"""
render_gc_graphs(; orders=1:8, target_dir=abspath(@__DIR__))

Uses VisualizationMethods to extract and render the static graphical model structure for
`gc_car_model` and saves eight PNGs labeled by order and an overall collage in `target_dir`.
"""
function render_gc_graphs(; orders = 1:8, target_dir::AbstractString = abspath(joinpath(@__DIR__, "images")))
    ensure_dir(target_dir)

    # Source of the @model
    gc_model_file = abspath(joinpath(@__DIR__, "..", "src", "GCModel.jl"))
    model_name = "gc_car_model"

    # First, render once via VisualizationMethods to get DOT/PNG
    assets = VisualizationMethods.render_graph_assets(gc_model_file, model_name;
                                                      outdir = target_dir)

    # If Graphviz is available, assets.png will be non-nothing; otherwise, only .dot/.mmd
    source_png = assets.png
    dot_path   = assets.dot

    # Ensure we have a PNG to copy; if not, try to generate it now from DOT
    if source_png === nothing
        try
            png_tmp = joinpath(target_dir, "$(model_name).png")
            run(`dot -Tpng $(dot_path) -o $(png_tmp)`)  # may throw if dot is unavailable
            source_png = png_tmp
        catch
            @warn "Graphviz 'dot' not found; cannot render PNGs. Only DOT/Mermaid will be available."
        end
    end

    generated_pngs = String[]
    for k in orders
        out_png = joinpath(target_dir, "$(model_name)_order_$(k).png")
        if source_png !== nothing
            cp(source_png, out_png; force=true)
            push!(generated_pngs, out_png)
        end
    end

    collage_path = joinpath(target_dir, "$(model_name)_orders_$(first(orders))_$(last(orders))_collage.png")

    # Attempt to create a collage if we have the individual PNGs
    if !isempty(generated_pngs)
        try
            # Prefer ImageMagick's montage if available
            run(`montage $(generated_pngs...) -tile 4x2 -geometry +10+10 $(collage_path)`)  # IM6
        catch
            try
                run(`magick montage $(generated_pngs...) -tile 4x2 -geometry +10+10 $(collage_path)`) # IM7
            catch
                @warn "ImageMagick 'montage' not found; skipping collage generation."
                collage_path = nothing
            end
        end
    else
        collage_path = nothing
    end

    timestamp = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")
    println("[", timestamp, "] Rendered gc_car_model visualizations for orders $(first(orders))..$(last(orders)) in \"$target_dir\"")
    return (; pngs = generated_pngs, collage = collage_path, dot = dot_path, mmd = assets.mmd)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    render_gc_graphs()
end


