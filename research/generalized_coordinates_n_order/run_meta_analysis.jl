ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

using Pkg
Pkg.activate(@__DIR__)

using Plots
using Statistics
using Printf

const OUTROOT = joinpath(@__DIR__, "outputs")
const META_OUT = joinpath(OUTROOT, "meta_analysis")
isdir(META_OUT) || mkpath(META_OUT)

# -------------------------------
# Lightweight CSV readers (no deps)
# -------------------------------
function _read_csv_lines(path::AbstractString)
    open(path, "r") do io
        return readlines(io)
    end
end

function _parse_csv(path::AbstractString)
    lines = _read_csv_lines(path)
    isempty(lines) && return String[], Vector{Vector{String}}()
    header = split(strip(first(lines)), ",")
    rows = [split(strip(l), ",") for l in Iterators.drop(lines, 1) if !isempty(strip(l))]
    return header, rows
end

_to_strings(vs) = String[string(v) for v in vs]
function _parse_floats(vec)
    return [
        begin
            s = string(v)
            x = tryparse(Float64, s)
            x === nothing ? NaN : x
        end for v in vec
    ]
end

# -------------------------------
# Discover orders and scenarios
# -------------------------------
order_dirs = filter(name -> startswith(name, "order_"), readdir(OUTROOT; join=false))
orders = sort(parse.(Int, replace.(order_dirs, "order_" => "")))
order_to_dir = Dict(k => joinpath(OUTROOT, "order_" * string(k)) for k in orders)

scenarios = String[]
for K in orders
    kdir = order_to_dir[K]
    for scen in readdir(kdir; join=false)
        isdir(joinpath(kdir, scen)) || continue
        push!(scenarios, scen)
    end
end
scenarios = sort(unique(scenarios))

println(@sprintf("Discovered %d orders: %s", length(orders), join(string.(orders), ", ")))
println(@sprintf("Discovered %d scenarios: %s", length(scenarios), join(scenarios, ", ")))

# -------------------------------
# Allocate metric matrices
# Rows: orders, Cols: scenarios
# -------------------------------
num_orders = length(orders)
num_scen = length(scenarios)

rmse_dim1 = fill(NaN, num_orders, num_scen)
coverage_dim1 = fill(NaN, num_orders, num_scen)
rmse_mean = fill(NaN, num_orders, num_scen)
coverage_mean = fill(NaN, num_orders, num_scen)
fe_iter_final = fill(NaN, num_orders, num_scen)
fe_time_total_sum = fill(NaN, num_orders, num_scen)
fe_time_final = fill(NaN, num_orders, num_scen)

function _write_matrix_csv(path::AbstractString, rows::Vector{Int}, cols::Vector{String}, M::Array{Float64,2})
    open(path, "w") do io
        println(io, join(vcat(["order\\scenario"], cols), ","))
        for (i, ord) in enumerate(rows)
            row = vcat([string(ord)], [string(M[i, j]) for j in 1:length(cols)])
            println(io, join(row, ","))
        end
    end
end

# -------------------------------
# Collect metrics from each run folder
# -------------------------------
scenario_to_col = Dict(s => j for (j, s) in enumerate(scenarios))
order_to_row = Dict(k => i for (i, k) in enumerate(orders))

for K in orders
    kdir = order_to_dir[K]
    for scen in scenarios
        sdir = joinpath(kdir, scen)
        isdir(sdir) || continue

        i = order_to_row[K]
        j = scenario_to_col[scen]

        # metrics.csv: two lines with rmse and coverage95
        metrics_path = joinpath(sdir, "metrics.csv")
        if isfile(metrics_path)
            header, rows = _parse_csv(metrics_path)
            # rows: [ ["rmse", dim_1, dim_2, ...], ["coverage95", dim_1, ...] ]
            if length(rows) >= 1
                rmse_vals = _parse_floats(rows[1][2:end])
                rmse_dim1[i, j] = isempty(rmse_vals) ? NaN : rmse_vals[1]
                rmse_mean[i, j] = isempty(rmse_vals) ? NaN : mean(skipmissing(rmse_vals))
            end
            if length(rows) >= 2
                cov_vals = _parse_floats(rows[2][2:end])
                coverage_dim1[i, j] = isempty(cov_vals) ? NaN : cov_vals[1]
                coverage_mean[i, j] = isempty(cov_vals) ? NaN : mean(skipmissing(cov_vals))
            end
        end

        # rxinfer_free_energy.csv: take last iteration value
        rxfe_path = joinpath(sdir, "rxinfer_free_energy.csv")
        if isfile(rxfe_path)
            _, rows = _parse_csv(rxfe_path)
            if !isempty(rows)
                last_row = rows[end]
                fe_iter_final[i, j] = tryparse(Float64, last_row[end]) === nothing ? NaN : tryparse(Float64, last_row[end])
            end
        end

        # gc_free_energy_timeseries.csv: sum of total, and final total
        fet_path = joinpath(sdir, "gc_free_energy_timeseries.csv")
        if isfile(fet_path)
            header, rows = _parse_csv(fet_path)
            # header: t,obs_term,prior_term,dyn_term,total
            total_idx = findfirst(==("total"), header)
            if total_idx !== nothing
                totals = Float64[]
                for r in rows
                    push!(totals, tryparse(Float64, r[total_idx]) === nothing ? NaN : tryparse(Float64, r[total_idx]))
                end
                fe_time_total_sum[i, j] = sum(skipmissing(totals))
                fe_time_final[i, j] = isempty(totals) ? NaN : last(totals)
            end
        end
    end
end

# -------------------------------
# Save tables
# -------------------------------
_write_matrix_csv(joinpath(META_OUT, "rmse_dim1.csv"), orders, scenarios, rmse_dim1)
_write_matrix_csv(joinpath(META_OUT, "coverage95_dim1.csv"), orders, scenarios, coverage_dim1)
_write_matrix_csv(joinpath(META_OUT, "rmse_mean.csv"), orders, scenarios, rmse_mean)
_write_matrix_csv(joinpath(META_OUT, "coverage95_mean.csv"), orders, scenarios, coverage_mean)
_write_matrix_csv(joinpath(META_OUT, "fe_iter_final.csv"), orders, scenarios, fe_iter_final)
_write_matrix_csv(joinpath(META_OUT, "fe_time_total_sum.csv"), orders, scenarios, fe_time_total_sum)
_write_matrix_csv(joinpath(META_OUT, "fe_time_final.csv"), orders, scenarios, fe_time_final)

# -------------------------------
# Heatmaps
# -------------------------------
default(; dpi=150)

function _heatmap_and_save(M::Array{Float64,2}, title::String, filename::String; clabel::String="value")
    p = heatmap(
        scenarios, string.(orders), M;
        xlabel = "scenario",
        ylabel = "order",
        colorbar_title = clabel,
        title = title,
        yflip = true,
    )
    savefig(p, joinpath(META_OUT, filename))
end

_heatmap_and_save(rmse_dim1, "RMSE (dim 1)", "heatmap_rmse_dim1.png"; clabel="rmse")
_heatmap_and_save(coverage_dim1, "Coverage 95% (dim 1)", "heatmap_coverage_dim1.png"; clabel="coverage")
_heatmap_and_save(rmse_mean, "Mean RMSE (across dims)", "heatmap_rmse_mean.png"; clabel="rmse")
_heatmap_and_save(coverage_mean, "Mean Coverage 95% (across dims)", "heatmap_coverage_mean.png"; clabel="coverage")
_heatmap_and_save(fe_iter_final, "Final Iteration Free Energy", "heatmap_fe_iter_final.png"; clabel="ELBO")
_heatmap_and_save(fe_time_total_sum, "Sum of Time-step Free Energy (total)", "heatmap_fe_time_total_sum.png"; clabel="sum FE")
_heatmap_and_save(fe_time_final, "Final Time-step Free Energy (total)", "heatmap_fe_time_final.png"; clabel="FE")

# -------------------------------
# Markdown summary index
# -------------------------------
open(joinpath(META_OUT, "README.md"), "w") do io
    println(io, "# Meta-analysis Summary")
    println(io, "\nThis folder aggregates metrics across orders and scenarios.")
    println(io, "\nTables:")
    for f in [
        "rmse_dim1.csv",
        "coverage95_dim1.csv",
        "rmse_mean.csv",
        "coverage95_mean.csv",
        "fe_iter_final.csv",
        "fe_time_total_sum.csv",
        "fe_time_final.csv",
    ]
        println(io, "- " * f)
    end
    println(io, "\nHeatmaps:")
    for f in [
        "heatmap_rmse_dim1.png",
        "heatmap_coverage_dim1.png",
        "heatmap_rmse_mean.png",
        "heatmap_coverage_mean.png",
        "heatmap_fe_iter_final.png",
        "heatmap_fe_time_total_sum.png",
        "heatmap_fe_time_final.png",
    ]
        println(io, "- " * f)
    end
    println(io, "\nNotes:")
    println(io, "- Dim 1 corresponds to position. Mean metrics are averaged across available state dimensions.")
    println(io, "- Correlation metrics require saved per-time posterior means and true states; if desired, extend the suite to dump those inputs and add correlation here.")
end

println("Meta-analysis complete → " * META_OUT)


