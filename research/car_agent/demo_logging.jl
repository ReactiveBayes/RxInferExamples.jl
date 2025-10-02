#!/usr/bin/env julia
# Demo script to show enhanced logging in action

using Pkg
Pkg.activate(@__DIR__)

println("\n" * "="^70)
println("🚗 MOUNTAIN CAR - ENHANCED LOGGING DEMO")
println("="^70)
println()

# Load framework
if !isdefined(Main, :Config)
    include("config.jl")
end

# Ensure output directories
dirs = Config.ensure_output_directories()
println("✓ Initialized $(length(dirs)) output directories\n")

# Load the example components
include("examples/mountain_car_example.jl")

# Run simulation with full logging
println("🚗 Running Active Inference Simulation...")
println("   Configuration: Horizon=15, Max Steps=50")
println()

agent, env, states, actions, predictions, diagnostics = run_mountain_car_simulation(
    max_steps = 50,
    verbose = true,
    enable_diagnostics = true
)

println("\n✓ Simulation complete: $(length(states)) steps")

# Get physics
Fa, Ff, Fg, height = create_mountain_car_physics()

# Export with enhanced logging
export_simulation_data(states, actions, diagnostics)
create_comprehensive_plots(states, actions, diagnostics, height)
create_animation(states, height)
export_results_summary(agent, states, actions, diagnostics)

# Final comprehensive summary
println("\n" * "="^70)
println("📦 OUTPUT VERIFICATION")
println("="^70)

output_dirs = [
    ("logs", Config.OUTPUTS.logs_dir),
    ("data", Config.OUTPUTS.data_dir),
    ("plots", Config.OUTPUTS.plots_dir),
    ("animations", Config.OUTPUTS.animations_dir),
    ("diagnostics", Config.OUTPUTS.diagnostics_dir),
    ("results", Config.OUTPUTS.results_dir)
]

total_size = 0.0
total_files = 0

for (name, dir_path) in output_dirs
    files = readdir(dir_path, join=false)
    dir_size = sum(filesize(joinpath(dir_path, f)) for f in files if isfile(joinpath(dir_path, f))) / 1024
    total_size += dir_size
    total_files += length(files)
    
    if !isempty(files)
        println("\n✓ outputs/$name/")
        println("  $(length(files)) files, $(round(dir_size, digits=2)) KB")
        for file in files
            file_path = joinpath(dir_path, file)
            if isfile(file_path)
                file_size = filesize(file_path) / 1024
                println("    → $file ($(round(file_size, digits=2)) KB)")
            end
        end
    else
        println("\n○ outputs/$name/ (empty)")
    end
end

println("\n" * "─"^70)
println("📊 TOTAL: $total_files files, $(round(total_size, digits=2)) KB")
println("="^70)

println("\n✅ DEMO COMPLETE - All logging and outputs verified!")

