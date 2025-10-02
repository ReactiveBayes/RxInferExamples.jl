# Visualization Guide

**Framework:** Generic Agent-Environment Framework  
**Version:** 0.1.1  
**Updated:** October 2, 2025

---

## Overview

The framework provides comprehensive visualization and animation capabilities for analyzing Active Inference agent behavior. All visualizations are automatically generated and saved in run-specific directories.

---

## Automatic Visualization

### Quick Start

Visualizations are generated automatically when running simulations:

```bash
# Using examples (automatic visualization)
julia examples/mountain_car.jl
julia examples/simple_nav.jl

# Using config-driven runner (automatic visualization)
julia run.jl simulate
```

All outputs (plots, animations, data) are saved in timestamped directories:
```
outputs/
└── mountaincar_20251002_140530/
    ├── plots/              # Static visualizations
    ├── animations/         # Animated GIFs
    ├── data/              # Raw data (CSV)
    ├── diagnostics/       # Performance metrics (JSON)
    ├── results/           # Summary statistics
    └── REPORT.md          # Comprehensive report
```

---

## Visualization Types

### 1D Trajectory Visualizations

For simple navigation tasks (1D state space):

#### Static Plot: `trajectory_1d.png`
- Position over time
- Actions over time
- Clear axis labels and titles

#### Animated GIF: `trajectory_1d.gif`
- Real-time position evolution
- Trajectory trace showing path taken
- Progress indicator (current step / total steps)

**Generated for:** SimpleNavAgent with SimpleNavEnv

### 2D Trajectory Visualizations

For complex tasks (2D state space like Mountain Car):

#### Static Plot: `trajectory_2d.png`
Four-panel visualization:
1. **Position over time** - Horizontal position evolution
2. **Velocity over time** - Velocity evolution
3. **Phase space** - Position vs. velocity (state space trajectory)
4. **Actions over time** - Control inputs (forces)

#### Landscape Plot: `mountain_car_landscape.png`
- Mountain car valley landscape
- Agent trajectory overlaid on terrain
- Start/end markers
- Goal position indicator

#### Animated GIF: `trajectory_2d.gif`
Four-panel animation showing:
1. Real-time position evolution
2. Real-time velocity evolution
3. Animated phase space trajectory with current position marker
4. Action sequence

**Generated for:** MountainCarAgent with MountainCarEnv

### Diagnostics Visualizations

When diagnostics are enabled: `diagnostics.png`

Panels may include:
- **Memory Usage** - Memory consumption over time
- **Inference Time** - RxInfer computation time per step (log scale)
- **Belief Uncertainty** - Uncertainty evolution (trace of covariance)

---

## Using the Visualization API

### Basic Usage

```julia
using .Visualization

# After running simulation
result = run_simulation(agent, env, config)

# Generate all visualizations
output_dir = "my_outputs"
generate_all_visualizations(result, output_dir, state_dim)
```

### Individual Plots

```julia
# 1D trajectory
plot_trajectory_1d(result, output_dir, title="My 1D Navigation")

# 2D trajectory
plot_trajectory_2d(result, output_dir, title="My 2D State Trajectory")

# Mountain car landscape
plot_mountain_car_landscape(result, output_dir)

# Diagnostics
plot_diagnostics(result.diagnostics, output_dir)
```

### Animations

```julia
# 1D animation
animate_trajectory_1d(result, output_dir, fps=10)

# 2D animation
animate_trajectory_2d(result, output_dir, fps=10)
```

### Comprehensive Output Saving

The framework provides `save_simulation_outputs()` for complete output management:

```julia
# Save everything: data + plots + animations + report
save_simulation_outputs(
    result,
    output_dir,
    goal_state,
    generate_visualizations=true,
    generate_animations=true
)
```

This saves:
- Trajectory data (CSV)
- Observations (CSV)
- Summary statistics (CSV)
- Diagnostics (JSON)
- Metadata (JSON)
- All visualizations (PNG)
- All animations (GIF)
- Comprehensive markdown report (REPORT.md)

---

## Customization

### Custom Plot Titles

```julia
plot_trajectory_1d(result, output_dir, 
                   title="Navigation to Target Position")

plot_trajectory_2d(result, output_dir,
                   title="Mountain Car Active Inference")
```

### Custom Animation Frame Rate

```julia
# Slower animation (5 fps)
animate_trajectory_1d(result, output_dir, fps=5)

# Faster animation (20 fps)
animate_trajectory_2d(result, output_dir, fps=20)
```

### Selective Output Generation

```julia
# Save without animations (faster)
save_simulation_outputs(
    result, output_dir, goal_state,
    generate_visualizations=true,
    generate_animations=false
)

# Save without any visualizations (data only)
save_simulation_outputs(
    result, output_dir, goal_state,
    generate_visualizations=false,
    generate_animations=false
)
```

---

## Output File Structure

Complete run directory structure:

```
outputs/mountaincar_20251002_140530/
├── REPORT.md                          # Comprehensive markdown report
├── metadata.json                      # Run configuration and metadata
├── plots/
│   ├── trajectory_2d.png             # Main trajectory visualization
│   ├── mountain_car_landscape.png    # Landscape view (2D only)
│   └── diagnostics.png               # Performance metrics
├── animations/
│   └── trajectory_2d.gif             # Animated trajectory
├── data/
│   ├── trajectory.csv                # State trajectory data
│   └── observations.csv              # Observation sequence
├── diagnostics/
│   ├── diagnostics.json              # Full diagnostics
│   └── performance.json              # Performance breakdown
└── results/
    └── summary.csv                   # Summary statistics
```

---

## Interpretation Guide

### Phase Space Plots (2D)

The phase space plot shows position vs. velocity:
- **Spiral patterns** - Agent oscillating to build momentum
- **Converging trajectories** - Agent approaching goal
- **Clustering** - Agent getting stuck or converging
- **Wide spread** - Agent exploring state space

### Inference Time Plots

Inference time visualization (log scale):
- **First spike** - Julia JIT compilation
- **Subsequent drops** - Optimized compiled code
- **Steady baseline** - Typical inference time
- **Spikes** - Convergence challenges or complex inference

### Belief Uncertainty

Uncertainty (trace of covariance) over time:
- **Decreasing** - Agent gaining confidence
- **Increasing** - Agent uncertain (poor observations)
- **Stable** - Agent maintaining belief quality
- **Oscillating** - Dynamic environment or model mismatch

### Memory Usage

Memory consumption:
- **Initial spike** - Initialization
- **Growth** - Data accumulation
- **Plateaus** - Stable operation
- **Spikes** - Garbage collection

---

## Performance Considerations

### Visualization Generation Time

- **Static plots**: ~0.1-0.5s each
- **Animations**: ~1-5s depending on length and fps
- **Total overhead**: Usually < 10s for full visualization suite

### File Sizes

Typical output sizes:
- PNG plots: 50-200 KB each
- GIF animations: 500 KB - 2 MB (depends on length and fps)
- CSV data: 10-100 KB per file
- JSON diagnostics: 1-10 KB

### Disabling Visualization for Performance

When running many experiments, disable visualization:

```julia
# In SimulationConfig
config = SimulationConfig(
    max_steps = 100,
    enable_diagnostics = false,  # Disable diagnostics
    enable_logging = false,      # Disable logging
    verbose = false
)

# When saving outputs
save_simulation_outputs(
    result, output_dir, goal_state,
    generate_visualizations=false,
    generate_animations=false
)
```

---

## Advanced Usage

### Custom Visualization Functions

Create your own visualizations using the framework's data:

```julia
using Plots

function my_custom_plot(result, output_dir)
    positions = [s[1] for s in result.states]
    actions = [a[1] for a in result.actions]
    
    # Your custom visualization
    p = plot(positions, actions, 
             xlabel="Position", 
             ylabel="Action",
             title="Custom State-Action Plot")
    
    savefig(p, joinpath(output_dir, "custom_plot.png"))
end
```

### Combining Multiple Runs

Compare multiple runs by loading their CSVs:

```julia
using CSV, DataFrames, Plots

# Load multiple trajectories
run1 = CSV.read("outputs/run1/data/trajectory.csv", DataFrame)
run2 = CSV.read("outputs/run2/data/trajectory.csv", DataFrame)

# Compare
plot(run1.step, run1.position, label="Run 1")
plot!(run2.step, run2.position, label="Run 2")
```

### Publication-Quality Figures

For papers, use higher DPI and specific formats:

```julia
# Modify visualization.jl or create custom functions
using Plots
gr(dpi=300)  # High resolution

# Save as PDF for LaTeX
savefig(my_plot, "figure.pdf")

# Save as SVG for vector graphics
savefig(my_plot, "figure.svg")
```

---

## Troubleshooting

### "Visualization generation failed"

**Cause**: Missing Plots.jl or GR backend issues

**Solution**:
```bash
julia -e 'using Pkg; Pkg.add("Plots"); Pkg.add("GR")'
```

### Animations don't play

**Cause**: GIF viewer compatibility

**Solution**: Use a modern browser or image viewer that supports animated GIFs

### Plots look compressed

**Cause**: Default resolution settings

**Solution**: Increase figure size in visualization.jl:
```julia
size=(1600, 1200)  # Instead of (1200, 900)
```

### Memory issues with long simulations

**Cause**: Large animations with many frames

**Solution**: 
- Reduce fps (5-10 instead of 20-30)
- Disable animations for very long runs
- Generate plots only

---

## Examples

### Minimal Visualization Example

```julia
# Just plots, no animations
output_dir = "quick_test"
mkpath(output_dir)

result = run_simulation(agent, env, config)

# Generate static plots only
plot_trajectory_2d(result, output_dir)
plot_diagnostics(result.diagnostics, output_dir)
```

### Full Suite Example

```julia
# Everything: data, plots, animations, report
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
output_dir = "outputs/full_run_$timestamp"

result = run_simulation(agent, env, config)

save_simulation_outputs(
    result, output_dir, goal_state,
    generate_visualizations=true,
    generate_animations=true
)

println("View report: $(joinpath(output_dir, "REPORT.md"))")
```

---

## Integration with Analysis

### Loading Saved Data

```julia
using CSV, DataFrames, JSON

# Load trajectory
traj = CSV.read("outputs/myrun/data/trajectory.csv", DataFrame)

# Load diagnostics
diag = JSON.parsefile("outputs/myrun/diagnostics/diagnostics.json")

# Load metadata
meta = JSON.parsefile("outputs/myrun/metadata.json")

# Analyze
println("Steps: $(meta["steps_taken"])")
println("Final position: $(traj[end, :position])")
println("Avg inference time: $(diag["performance"]["operations"]["inference"]["avg_time"])")
```

### Batch Analysis

Process multiple runs:

```julia
run_dirs = readdir("outputs", join=true)

for run_dir in run_dirs
    if isdir(run_dir)
        meta_file = joinpath(run_dir, "metadata.json")
        if isfile(meta_file)
            meta = JSON.parsefile(meta_file)
            println("$(basename(run_dir)): $(meta["steps_taken"]) steps")
        end
    end
end
```

---

## See Also

- [README.md](README.md) - Main documentation
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [COMPREHENSIVE_SUMMARY.md](COMPREHENSIVE_SUMMARY.md) - Framework overview
- [docs/index.md](docs/index.md) - API reference

---

**Happy Visualizing! 📊📈🎬**

