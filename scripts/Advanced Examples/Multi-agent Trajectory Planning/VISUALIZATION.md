# Multi-agent Trajectory Planning Visualizations

This document describes the comprehensive visualization capabilities for the Multi-agent Trajectory Planning module, which allow users to generate rich, informative visualizations of agent trajectories, control signals, and position uncertainties.

## Visualization Types

### 1. Path Uncertainty Visualization

Visualizes agent paths with uncertainty ellipses, showing how position uncertainty evolves throughout the trajectory.

```julia
plot_path_uncertainties(agents, paths, uncertainties, 
                       filename = "path_uncertainty.png",
                       uncertainty_scale = 3.0,
                       show_step = 5)
```

**Options:**
- `uncertainty_scale`: Scale factor for the uncertainty ellipses (default: 3.0)
- `show_step`: Which steps to show uncertainties for:
  - `:all`: Show all steps
  - `:endpoints`: Show only start and end
  - Integer: Show every n-th step
  - Vector: Show specific steps

### 2. Control Signal Visualization

Visualizes the control signals over time, showing how agents adjust their movement throughout the trajectory.

```julia
plot_control_signals(controls, 
                    filename = "control_signals.png",
                    component = :both)
```

**Options:**
- `component`: Which control component(s) to show:
  - `:both`: Show both X and Y components in a 2x1 layout
  - `:x`: Show only X component
  - `:y`: Show only Y component
  - `:magnitude`: Show the magnitude of the control signal (√(x² + y²))

### 3. Combined Visualization

Creates a comprehensive visualization showing paths, uncertainties, and control signals at a specific time step.

```julia
plot_combined_visualization(environment, agents, paths, uncertainties, controls,
                          filename = "combined.png",
                          step = :last,
                          uncertainty_scale = 2.0,
                          control_scale = 5.0)
```

**Options:**
- `step`: Which time step to visualize (`:last`, `:first`, or specific step number)
- `uncertainty_scale`: Scale factor for uncertainty ellipses
- `control_scale`: Scale factor for control vectors
- `show_uncertainty_steps`: Number of steps to show uncertainties for

### 4. 3D Path Visualization

Creates a 3D visualization of an agent's path, using either time or uncertainty magnitude as the third dimension.

```julia
plot_agent_path_3d(paths, agent_idx,
                 uncertainties = uncertainties,
                 filename = "agent_path_3d.png",
                 with_time = true)
```

**Options:**
- `agent_idx`: Index of the agent to visualize
- `with_time`: If true, use time as z-axis; if false, use uncertainty magnitude
- `colormap`: Colormap for the path points (default: `:viridis`)

### 5. Path Density Heatmap

Creates a heatmap showing the density of agent paths across the environment.

```julia
create_path_heatmap(environment, paths,
                  filename = "path_heatmap.png",
                  resolution = 100,
                  colormap = :thermal)
```

**Options:**
- `resolution`: Resolution of the heatmap grid
- `colormap`: Colormap for the heatmap

## Animations

### 1. Path Uncertainty Animation

Animates agent paths with uncertainty ellipses over time.

```julia
animate_path_uncertainties(environment, agents, paths, uncertainties,
                         filename = "path_uncertainty.gif",
                         fps = 10,
                         uncertainty_scale = 3.0,
                         show_full_path = false)
```

**Options:**
- `fps`: Frames per second for the animation
- `uncertainty_scale`: Scale factor for uncertainty ellipses
- `show_full_path`: If true, show the complete path at each frame; if false, show only the path up to the current step

### 2. Control Signal Animation

Animates agent positions and control vectors over time.

```julia
animate_control_signals(environment, paths, controls,
                      filename = "control_signals.gif",
                      fps = 10,
                      control_scale = 5.0)
```

**Options:**
- `fps`: Frames per second for the animation
- `control_scale`: Scale factor for control vectors

### 3. Combined Visualization Animation

Comprehensive animation showing paths, uncertainties, and controls over time.

```julia
animate_combined_visualization(environment, agents, paths, uncertainties, controls,
                             filename = "combined_visualization.gif",
                             fps = 10,
                             uncertainty_scale = 2.0,
                             control_scale = 5.0,
                             show_trail = true,
                             trail_length = 10)
```

**Options:**
- `fps`: Frames per second for the animation
- `uncertainty_scale`: Scale factor for uncertainty ellipses
- `control_scale`: Scale factor for control vectors
- `show_trail`: If true, show a trail of recent positions
- `trail_length`: Number of recent positions to show in the trail

## Using the Visualization Tools

### For Existing Results

The `visualize_results.jl` script can be used to generate all visualizations for an existing result directory:

```bash
julia visualize_results.jl results/2025-06-04_13-47-58
```

This will create a `visualizations` subdirectory containing all static and animated visualizations, along with a README.md file explaining each visualization.

### In Custom Scripts

You can also use the visualization functions directly in your own scripts:

```julia
using Plots
include("Environment.jl")
include("Visualizations.jl")

# Load your data
# ...

# Create visualizations
plot_path_uncertainties(agents, paths, uncertainties, filename = "path_uncertainty.png")
plot_control_signals(controls, filename = "control_signals.png")
# ...
```

## Data Format Requirements

The visualization functions expect data in the following formats:

- `paths`: Matrix of agent positions over time, dimensions: (nr_agents, nr_steps), each element is a tuple (x, y)
- `controls`: Matrix of control signals, dimensions: (nr_agents, nr_steps), each element is a tuple (x, y)
- `uncertainties`: Matrix of uncertainty values, dimensions: (nr_agents, nr_steps), each element is a tuple (unc_x, unc_y)
- `agents`: Vector of Agent objects with properties: initial_position, target_position, radius, id
- `environment`: Environment object containing obstacles

## Example of Loading CSV Data

```julia
# Load paths data
paths_data = CSV.read("paths.csv", header=false)
nr_agents = length(unique(paths_data[!, 1]))
nr_steps = div(size(paths_data, 1), nr_agents)

# Create paths matrix
paths = Matrix{Tuple{Float64, Float64}}(undef, nr_agents, nr_steps)
for i in 1:size(paths_data, 1)
    agent_idx = paths_data[i, 1]
    step_idx = paths_data[i, 2]
    x = paths_data[i, 3]
    y = paths_data[i, 4]
    paths[agent_idx, step_idx] = (x, y)
end

# Similarly for controls and uncertainties
# ...
``` 