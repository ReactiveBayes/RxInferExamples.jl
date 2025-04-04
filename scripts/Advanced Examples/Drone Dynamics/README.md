# Drone Dynamics with RxInfer.jl

This example demonstrates how to use RxInfer.jl to perform Bayesian inference for drone motion planning in both 2D and 3D environments.

## Overview

The code simulates a drone that needs to navigate to specified targets (in 2D) or through waypoints (in 3D). It uses RxInfer.jl to perform probabilistic inference on the optimal control inputs needed to achieve the desired trajectory.

### Features

- 2D drone model with two motors for simple pitch control
- 3D drone model with four motors for complete attitude control
- Probabilistic planning using Bayesian inference
- Automatic force planning to reach targets
- Visualization of drone trajectories
- Rain visualization effects in 3D animation
- Interactive mode with guided configuration
- Detailed progress tracking with emoji indicators and timing information

## Running the Examples

There are three ways to run the examples:

### 1. Using Interactive Mode (Recommended)

The easiest way to run the examples is through the interactive mode:

```bash
# Make it executable if needed
chmod +x run_examples.jl

# Run in interactive mode (default)
./run_examples.jl interactive

# Or simply
./run_examples.jl
```

The interactive mode will guide you through setup options with:
- Selection of which examples to run (2D, 3D, or both)
- Custom configuration of parameters
- Clear explanations of each option

### 2. Using the run_examples.jl script with preset configurations

For quick starts with preset configurations:

```bash
# Run only the 2D example with reduced parameters (fastest)
./run_examples.jl 2d-quick

# Run only the 3D example with reduced parameters
./run_examples.jl 3d-quick

# Run both examples with reduced parameters
./run_examples.jl both-quick

# Run both examples with full quality settings
./run_examples.jl full
```

### 3. Directly using the main script with arguments

For advanced users who want more control, you can invoke the main script directly with custom parameters:

```bash
# Show help
julia "Drone Dynamics.jl" --help

# Run only the 2D example (skip 3D)
julia "Drone Dynamics.jl" --no-3d

# Run only the 3D example (skip 2D)
julia "Drone Dynamics.jl" --no-2d

# Run with custom parameters
julia "Drone Dynamics.jl" --horizon-2d 30 --horizon-3d 15 --dt 0.04 --fps 25
```

## Parameters

- `--horizon-2d`: Time horizon for 2D drone planning (default: 15)
- `--horizon-3d`: Time horizon for 3D drone per waypoint (default: 8)
- `--dt`: Time step for simulation (default: 0.1)
- `--fps`: Frames per second for animations (default: 20)

The time horizon parameters control how many timesteps the simulation calculates. Lower values make the simulation run faster but may produce less optimal paths. The dt parameter controls the physical time step size - larger values mean the simulation takes bigger steps, which is faster but less accurate.

## Performance Tips

For faster execution, you can:

1. Use the `2d-quick` or `3d-quick` presets which run with minimal parameters
2. Further reduce `horizon-2d` and `horizon-3d` values (e.g., 5-10 range)
3. Increase the `dt` value (e.g., 0.15-0.2 range) for even faster but less accurate simulation
4. Skip the 3D example with `--no-3d` as it's more computationally intensive

## Progress Tracking

The simulation includes detailed progress tracking with emoji indicators and timing information:

- 🚀 Overall simulation progress
- 🛩️ 2D path planning calculations
- 🚁 3D path planning calculations
- 🎬 Animation generation
- 📊 Plot generation
- ✅ Completion indicators
- ⏱️ Timing for each major operation

This makes it easy to follow the simulation stages and understand how long each part takes.

## Outputs

The examples generate the following outputs:

- `drone.gif`: Animation of the 2D drone trajectory
- `drone_angle.png`: Plot of the inferred angle of the 2D drone
- `drone_forces.png`: Plot of the inferred forces applied to the 2D drone
- `drone_3d_multi.gif`: Animation of the 3D drone trajectory through waypoints

## Code Structure

The code is organized into modules:

- `DroneCore`: Basic drone types and dynamics for 2D
- `DroneModels`: Bayesian models and inference for 2D
- `DroneViz`: Visualization functions for 2D
- `Drone3DCore`: Extended drone types and dynamics for 3D
- `Drone3DModels`: Bayesian models and inference for 3D
- `Drone3DViz`: Visualization functions for 3D

The code separates the core program flow from the statistical descriptions, making it easier to understand and modify.

## Requirements

- Julia 1.6 or later
- RxInfer.jl
- LinearAlgebra
- Statistics
- Plots
- ArgParse
- Dates 