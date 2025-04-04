# Room Temperature and Humidity Parameter Optimization

This example demonstrates parameter optimization for a coupled temperature-humidity dynamical system using RxInfer.jl. The model simulates how temperature and humidity in a room interact and evolve over time.

## Model Description

The dynamical system is defined by the following equations:

```
dT/dt = -β*T + α*H
dH/dt = -γ*H - α*T
```

Where:
- T is temperature (°C)
- H is humidity (%)
- α is the coupling strength between temperature and humidity
- β is the temperature decay rate
- γ is the humidity decay rate

These equations model:
- Temperature decreases at rate β, but increases with humidity (α*H)
- Humidity decreases at rate γ, and decreases with temperature (-α*T)

## Files

- `room_temperature_humidity.jl`: Main script implementing data generation, model definition, parameter optimization, and inference
- `visualization_utils.jl`: Utility functions for visualization, animation, and logging
- `README.md`: This file

## Running the Example

To run the example:

```bash
cd room
julia room_temperature_humidity.jl
```

The script will:
1. Generate synthetic temperature and humidity data with known parameters
2. Define a probabilistic model using RxInfer.jl
3. Optimize the parameters using Optim.jl
4. Run inference with the optimized parameters
5. Create visualizations and animations of the results

## Outputs

All outputs are saved to timestamped directories under `room/outputs/`. For each run, you'll find:

- Logs of the optimization process
- Plots of the original data and inferred states
- Optimization trace visualization
- Parameter convergence animations
- Free energy landscape visualizations
- State evolution animations

## Parameters

The model optimizes for five parameters:
- α: Coupling strength between temperature and humidity
- β: Temperature decay rate
- γ: Humidity decay rate
- T₀: Initial temperature
- H₀: Initial humidity

## Dependencies

This example requires the following Julia packages:
- RxInfer
- Optim
- Plots
- StatsPlots
- LaTeXStrings
- StableRNGs
- LinearAlgebra
- Statistics
- Dates
- ProgressMeter
- Measures

The script will automatically check for and install any missing packages. 