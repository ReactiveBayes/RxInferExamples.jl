# Multi-agent Trajectory Planning with Probabilistic Inference

## Overview

This project implements multi-agent trajectory planning using probabilistic inference with the RxInfer.jl framework. The approach allows multiple agents to navigate through environments with obstacles while avoiding collisions between agents.

## Project Organization

The project is organized into several modular components:

1. **Core Module Structure**:
   - `TrajectoryPlanning.jl`: Main module that integrates all components
   - `Environment.jl`: Environment and agent definitions
   - `Models.jl`: Probabilistic model implementation
   - `Visualizations.jl`: Plotting and animation functions
   - `Experiments.jl`: Experiment execution functions

2. **Execution Scripts**:
   - `run_experiment.jl`: Command-line configurable experiment runner
   - `main.jl`: Entry point for running all standard experiments

3. **File Interactions**:
   - Environment definitions feed into the probabilistic model
   - Model outputs are processed by visualization functions
   - Experiment functions coordinate the execution and data collection
   - TrajectoryPlanning module provides a unified interface to all components

See `README.md` for detailed usage instructions and examples.

## Technical Approach

The trajectory planning is formulated as a probabilistic inference problem, where:
- Each agent has a state vector [x, y, vx, vy] representing position and velocity
- The environment contains obstacles defined as rectangles
- The model enforces constraints for obstacle avoidance and agent-agent collision avoidance
- Inference is performed using variational message passing

## Model Description

The probabilistic model is defined using a factor graph approach where random variables are connected through factors representing probabilistic dependencies. The core components include:

### State Space Model

Each agent follows linear dynamics:
- State transition: `state[k, t+1] ~ A * state[k, t] + B * control[k, t]`
- Observation model: `path[k, t] ~ C * state[k, t+1]`

Where:
- `A` is the state transition matrix [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
- `B` is the control input matrix [0 0; dt 0; 0 0; 0 dt]
- `C` is the observation matrix [1 0 0 0; 0 0 1 0]
- `dt` is the time step (set to 1 in this implementation)

### Constraint Formulation

Constraints are encoded as observations through a custom Halfspace node, which creates soft constraints that the inference process tries to satisfy.

Two main constraints are modeled:
1. **Environment constraints**: Agents should not collide with obstacles
   - `z[k, t] ~ g(environment, rs[k], path[k, t])`
   - `z[k, t] ~ Halfspace(0, zσ2[k, t], γ)`
   
2. **Agent-agent collision avoidance**:
   - `d[t] ~ h(environment, rs, path[1, t], path[2, t], path[3, t], path[4, t])`
   - `d[t] ~ Halfspace(0, dσ2[t], γ)`

### Distance Functions

The model uses two key distance functions:
- `g()`: Calculates distance from an agent to obstacles, adjusted for agent radius
- `h()`: Calculates minimum distance between all agent pairs, adjusted for agent radii

The `softmin` function creates a differentiable approximation of the minimum function, allowing the model to handle multiple constraints simultaneously.

## The @model Block Explained

The core of the implementation is the `@model` block which defines the probabilistic model:

```julia
@model function path_planning_model(environment, agents, goals, nr_steps)
    # Model parameters
    local dt = 1
    local A  = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
    local B  = [0 0; dt 0; 0 0; 0 dt]
    local C  = [1 0 0 0; 0 0 1 0]
    local γ  = 1

    # Variables to infer
    local control  # Control inputs for each agent
    local state    # Agent states [x, y, vx, vy]
    local path     # Observed positions [x, y]
    
    # Extract agent radii
    local rs = map((a) -> a.radius, agents)

    # Model for each agent (fixed at 4 agents)
    for k in 1:4
        # Prior on initial state
        state[k, 1] ~ MvNormal(mean = zeros(4), covariance = 1e2I)

        for t in 1:nr_steps
            # Prior on control inputs
            control[k, t] ~ MvNormal(mean = zeros(2), covariance = 1e-1I)

            # State transition model
            state[k, t+1] ~ A * state[k, t] + B * control[k, t]

            # Observation model
            path[k, t] ~ C * state[k, t+1]

            # Environmental constraints
            zσ2[k, t] ~ GammaShapeRate(3 / 2, γ^2 / 2)
            z[k, t]   ~ g(environment, rs[k], path[k, t])
            z[k, t] ~ Halfspace(0, zσ2[k, t], γ)
        end

        # Goals are observed values that constrain initial and final states
        goals[1, k] ~ MvNormal(mean = state[k, 1], covariance = 1e-5I)
        goals[2, k] ~ MvNormal(mean = state[k, nr_steps+1], covariance = 1e-5I)
    end

    # Agent-agent collision avoidance
    for t = 1:nr_steps
        dσ2[t] ~ GammaShapeRate(3 / 2, γ^2 / 2)
        d[t] ~ h(environment, rs, path[1, t], path[2, t], path[3, t], path[4, t])
        d[t] ~ Halfspace(0, dσ2[t], γ)
    end
end
```

### Key Aspects of the Model

1. **State representation**: Each agent's state includes position and velocity in 2D
2. **Control inputs**: Acceleration in x and y directions
3. **Obstacle avoidance**: Achieved through distance functions and Halfspace constraints
4. **Goal specification**: Initial and target positions are specified as constraints
5. **Collision avoidance**: Pairwise distances between agents are constrained to be positive

## Inference Approach

The inference is performed using variational message passing with the following components:

1. **Initialization**: Initial distributions for states and controls
2. **Constraints**: Mean-field variational factorization
3. **Linearization**: Non-linear functions are approximated using linearization
4. **Iterations**: Typically 350 iterations are performed for convergence

## Implementation Notes

- The model currently supports exactly 4 agents
- Obstacles are represented as rectangles
- Distance functions use a differentiable softmin approximation
- The Halfspace node implements the constraint mechanism
- Results are visualized as animated GIFs showing agent trajectories

## Model Trace and Convergence Analysis

During the inference process, the implementation tracks several key aspects:

1. **ELBO Convergence**: The Evidence Lower Bound (ELBO) is tracked at each iteration to monitor convergence
2. **Path Uncertainty**: The variance of agent positions is computed to assess confidence in the paths
3. **Control Signals**: Control inputs required for each agent are visualized and analyzed
4. **Distance Metrics**: Distance to obstacles and between agents is visualized as heatmaps

### Output Files

The implementation generates various output files for analysis:

1. **Animation GIFs**: Showing agent movements through the environment
2. **CSV Data Files**: Raw path, control, and uncertainty data
3. **Visualization PNGs**: Heatmaps and plots of various metrics
4. **Summary Files**: Documentation of experiment parameters and results
5. **Convergence Metrics**: ELBO values throughout the inference process

### Visualization Functions

The implementation includes several advanced visualization functions:

- `animate_paths()`: Creates animations of agent movements
- `visualize_controls()`: Shows control inputs as quiver plots
- `visualize_obstacle_distance()`: Creates heatmaps of obstacle distances
- `visualize_path_uncertainty()`: Displays path uncertainty visually
- `plot_convergence_metrics()`: Shows convergence of the inference process

### Future Improvements

Potential improvements to the model include:
- Supporting variable numbers of agents
- Implementing more sophisticated environment representations
- Adding dynamic obstacles
- Using more advanced approximation methods for non-linear functions
- Implementing online/incremental inference for real-time applications
