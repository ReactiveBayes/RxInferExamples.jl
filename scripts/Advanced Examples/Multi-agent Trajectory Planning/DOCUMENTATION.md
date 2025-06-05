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

The `softmin` function creates a differentiable approximation of the minimum function, allowing the model to handle multiple constraints simultaneously:
```julia
softmin(x; l=10) = -logsumexp(-l .* x) / l
```
Where `l` is a temperature parameter that controls the smoothness of the approximation.

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

The inference constraints are defined with:

```julia
@constraints function path_planning_constraints()
    # Mean-field variational constraints on the parameters
    q(d, dσ2) = q(d)q(dσ2)
    q(z, zσ2) = q(z)q(zσ2)
end
```

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
   - `door_42.gif`, `door_123.gif`: Animations of agents navigating through a door environment with different seeds
   - `wall_42.gif`, `wall_123.gif`: Animations of agents navigating around a wall obstacle
   - `combined_42.gif`, `combined_123.gif`: Animations of agents navigating through a more complex environment
   
2. **CSV Data Files**:
   - `paths.csv`: Raw path data with columns for agent index, time step, x-position, y-position
   - `controls.csv`: Control inputs with columns for agent index, time step, x-control, y-control
   - `uncertainties.csv`: Path uncertainties with columns for agent index, time step, x-variance, y-variance

3. **Visualization PNGs**:
   - `obstacle_distance.png`: Heatmap showing distance to obstacles throughout the environment
   - `path_uncertainty.png`: Visualization of the uncertainty in the planned paths
   - `convergence.png`: Plot showing convergence of the ELBO during inference
   - `door_environment_heatmap.png`, `wall_environment_heatmap.png`, `combined_environment_heatmap.png`: Heatmaps of the obstacle distance fields for different environments

4. **Summary Files**:
   - `experiment_summary.txt`: Summary of experiment parameters and results
   - `experiment.log`: Detailed log of the experiment execution
   - `README.md`: Overview of the results directory contents

### Environment Definition

The project includes three standard environments:

1. **Door Environment**: Two parallel walls with a gap between them
   ```julia
   function create_door_environment()
       return Environment(obstacles = [
           Rectangle(center = (-40, 0), size = (70, 5)),
           Rectangle(center = (40, 0), size = (70, 5))
       ])
   end
   ```

2. **Wall Environment**: A single wall obstacle in the center
   ```julia
   function create_wall_environment()
       return Environment(obstacles = [
           Rectangle(center = (0, 0), size = (10, 5))
       ])
   end
   ```

3. **Combined Environment**: A combination of walls and obstacles
   ```julia
   function create_combined_environment()
       return Environment(obstacles = [
           Rectangle(center = (-50, 0), size = (70, 2)),
           Rectangle(center = (50, -0), size = (70, 2)),
           Rectangle(center = (5, -1), size = (3, 10))
       ])
   end
   ```

### Visualization Functions

The implementation includes several advanced visualization functions that generate multiple output types:

1. **Animated Visualizations**:
   - **`animate_paths(environment, agents, paths; filename, fps)`**:
     - Creates animations of agent movements over time
     - Renders agents as circles with their respective radii
     - Shows the complete path of each agent with dashed lines
     - Outputs: GIF animations (`door_42.gif`, `wall_42.gif`, etc.)

   - **`visualize_controls(agents, controls, paths; filename, fps)`**:
     - Shows control inputs as quiver plots for each agent
     - Visualizes the magnitude and direction of acceleration at each time step
     - Outputs: `control_signals.gif`

2. **Static Visualizations**:
   - **`visualize_obstacle_distance(environment; filename, resolution)`**:
     - Creates heatmaps of distances to obstacles
     - Uses a color gradient to show the distance field around obstacles
     - Outputs: `obstacle_distance.png`, environment-specific heatmaps

   - **`visualize_path_uncertainty(environment, agents, paths, path_vars; filename)`**:
     - Displays path uncertainty visually using circles of varying sizes
     - Larger circles indicate higher uncertainty in the position
     - Outputs: `path_uncertainty.png`

   - **`plot_path_visualization(environment, agents, paths; filename)`**:
     - Creates a static visualization of agent paths
     - Shows start and end positions with markers
     - Outputs: `path_visualization.png`

   - **`plot_control_magnitudes(agents, controls; filename)`**:
     - Plots the magnitude of control signals over time
     - Compares control effort across different agents
     - Outputs: `control_magnitudes.png`

   - **`plot_convergence_metrics(metrics; filename)`**:
     - Shows convergence of the inference process by plotting ELBO values
     - Creates placeholder visualization when convergence data isn't available
     - Outputs: `convergence.png`

3. **Data Files**:
   - **`save_path_data(paths, output_file)`**:
     - Saves agent paths as CSV data
     - Format: agent_id, time_step, x_position, y_position
     - Outputs: `paths.csv`

   - **`save_control_data(controls, output_file)`**:
     - Saves control inputs as CSV data
     - Format: agent_id, time_step, x_control, y_control
     - Outputs: `controls.csv`

   - **`save_uncertainty_data(uncertainties, output_file)`**:
     - Saves path uncertainties as CSV data
     - Format: agent_id, time_step, x_variance, y_variance
     - Outputs: `uncertainties.csv`

### Experiment Execution

The experiment execution is handled by the `Experiments.jl` module, which provides:

1. **`execute_and_save_animation(environment, agents; gifname, kwargs...)`**:
   - Plans paths for agents in the given environment
   - Creates and saves an animation of the resulting paths
   - Returns the planned paths

2. **`run_all_experiments()`**:
   - Runs experiments for all standard environments
   - Uses different random seeds to generate diverse results
   - Saves all outputs to the results directory

## Hard-Coded Parameters

The implementation contains several hard-coded parameters that could be moved to a configuration file:

1. **Model parameters**:
   - `dt = 1`: Time step for the state space model
   - `A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]`: State transition matrix
   - `B = [0 0; dt 0; 0 0; 0 dt]`: Control input matrix
   - `C = [1 0 0 0; 0 0 1 0]`: Observation matrix
   - `γ = 1`: Constraint parameter for the Halfspace node
   - `nr_steps = 40`: Number of time steps in the trajectory
   - `nr_iterations = 350`: Number of inference iterations

2. **Prior distributions**:
   - `MvNormal(mean = zeros(4), covariance = 1e2I)`: Prior on initial state
   - `MvNormal(mean = zeros(2), covariance = 1e-1I)`: Prior on control inputs
   - `MvNormal(mean = state[k, 1], covariance = 1e-5I)`: Goal constraints
   - `GammaShapeRate(3 / 2, γ^2 / 2)`: Prior on constraint parameters

3. **Visualization parameters**:
   - `fps = 15`: Animation frames per second
   - `resolution = 100`: Heatmap resolution
   - Plot sizes, axis limits, and color schemes

4. **Environment parameters**:
   - Fixed number of agents (4)
   - Obstacle positions and sizes
   - Agent radii, initial positions, and target positions
   - Plot boundaries (`xlims = (-20, 20), ylims = (-20, 20)`)

## Configuration File Recommendations

To improve flexibility, the following configurations could be moved to a dedicated file (e.g., TOML, JSON, or YAML):

1. **Model Configuration**:
   ```toml
   [model]
   dt = 1.0
   gamma = 1.0
   nr_steps = 40
   nr_iterations = 350
   nr_agents = 4
   softmin_temperature = 10.0
   
   [priors]
   initial_state_variance = 100.0
   control_variance = 0.1
   goal_constraint_variance = 1e-5
   ```

2. **Environment Configuration**:
   ```toml
   [plotting]
   x_limits = [-20, 20]
   y_limits = [-20, 20]
   fps = 15
   heatmap_resolution = 100
   
   [door_environment]
   obstacles = [
     { center = [-40, 0], size = [70, 5] },
     { center = [40, 0], size = [70, 5] }
   ]
   
   [wall_environment]
   obstacles = [
     { center = [0, 0], size = [10, 5] }
   ]
   
   [combined_environment]
   obstacles = [
     { center = [-50, 0], size = [70, 2] },
     { center = [50, 0], size = [70, 2] },
     { center = [5, -1], size = [3, 10] }
   ]
   ```

3. **Agent Configuration**:
   ```toml
   [[agents]]
   radius = 2.5
   initial_position = [-4, 10]
   target_position = [-10, -10]
   
   [[agents]]
   radius = 1.5
   initial_position = [-10, 5]
   target_position = [10, -15]
   
   [[agents]]
   radius = 1.0
   initial_position = [-15, -10]
   target_position = [10, 10]
   
   [[agents]]
   radius = 2.5
   initial_position = [0, -10]
   target_position = [-10, 15]
   ```

### Future Improvements

Potential improvements to the model include:
- Supporting variable numbers of agents (removing the hard-coded limit of 4)
- Implementing more sophisticated environment representations (beyond rectangles)
- Adding dynamic obstacles that move over time
- Using more advanced approximation methods for non-linear functions
- Implementing online/incremental inference for real-time applications
- Creating a configuration file system to replace hard-coded parameters
- Adding more complex state dynamics models (e.g., including acceleration)
- Implementing hierarchical planning for longer time horizons
- Adding uncertainty in environmental perception

### Output Organization

The implementation now organizes outputs into a clear subdirectory structure:

1. **Directory Structure**:
   ```
   results/YYYY-MM-DD_HH-MM-SS/
   ├── animations/      # Contains all animated GIFs
   ├── data/            # Contains all data files (CSV, logs)
   ├── heatmaps/        # Contains environment heatmaps
   ├── visualizations/  # Contains static visualizations
   └── README.md        # Overview of results
   ```

2. **Animations Directory (`animations/`)**:
   - `door_42.gif`, `door_123.gif`: Animations of door environment experiments
   - `wall_42.gif`, `wall_123.gif`: Animations of wall environment experiments
   - `combined_42.gif`, `combined_123.gif`: Animations of combined environment experiments
   - `control_signals.gif`: Visualization of control signals over time

3. **Data Directory (`data/`)**:
   - `paths.csv`: Path data for all agents
   - `controls.csv`: Control inputs for all agents
   - `uncertainties.csv`: Path uncertainties for all agents
   - `experiment.log`: Detailed experiment log
   - `convergence_metrics.csv`: ELBO convergence metrics (if available)

4. **Heatmaps Directory (`heatmaps/`)**:
   - `door_environment_heatmap.png`: Distance field for door environment
   - `wall_environment_heatmap.png`: Distance field for wall environment
   - `combined_environment_heatmap.png`: Distance field for combined environment
   - `obstacle_distance.png`: Obstacle distance heatmap for the current experiment

5. **Visualizations Directory (`visualizations/`)**:
   - `path_visualization.png`: Static visualization of agent paths
   - `path_uncertainty.png`: Visualization of uncertainty in agent paths
   - `control_magnitudes.png`: Plot of control signal magnitudes over time
   - `convergence.png`: ELBO convergence plot
   - `convergence_detailed.png`: Detailed convergence analysis

This organization improves the clarity of the output structure and makes it easier to locate specific results. The subdirectory structure is created automatically when running experiments, and all visualization functions have been updated to use this structure.
