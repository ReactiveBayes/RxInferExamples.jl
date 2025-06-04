# Multi-agent Trajectory Planning: Technical Documentation

This document provides a comprehensive technical overview of the Multi-agent Trajectory Planning implementation using probabilistic inference with RxInfer.jl.

## Table of Contents

- [Module Structure](#module-structure)
- [Runtime Flow](#runtime-flow)
- [Probabilistic Model](#probabilistic-model)
- [Distance Functions](#distance-functions)
- [Configuration System](#configuration-system)
- [Visualization Pipeline](#visualization-pipeline)
- [Experiment Process](#experiment-process)
- [Detailed System Flow](#detailed-system-flow)
- [Mathematical Implementation](#mathematical-implementation)
- [Configuration File Structure](#configuration-file-structure)

## Module Structure

The codebase is organized into several modular components that interact with each other. The following diagram shows the module dependencies:

```mermaid
graph TD
    TP[TrajectoryPlanning.jl] --> E[Environment.jl]
    TP --> M[Models.jl]
    TP --> V[Visualizations.jl]
    TP --> EX[Experiments.jl]
    TP --> CL[ConfigLoader.jl]
    
    M --> HS[HalfspaceNode.jl]
    M --> DF[DistanceFunctions.jl]
    M --> IM[InferenceModel.jl]
    
    IM --> HS
    IM --> DF
    IM --> E
    IM --> CL
    
    EX --> E
    EX --> M
    EX --> V
    
    V --> E
    
    DF --> E
    
    classDef core fill:#f9f,stroke:#333,stroke-width:2px
    classDef model fill:#bbf,stroke:#333,stroke-width:1px
    classDef viz fill:#bfb,stroke:#333,stroke-width:1px
    classDef util fill:#fbb,stroke:#333,stroke-width:1px
    
    class TP core
    class M,IM,HS,DF model
    class V,EX viz
    class E,CL util
```

### Module Descriptions

1. **TrajectoryPlanning.jl**: Main module that re-exports all components
2. **Environment.jl**: Defines environment and agent structures
3. **Models.jl**: Integrates model components (HalfspaceNode, DistanceFunctions, InferenceModel)
4. **Visualizations.jl**: Provides visualization and animation functions
5. **Experiments.jl**: Contains experiment execution logic
6. **ConfigLoader.jl**: Manages configuration loading from TOML
7. **HalfspaceNode.jl**: Defines custom RxInfer node for constraints
8. **DistanceFunctions.jl**: Implements distance calculations for collision avoidance
9. **InferenceModel.jl**: Contains the main probabilistic model and inference logic

## Runtime Flow

The following diagram illustrates the runtime flow of a typical experiment:

```mermaid
sequenceDiagram
    participant U as User
    participant CL as ConfigLoader
    participant E as Environment
    participant IM as InferenceModel
    participant V as Visualizations
    
    U->>CL: load_config("config.toml")
    CL->>CL: validate_config()
    CL->>CL: apply_config()
    CL-->>U: Configuration components
    
    U->>E: Create environment
    U->>E: Create agents
    E-->>U: Environment & agents
    
    U->>IM: path_planning(environment, agents)
    IM->>IM: Setup model
    IM->>IM: Run inference (350 iterations)
    IM-->>U: Inference results
    
    U->>V: animate_paths(environment, agents, paths)
    V->>V: Generate animation frames
    V-->>U: Animation (GIF)
    
    U->>V: plot_elbo_convergence(elbo_values)
    V-->>U: Convergence plot
```

## Probabilistic Model

The probabilistic model is formulated as a factor graph:

```mermaid
graph LR
    subgraph "Agent k, Time t"
        S[state k,t] --> |A, B|S1[state k,t+1]
        C[control k,t] --> S1
        S1 --> |C|P[path k,t]
        P --> |g|Z[z k,t]
        Z --> H[Halfspace]
        Z2[zσ2 k,t] --> H
        Z2 --> G[GammaShapeRate]
    end
    
    subgraph "Time t"
        P1[path 1,t] --> |h|D[d t]
        P2[path 2,t] --> |h|D
        P3[path 3,t] --> |h|D
        P4[path 4,t] --> |h|D
        D --> H2[Halfspace]
        D2[dσ2 t] --> H2
        D2 --> G2[GammaShapeRate]
    end
    
    classDef random fill:#bbf,stroke:#333,stroke-width:1px
    classDef deterministic fill:#bfb,stroke:#333,stroke-width:1px
    classDef factor fill:#fbb,stroke:#333,stroke-width:1px
    
    class S,S1,C,P,P1,P2,P3,P4,Z,Z2,D,D2 random
    class H,H2,G,G2 factor
```

### State-Space Model

The state-space model for each agent is defined as:

```mermaid
graph TD
    subgraph "Agent k"
        S0[state k,1] --> |A|S1[state k,2]
        C1[control k,1] --> |B|S1
        S1 --> |A|S2[state k,3]
        C2[control k,2] --> |B|S2
        S2 --> |...|S3[state k,t]
        C3[control k,t-1] --> |B|S3
        S3 --> |A|S4[state k,t+1]
        C4[control k,t] --> |B|S4
        
        S1 --> |C|P1[path k,1]
        S2 --> |C|P2[path k,2]
        S3 --> |C|P3[path k,t-1]
        S4 --> |C|P4[path k,t]
        
        P1 --> Z1[z k,1]
        P2 --> Z2[z k,2]
        P3 --> Z3[z k,t-1]
        P4 --> Z4[z k,t]
    end
    
    classDef state fill:#bbf,stroke:#333,stroke-width:1px
    classDef control fill:#fbb,stroke:#333,stroke-width:1px
    classDef path fill:#bfb,stroke:#333,stroke-width:1px
    classDef constraint fill:#ffb,stroke:#333,stroke-width:1px
    
    class S0,S1,S2,S3,S4 state
    class C1,C2,C3,C4 control
    class P1,P2,P3,P4 path
    class Z1,Z2,Z3,Z4 constraint
```

### Mathematical Model

The model is defined by the following equations:

- **State Transition**: `state[k, t+1] ~ A * state[k, t] + B * control[k, t]`
- **Observation Model**: `path[k, t] ~ C * state[k, t+1]`
- **Environment Constraints**: `z[k, t] ~ g(environment, rs[k], path[k, t])` and `z[k, t] ~ Halfspace(0, zσ2[k, t], γ)`
- **Collision Avoidance**: `d[t] ~ h(environment, rs, path[1, t], path[2, t], path[3, t], path[4, t])` and `d[t] ~ Halfspace(0, dσ2[t], γ)`

## Distance Functions

The distance functions are crucial for collision avoidance:

```mermaid
graph LR
    subgraph "DistanceFunctions"
        D[distance] --> DR[distance(r::Rectangle, state)]
        D --> DE[distance(env::Environment, state)]
        G[g] --> D
        H[h] --> D
        S[softmin]
    end
    
    DR --> "Calculates distance<br/>from point to rectangle"
    DE --> "Minimum distance to<br/>any obstacle via softmin"
    G --> "Distance with<br/>radius offset"
    H --> "Minimum pairwise<br/>distance between agents"
    S --> "Smooth approximation<br/>of min function"
```

### Softmin Function

The softmin function is a differentiable approximation of the minimum function:

```
softmin(x; l=SOFTMIN_TEMPERATURE) = -logsumexp(-l .* x) / l
```

This allows for smooth gradient-based optimization during inference.

## Configuration System

The configuration system manages all parameters for the model:

```mermaid
graph TD
    subgraph "ConfigLoader"
        LC[load_config] --> VC[validate_config]
        AC[apply_config] --> GMP[get_model_parameters]
        AC --> GAC[get_agents_from_config]
        AC --> GEC[get_environment_from_config]
        AC --> GVP[get_visualization_parameters]
        AC --> GEP[get_experiment_parameters]
    end
    
    TOML[config.toml] --> LC
    
    LC --> AC
    
    AC --> CFG[Configuration Components]
    
    CFG --> MP[model_params]
    CFG --> A[agents]
    CFG --> E[environments]
    CFG --> VP[vis_params]
    CFG --> EP[exp_params]
    
    classDef config fill:#bbf,stroke:#333,stroke-width:1px
    classDef function fill:#bfb,stroke:#333,stroke-width:1px
    classDef output fill:#fbb,stroke:#333,stroke-width:1px
    
    class TOML config
    class LC,VC,AC,GMP,GAC,GEC,GVP,GEP function
    class CFG,MP,A,E,VP,EP output
```

### Configuration Parameters

The configuration file (`config.toml`) contains sections for:

1. **Model Parameters**: Time step, matrices, iterations, etc.
2. **Agent Configurations**: Radius, initial and target positions
3. **Environment Definitions**: Obstacle positions and sizes
4. **Visualization Parameters**: Plot boundaries, FPS, etc.
5. **Experiment Parameters**: Random seeds, output filenames

## Visualization Pipeline

The visualization system creates animations and plots:

```mermaid
graph TD
    subgraph "Visualizations"
        AP[animate_paths] --> PE[plot_environment]
        AP --> PMAP[plot_marker_at_position]
        
        PE --> PR[plot_rectangle]
        
        PEC[plot_elbo_convergence] --> MM[movmean]
    end
    
    IR[Inference Results] --> AP
    EV[ELBO Values] --> PEC
    
    AP --> GIF[Animation GIF]
    PEC --> PNG[Convergence Plot]
    
    classDef function fill:#bbf,stroke:#333,stroke-width:1px
    classDef input fill:#bfb,stroke:#333,stroke-width:1px
    classDef output fill:#fbb,stroke:#333,stroke-width:1px
    
    class AP,PE,PMAP,PR,PEC,MM function
    class IR,EV input
    class GIF,PNG output
```

### Animation Process

1. For each time step:
   - Plot the environment with obstacles
   - Plot each agent at its position
   - Draw the path taken so far
   - Optionally show target positions
2. Combine frames into an animated GIF

## Experiment Process

The experiment workflow integrates all components:

```mermaid
graph TD
    subgraph "Experiments"
        EA[execute_and_save_animation] --> PP[path_planning]
        EA --> AP[animate_paths]
        EA --> PEC[plot_elbo_convergence]
        
        RAE[run_all_experiments] --> CDE[create_door_environment]
        RAE --> CWE[create_wall_environment]
        RAE --> CCE[create_combined_environment]
        RAE --> CSA[create_standard_agents]
        RAE --> EA
    end
    
    PP --> IR[Inference Results]
    AP --> AGIF[Animation GIF]
    PEC --> CPNG[Convergence Plot]
    
    classDef function fill:#bbf,stroke:#333,stroke-width:1px
    classDef input fill:#bfb,stroke:#333,stroke-width:1px
    classDef output fill:#fbb,stroke:#333,stroke-width:1px
    
    class EA,PP,AP,PEC,RAE,CDE,CWE,CCE,CSA function
    class IR,AGIF,CPNG output
```

### Full Execution Flow

```mermaid
flowchart TD
    Start([Start]) --> LC[Load Configuration]
    LC --> CE[Create Environment]
    CE --> CA[Create Agents]
    CA --> PP[Run Path Planning]
    PP --> CR[Process Results]
    CR --> CV[Create Visualizations]
    CV --> SV[Save Visualizations]
    SV --> End([End])
    
    subgraph "Path Planning"
        PP1[Setup Model] --> PP2[Initialize Variables]
        PP2 --> PP3[Run Inference]
        PP3 --> PP4[Extract Posteriors]
    end
    
    PP --> PP1
    PP4 --> CR
    
    subgraph "Visualizations"
        CV1[Animate Paths] --> CV2[Plot ELBO Convergence]
        CV2 --> CV3[Visualize Control Signals]
    end
    
    CV --> CV1
```

## Detailed System Flow

The following diagram provides a more detailed view of the probabilistic model execution and the connections between key components:

```mermaid
graph LR
    subgraph "Probabilistic Model Execution"
        PM[InferenceModel] --> PP[path_planning]
        PP --> PM1[Setup path_planning_model]
        PM1 --> INF[Run inference]
        INF --> RES[Extract results]
    end

    subgraph "Agents and Environment"
        ENV[Environment] --> AG[Agents]
        ENV --> OBS[Obstacles]
        AG --> RAD[Agent Radius]
        AG --> IP[Initial Position]
        AG --> TP[Target Position]
        OBS --> RECT[Rectangle obstacles]
    end

    subgraph "Model Definition"
        PMD[path_planning_model] --> SSM[State space model]
        PMD --> CON[Constraints]
        SSM --> ST[State transitions<br/>A, B matrices]
        SSM --> OM[Observation model<br/>C matrix]
        CON --> EC[Environment constraints]
        CON --> CA[Collision avoidance]
    end

    subgraph "Inference Process"
        INIT[Initialization] --> MFV[Mean-field<br/>variational]
        MFV --> LIN[Linearization<br/>approximation]
        LIN --> ITER[Iterations]
        ITER --> CONV[Convergence]
    end

    ENV --> PM
    AG --> PM
    PM --> PMD
    INIT --> INF

    classDef environment fill:#bbf,stroke:#333,stroke-width:1px
    classDef model fill:#bfb,stroke:#333,stroke-width:1px
    classDef inference fill:#fbb,stroke:#333,stroke-width:1px
    classDef process fill:#fbf,stroke:#333,stroke-width:1px

    class ENV,AG,OBS,RAD,IP,TP,RECT environment
    class PMD,SSM,CON,ST,OM,EC,CA model
    class INIT,MFV,LIN,ITER,CONV inference
    class PM,PP,PM1,INF,RES process
```

## Step-by-Step Execution Sequence

The complete sequence of operations from configuration to visualization:

```mermaid
sequenceDiagram
    participant U as User/Script
    participant CL as ConfigLoader
    participant E as Environment
    participant M as Models
    participant IM as InferenceModel
    participant V as Visualizations
    
    U->>CL: load_config("config.toml")
    CL->>CL: validate_config()
    CL->>CL: apply_config()
    CL-->>U: Configuration components
    
    U->>E: Create environment
    Note over U,E: Door, Wall, or Combined
    U->>E: Create agents
    Note over U,E: With radii and positions
    
    U->>M: path_planning(...)
    M->>IM: Setup model
    Note over IM: path_planning_model
    IM->>IM: Initialize variables
    IM->>IM: Configure softmin temperature
    loop nr_iterations times
        IM->>IM: Update messages
        IM->>IM: Track ELBO
    end
    IM-->>M: Inference results
    M-->>U: Paths, controls, ELBO
    
    U->>V: animate_paths(...)
    loop Each time step
        V->>V: Plot environment
        V->>V: Plot agents
        V->>V: Draw paths
    end
    V->>V: Generate GIF
    V-->>U: Animation
    
    U->>V: plot_elbo_convergence(...)
    V-->>U: Convergence plot
```

## Mathematical Implementation

### State-Space Representation

The state for each agent is a 4-dimensional vector:
- `[x_position, x_velocity, y_position, y_velocity]`

The system matrices are:
- `A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]` (state transition)
- `B = [0 0; dt 0; 0 0; 0 dt]` (control input)
- `C = [1 0 0 0; 0 0 1 0]` (observation)

### Variational Inference

The inference uses mean-field variational approximation with the following constraints:
```julia
@constraints function path_planning_constraints()
    q(d, dσ2) = q(d)q(dσ2)
    q(z, zσ2) = q(z)q(zσ2)
end
```

### Prior Distributions

- Initial state: `MvNormal(mean = zeros(4), covariance = initial_state_variance * I)`
- Control inputs: `MvNormal(mean = zeros(2), covariance = control_variance * I)`
- Constraint parameters: `GammaShapeRate(gamma_shape, gamma_scale)`

## Configuration File Structure

The configuration file is structured in TOML format with the following hierarchy:

```mermaid
graph TD
    subgraph "TOML Configuration Structure"
        CT[config.toml] --> MP[model]
        CT --> AG[agents]
        CT --> EN[environments]
        CT --> VS[visualization]
        CT --> EX[experiments]
        
        MP --> MT[dt, gamma, nr_steps, etc.]
        MP --> MM[matrices]
        MP --> PR[priors]
        
        MM --> MA[A matrix]
        MM --> MB[B matrix]
        MM --> MC[C matrix]
        
        AG --> AG1[agent 1]
        AG --> AG2[agent 2]
        AG --> AGN[agent N]
        
        AG1 --> AR[radius]
        AG1 --> AIP[initial_position]
        AG1 --> ATP[target_position]
        
        EN --> ED[door]
        EN --> EW[wall]
        EN --> EC[combined]
        
        ED --> EDO[obstacles]
        
        VS --> VL[x_limits, y_limits]
        VS --> VF[fps]
        VS --> VR[resolution]
        
        EX --> ES[seeds]
        EX --> ERD[results_dir]
        EX --> EFT[filename templates]
    end
    
    classDef section fill:#bbf,stroke:#333,stroke-width:1px
    classDef param fill:#bfb,stroke:#333,stroke-width:1px
    classDef object fill:#fbb,stroke:#333,stroke-width:1px
    
    class CT section
    class MP,AG,EN,VS,EX section
    class MT,MM,PR,AR,AIP,ATP,VL,VF,VR,ES,ERD,EFT param
    class AG1,AG2,AGN,ED,EW,EC,EDO,MA,MB,MC object
```

The TOML configuration allows for flexible customization of all aspects of the simulation without modifying the core code. Example entries from the configuration file:

```toml
# Model parameters
[model]
dt = 1.0
gamma = 1.0
nr_steps = 40
nr_iterations = 350

# Agent configuration
[[agents]]
radius = 2.5
initial_position = [-4.0, 10.0]
target_position = [-10.0, -10.0]

# Environment definition
[environments.door]
description = "Two parallel walls with a gap between them"

[[environments.door.obstacles]]
center = [-40.0, 0.0]
size = [70.0, 5.0]
```

## Conclusion

The Multi-agent Trajectory Planning system uses probabilistic inference to generate collision-free trajectories for multiple agents. The modular architecture allows for easy configuration and extension, while the visualizations provide insights into the planning process.

For implementation details, refer to the source code and comments in each module. 