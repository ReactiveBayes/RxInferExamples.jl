# Active Inference Mountain Car

A comprehensive, production-ready implementation of the Active Inference Mountain Car example demonstrating sophisticated control through probabilistic planning, with enhanced logging, visualization, and analysis capabilities.

## Overview

This example shows how an active inference agent can learn to navigate a challenging mountain car environment by predicting future states multiple steps ahead and selecting actions that minimize expected free energy. The car must reach the goal position (top of the right hill) using limited engine power, requiring intelligent swinging behavior rather than brute force.

## Key Features

### 🎯 **Advanced Active Inference**
- Multi-step probabilistic planning with configurable horizon
- Free energy minimization for optimal control
- Belief updates with temporal consistency
- Sliding window mechanism for real-time planning

### 📊 **Comprehensive Logging & Analytics**
- Structured JSON logging with timestamps
- Performance metrics and timing
- Multiple output formats (CSV, JSON, structured logs)
- Real-time progress tracking with progress bars

### 🎨 **Enhanced Visualization**
- Multiple color themes (default, dark, colorblind-friendly)
- Real-time dashboards with multiple subplots
- Animation with trajectory and prediction visualization
- Performance metrics visualization
- Height contours and velocity field overlays

### 🧪 **Robust Testing**
- Comprehensive test suite (50+ tests)
- Performance benchmarking
- Integration testing
- Error handling validation
- Configuration validation

### 🛠 **Production-Ready Features**
- Modular architecture with clear separation of concerns
- Configuration validation and error handling
- Data export capabilities
- Real-time monitoring tools
- Extensive documentation

## Modular Structure

The example is organized into separate, well-documented modules:

```
research/Active Inference Mountain car/
├── config.jl              # Configuration management
├── src/
│   ├── physics.jl         # Physics simulation & forces
│   ├── world.jl           # Environment state management
│   ├── agent.jl           # Active inference agent
│   ├── visualization.jl   # Enhanced plotting & animation
│   └── utils.jl           # Logging, data export, utilities
├── test/
│   └── runtests.jl        # Comprehensive test suite
├── run.jl                 # Enhanced execution script
├── Project.toml           # Dependencies
├── meta.jl                # Metadata
└── README.md              # This documentation
```

## Module Descriptions

### 🛠 Configuration Management (`config.jl`)
- **Centralized parameter management** for all system components
- **Validation functions** to ensure parameter consistency
- **Named tuples** for type-safe configuration access
- **Easy experimentation** through parameter modification

### ⚗️ Physics Module (`src/physics.jl`)
- **Mountain landscape geometry** with accurate height calculations
- **Force calculations**: Engine force, friction, and gravitational forces
- **State transition dynamics** with numerical stability
- **Modular design** for easy testing and extension
- **Hypergeometric functions** for precise landscape modeling

### 🌍 World Module (`src/world.jl`)
- **Environment state management** with clean API
- **Action execution and observation** functions
- **Trajectory simulation** with configurable parameters
- **Reset and state manipulation** utilities
- **Error handling** for invalid states

### 🧠 Agent Module (`src/agent.jl`)
- **Advanced active inference** implementation
- **Multi-step probabilistic planning** with configurable horizon T
- **Free energy minimization** for optimal action selection
- **Temporal belief updates** with sliding window mechanism
- **Performance monitoring** and prediction error tracking

### 🎨 Enhanced Visualization Module (`src/visualization.jl`)
- **Multiple visualization themes** (default, dark, colorblind-friendly)
- **Real-time dashboards** with multiple subplots
- **Advanced plotting features**:
  - Height contours and velocity field overlays
  - Gradient trajectory visualization
  - Prediction uncertainty visualization
  - Performance metrics plots
- **Animation creation** with trajectory and predictions
- **Interactive elements** with annotations and statistics

### 🧰 Utilities Module (`src/utils.jl`)
- **Enhanced logging system**:
  - Structured JSON logging
  - Performance metrics logging
  - Multiple output formats
- **Data export capabilities**:
  - CSV export with flattening
  - JSON export with nesting
  - Automatic results packaging
- **Performance monitoring**:
  - Context managers for timing
  - Memory usage tracking
  - Statistical analysis tools
- **Validation utilities**:
  - Configuration validation
  - Error detection and reporting
  - Progress bars for long operations

## Usage

### 🚀 Quick Start

```bash
# Navigate to the example directory
cd research/Active\ Inference\ Mountain\ car/

# Run the complete experiment with all features
julia run.jl --verbose --animation --export --performance

# Run with structured logging and data export
julia run.jl --structured --export
```

### 📋 Command Line Options

| Option | Description |
|--------|-------------|
| `--help` | Show comprehensive help message |
| `--verbose` | Enable detailed console logging |
| `--structured` | Enable structured JSON logging |
| `--performance` | Enable performance logging to CSV |
| `--export` | Export results to JSON/CSV files |
| `--animation` | Save animations to GIF files |
| `--naive` | Run only naive policy comparison |

### 🎯 Usage Examples

```bash
# Complete experiment with all features
julia run.jl --verbose --animation --export --performance

# Quick comparison of naive vs AI
julia run.jl --naive --animation

# Structured logging for analysis
julia run.jl --structured --export

# Performance benchmarking
julia run.jl --performance

# Just save animations
julia run.jl --animation
```

### 📊 Output Files

The enhanced system generates multiple output formats:

- **Log files**: `mountain_car_log.txt` (text), `_structured.jsonl` (JSON), `_performance.csv` (CSV)
- **Animation files**: `ai-mountain-car-naive.gif`, `ai-mountain-car-ai.gif`
- **Results directory**: `results/mountain_car_experiment_TIMESTAMP/`
  - `results.json` - Complete experiment data
  - `results.csv` - Flattened data for analysis
- **Performance metrics**: Automatic timing and memory usage tracking

### 🧪 Running Tests

```bash
# Run comprehensive test suite (50+ tests)
julia test/runtests.jl

# Test individual components
julia -e "include(\"test/runtests.jl\"); MountainCarTests.test_physics()"
```

### 🔧 Configuration

All parameters are centralized in `config.jl` and can be easily modified:

```julia
# Example: Modify physics parameters
PHYSICS = (
    engine_force_limit = 0.06,      # Increase engine power
    friction_coefficient = 0.08,    # Reduce friction
)

# Example: Change agent planning horizon
SIMULATION = (
    planning_horizon = 30,          # Longer planning
    time_steps_ai = 200,            # More time steps
)
```

## Configuration Reference

### 🔧 Physics Parameters
- `engine_force_limit`: Maximum engine force (default: 0.04)
- `friction_coefficient`: Friction coefficient (default: 0.1)

### 🌍 World Parameters
- `initial_position`: Starting position (default: -0.5)
- `initial_velocity`: Starting velocity (default: 0.0)
- `target_position`: Goal position (default: 0.5)
- `target_velocity`: Goal velocity (default: 0.0)

### ⚙️ Simulation Parameters
- `time_steps_naive`: Steps for naive policy (default: 100)
- `time_steps_ai`: Steps for AI policy (default: 100)
- `planning_horizon`: Agent planning horizon T (default: 20)
- `naive_action`: Fixed action for naive policy (default: 100.0)

### 🧠 Agent Parameters
- `transition_precision`: State transition precision (default: 1e4)
- `observation_variance`: Observation noise variance (default: 1e-4)
- `control_prior_variance`: Control prior variance (default: 1e6)
- `goal_prior_variance`: Goal prior variance (default: 1e-4)
- `initial_state_variance`: Initial state variance (default: 1e-6)

### 🎨 Visualization Parameters
- `landscape_points`: Points for landscape plot (default: 400)
- `landscape_range`: Range for landscape x-axis (default: (-2.0, 2.0))
- `animation_fps`: Frames per second for animations (default: 24)
- `plot_size`: Size of plots (default: (800, 400))

### 📊 Output Parameters
- `naive_animation`: Naive policy animation filename
- `ai_animation`: AI policy animation filename
- `log_file`: Log file name

## Algorithm Deep Dive

### 🎯 Active Inference Process

The enhanced active inference agent follows this sophisticated process:

1. **Multi-Step Prediction**: Predicts future states over configurable horizon T (default: 20 steps)
2. **Free Energy Minimization**: Evaluates actions based on expected free energy
3. **Optimal Action Selection**: Chooses actions that minimize future uncertainty
4. **Temporal Belief Updates**: Updates beliefs with temporal consistency constraints
5. **Sliding Window Planning**: Shifts planning horizon forward in time for real-time control

### 🧮 Mathematical Foundation

The agent uses a generative model that encodes:

- **State Transitions**: `s_t = f(s_{t-1}, a_t)` with physics constraints
- **Observations**: `x_t = g(s_t)` with observation noise
- **Goal Priors**: Target states with precision weighting
- **Control Priors**: Expected action distributions

### 📈 Performance Monitoring

The enhanced system tracks:
- **Inference Time**: Per-step computation time
- **Prediction Error**: Accuracy of state predictions
- **Action Variance**: Control signal stability
- **Trajectory Efficiency**: Distance traveled vs. progress

## Enhanced Output & Analytics

### 📊 Multiple Output Formats

The system generates comprehensive outputs:

#### Log Files
- **Text logs**: `mountain_car_log.txt` - Human-readable format
- **Structured logs**: `mountain_car_log_structured.jsonl` - JSON Lines format
- **Performance logs**: `mountain_car_log_performance.csv` - Metrics in CSV

#### Animation Files
- **Naive policy**: `ai-mountain-car-naive.gif` - Baseline comparison
- **AI policy**: `ai-mountain-car-ai.gif` - Active inference solution

#### Data Export (when using `--export`)
- **Results directory**: `results/mountain_car_experiment_TIMESTAMP/`
  - `results.json` - Complete structured data
  - `results.csv` - Flattened data for analysis
  - All configuration and performance metrics

#### Real-time Features
- **Progress bars**: Visual feedback for long operations
- **Live metrics**: Performance monitoring during execution
- **Error tracking**: Comprehensive error detection and reporting

## Dependencies

### Core Dependencies
- **RxInfer**: Probabilistic programming framework for inference
- **Plots**: Advanced visualization library
- **HypergeometricFunctions**: Special mathematical functions
- **LinearAlgebra**: Linear algebra operations
- **Logging**: Enhanced logging functionality

### Enhanced Features Dependencies
- **DataFrames**: Data manipulation and analysis
- **CSV**: CSV file I/O operations
- **JSON**: JSON data format support
- **Dates**: Date and time utilities
- **Colors**: Color manipulation
- **ColorSchemes**: Predefined color palettes

## Comprehensive Testing

### 🧪 Test Suite Features

The enhanced test suite includes **50+ comprehensive tests**:

#### Core Functionality Tests
- **Physics validation**: Force calculations, state transitions
- **World simulation**: Environment dynamics, state management
- **Agent behavior**: Planning, inference, action selection
- **Visualization functions**: Plotting, animation, themes
- **Configuration validation**: Parameter checking, error detection

#### Advanced Testing Features
- **Performance benchmarking**: Timing, memory usage
- **Integration testing**: Cross-module functionality
- **Error handling**: Edge cases, invalid inputs
- **Data export validation**: CSV/JSON output verification

### 🚀 Running Tests

```bash
# Run complete test suite
julia test/runtests.jl

# Run individual test categories
julia -e "include(\"test/runtests.jl\"); MountainCarTests.test_physics()"
julia -e "include(\"test/runtests.jl\"); MountainCarTests.test_utils()"
julia -e "include(\"test/runtests.jl\"); MountainCarTests.test_performance()"

# Test with coverage (if coverage tools available)
julia --code-coverage=user test/runtests.jl
```

### 📊 Test Categories

| Category | Tests | Coverage |
|----------|-------|----------|
| Configuration | 11 | Parameter validation |
| Physics | 15 | Force calculations, dynamics |
| World | 7 | State management, simulation |
| Agent | 5 | Planning, inference |
| Visualization | 7 | Plotting, themes, colors |
| Utils | 6 | Logging, data export |
| Performance | 5 | Benchmarking, timing |
| Integration | 5 | Cross-module functionality |
| Error Handling | 4 | Edge cases, validation |

## Extension & Customization

### 🛠 Easy Modification Points

The modular design enables easy experimentation:

#### Physics Modifications
```julia
# Add new force types
Fg = (y) -> y < 0 ? -0.05*(2*y + 1) + wind_force(y) : ...
```

#### Agent Enhancements
```julia
# Modify planning horizon
SIMULATION.planning_horizon = 30

# Adjust precision parameters
AGENT.transition_precision = 1e5
```

#### Visualization Themes
```julia
# Add custom color scheme
colors = get_color_scheme(:custom)
```

#### New Metrics
```julia
# Add custom performance metrics
@info "PERF custom_metric $(calculate_custom_metric())"
```

### 🎨 Visualization Enhancements

#### Custom Themes
```julia
# Create colorblind-friendly theme
custom_colors = (
    landscape = colorant"#1f77b4",
    car = colorant"#ff7f0e",
    # ... other colors
)
```

#### Real-time Dashboards
```julia
# Create custom dashboard
dashboard = create_dashboard(
    car_position, actions, current_step;
    car_states=states,
    performance_metrics=metrics,
    theme=:dark
)
```

### 📈 Performance Optimization

#### Benchmarking
```julia
# Use built-in benchmarking
@benchmark "custom_operation" begin
    # Your operation here
end
```

#### Memory Monitoring
```julia
# Track memory usage
memory_usage()  # Returns current memory usage
```

#### Timing Analysis
```julia
# Context manager for timing
timer = Timer("my_operation")
# ... do work ...
close(timer)  # Logs timing automatically
```

## Advanced Features

### 🔄 Real-time Monitoring

The system supports real-time monitoring:

```julia
# Create real-time plotter
start_monitoring, stop_monitoring, get_data = create_realtime_plotter(data_source)

# Start monitoring
start_monitoring()

# Stop monitoring
stop_monitoring()

# Get collected data
data = get_data()
```

### 📊 Data Analysis Pipeline

Complete data analysis workflow:

```julia
# 1. Run experiment with export
julia run.jl --verbose --export --performance

# 2. Load results
results = JSON.parsefile("results/mountain_car_experiment_TIMESTAMP/results.json")

# 3. Analyze performance
performance_data = results["results"]["active_inference"]
avg_inference_time = performance_data["avg_inference_time"]

# 4. Create custom visualizations
using Plots
plot(performance_data["inference_times"], label="Inference Time")
```

### 🏗 Architectural Benefits

- **Separation of Concerns**: Each module has a single responsibility
- **Type Safety**: Named tuples and structured data
- **Error Handling**: Comprehensive validation and error reporting
- **Extensibility**: Easy to add new features and modify existing ones
- **Testability**: Modular design enables thorough testing
- **Documentation**: Comprehensive docstrings and examples

## References & Further Reading

### 📚 Core References
- **RxInfer Documentation**: [https://rxinfer.ml](https://rxinfer.ml)
- **Active Inference Theory**: [Friston et al.](https://www.nature.com/articles/nrn2787)
- **Mountain Car Problem**: [Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html)

### 🎓 Advanced Topics
- **Variational Inference**: [Blei et al.](https://www.jmlr.org/papers/v18/16-107.html)
- **Free Energy Principle**: [Friston](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20-%20a%20unified%20brain%20theory.pdf)
- **Probabilistic Programming**: [van de Meent et al.](https://arxiv.org/abs/1809.10756)

---

**Based on the original Active Inference Mountain Car example from RxInfer, enhanced with comprehensive logging, visualization, testing, and analysis capabilities for production-ready research and development.**
