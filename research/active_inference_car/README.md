# Generalized Active Inference Car Examples

A comprehensive, extensible framework for active inference in various car scenarios including mountain car, racing, and autonomous driving applications. This system demonstrates advanced active inference techniques with modular architecture and production-ready features.

## 🚗 Overview

This project provides a generalized framework for applying active inference to different types of car control problems. Built on the RxInfer probabilistic programming framework, it offers:

- **Multiple Car Types**: Mountain car, race car, and autonomous car scenarios
- **Extensible Architecture**: Easy to add new car types and dynamics
- **Advanced Inference**: Multiple inference algorithms and planning strategies
- **Comprehensive Visualization**: Real-time animations and performance monitoring
- **Production Features**: Logging, testing, benchmarking, and data export

## 🎯 Key Features

### **Multi-Modal Car Support**
- **Mountain Car**: Classic gravitational dynamics with energy management
- **Race Car**: High-speed racing with aerodynamic forces and tire modeling
- **Autonomous Car**: Urban navigation with obstacle avoidance and safety constraints

### **Advanced Active Inference**
- **Modular Inference**: Pluggable inference algorithms (standard, adaptive, multi-objective)
- **Dynamic Planning**: Adaptive planning horizons and multi-scale strategies
- **Online Learning**: Parameter adaptation based on performance feedback
- **Multi-Objective Optimization**: Handles competing goals with priority systems

### **Extensible Physics Engine**
- **Multiple Dynamics Models**: Support for different physics implementations
- **Numerical Integration**: Euler, Runge-Kutta 4, and adaptive integration methods
- **Force Modeling**: Engine forces, friction, aerodynamics, and custom forces
- **Environmental Interactions**: Obstacles, boundaries, and dynamic conditions

### **Production-Ready Features**
- **Comprehensive Logging**: Structured JSON logging with performance metrics
- **Data Export**: CSV and JSON export for analysis and visualization
- **Performance Monitoring**: Real-time timing and memory usage tracking
- **Error Handling**: Robust error handling with graceful degradation
- **Testing Framework**: Comprehensive test suite with 100+ test cases

## 📁 Project Structure

```
research/active_inference_car/
├── Project.toml              # Package configuration and dependencies
├── config.jl                 # Modular configuration system
├── run.jl                    # Main execution script with CLI interface
├── README.md                 # This documentation
├── src/                      # Core implementation modules
│   ├── physics.jl           # Physics models and dynamics
│   ├── world.jl             # Environment and obstacle management
│   ├── agent.jl             # Active inference agents
│   ├── visualization.jl     # Visualization and animation system
│   └── utils.jl             # Utilities, logging, and data export
└── test/                    # Comprehensive test suite
    ├── runtests.jl          # Main test runner
    └── [additional test files]
```

## 🚀 Quick Start

### Prerequisites

- Julia 1.10 or later
- Required packages (automatically installed)

### Basic Usage

```bash
# Navigate to the project directory
cd research/active_inference_car

# Run mountain car example
julia run.jl mountain_car

# Run race car with animations
julia run.jl race_car --animation

# Run autonomous car with export
julia run.jl autonomous_car --export

# Compare all car types
julia run.jl --comparison --animation

# Run with verbose logging
julia run.jl mountain_car --verbose --export
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--help` | Show comprehensive help message |
| `--list-car-types` | List all available car types |
| `--naive` | Run naive policy comparison |
| `--animation` | Create GIF animations |
| `--verbose` | Enable detailed console logging |
| `--structured` | Enable structured JSON logging |
| `--performance` | Enable performance CSV logging |
| `--export` | Export results to JSON/CSV files |
| `--benchmark` | Run performance benchmarking |
| `--comparison` | Create comparison animations |

## 🏎️ Car Types

### Mountain Car
Classic reinforcement learning benchmark with gravitational dynamics.

**Key Features:**
- Gravitational landscape with hills and valleys
- Energy management and momentum building
- Limited engine power requiring swinging behavior

**Usage:**
```bash
julia run.jl mountain_car --animation
```

### Race Car
High-performance racing car with advanced physics.

**Key Features:**
- Aerodynamic drag and downforce modeling
- Tire grip and temperature effects
- Multi-lap racing with sector timing
- High-speed stability considerations

**Usage:**
```bash
julia run.jl race_car --animation --export
```

### Autonomous Car
Urban autonomous vehicle with navigation and safety.

**Key Features:**
- Obstacle detection and avoidance
- Path planning and navigation
- Traffic light and intersection handling
- Multi-objective optimization (safety, efficiency, comfort)

**Usage:**
```bash
julia run.jl autonomous_car --animation --comparison
```

## 🛠️ Configuration System

The system uses a modular configuration approach that supports:

### Configuration Sections
- **Physics**: Engine forces, friction, aerodynamics
- **World**: Environment bounds, obstacles, goals
- **Agent**: Inference parameters, planning horizons
- **Simulation**: Time steps, episodes, experimental settings
- **Visualization**: Plot settings, themes, animation parameters

### Custom Configuration Example

```julia
# Create custom configuration
custom_config = Config.create_custom_config(:mountain_car, Dict(
    :physics => Dict(
        :engine_force_limit => 0.06,      # Increase engine power
        :friction_coefficient => 0.08     # Reduce friction
    ),
    :agent => Dict(
        :planning_horizon => 30,          # Longer planning
        :transition_precision => 1e5      # Higher precision
    ),
    :simulation => Dict(
        :time_steps => 200                # More time steps
    )
))
```

## 📊 Visualization and Output

### Animation Output
- Individual animations for each car type
- Side-by-side comparison animations
- Real-time visualization with multiple panels
- Performance metrics and trajectory visualization

### Data Export
- **JSON Export**: Complete experimental data with metadata
- **CSV Export**: Flattened data for analysis and plotting
- **Performance Logs**: Timing and memory usage tracking
- **Structured Logs**: Machine-readable event logging

### Example Output Files
```
outputs/
├── animations/
│   ├── naive_mountain_car.gif
│   ├── ai_mountain_car.gif
│   └── comparison_mountain_car_vs_race_car.gif
├── results/
│   ├── experiment_TIMESTAMP/
│   │   ├── results.json
│   │   ├── results.csv
│   │   └── visualizations/
└── logs/
    ├── active_inference_car.log
    ├── active_inference_car_structured.jsonl
    └── active_inference_car_performance.csv
```

## 🧪 Testing

### Comprehensive Test Suite
```bash
# Run all tests
julia test/runtests.jl

# Run specific test categories
julia test/runtests.jl --quick          # Essential tests only
julia test/runtests.jl --stress         # Stress tests only
julia test/runtests.jl --integration    # Integration tests only

# Run with verbose output
julia test/runtests.jl --verbose
```

### Test Coverage
- **Configuration System**: Parameter validation and custom configs
- **Physics Engine**: Dynamics models and numerical integration
- **World Management**: Environment simulation and obstacle handling
- **Agent Systems**: Inference algorithms and planning strategies
- **Visualization**: Plotting and animation generation
- **Integration**: Cross-module functionality
- **Performance**: Speed and memory usage benchmarks
- **Error Handling**: Edge cases and failure scenarios

## 🔧 Architecture

### Modular Design Principles
1. **Separation of Concerns**: Each module has a single responsibility
2. **Abstract Interfaces**: Common interfaces for extensibility
3. **Factory Patterns**: Centralized object creation
4. **Configuration-Driven**: Behavior controlled by configuration
5. **Plugin Architecture**: Easy to add new components

### Core Modules

#### Physics Module (`src/physics.jl`)
- Multiple dynamics models (Mountain Car, Race Car, Autonomous Car)
- Numerical integration methods (Euler, RK4, Adaptive)
- Extensible force modeling system
- Real-time physics simulation

#### World Module (`src/world.jl`)
- Environment management with obstacles and events
- Boundary conditions and collision detection
- Dynamic obstacle modeling
- Event-driven environmental changes

#### Agent Module (`src/agent.jl`)
- Multiple inference algorithms (Standard, Adaptive, Multi-objective)
- Planning strategies (Standard, Adaptive, Hierarchical)
- Online learning and parameter adaptation
- Multi-objective optimization

#### Visualization Module (`src/visualization.jl`)
- Flexible visualization system with themes
- Multiple output formats (GIF, PNG, interactive)
- Real-time plotting capabilities
- Modular visualization components

#### Utilities Module (`src/utils.jl`)
- Advanced logging system with multiple formats
- Performance monitoring and benchmarking
- Data export and analysis tools
- System utilities and error handling

## 📈 Performance

### Benchmarking
```bash
# Run performance benchmarks
julia run.jl mountain_car --benchmark

# Benchmark specific iterations
julia run.jl race_car --benchmark --benchmark_iterations 10
```

### Performance Metrics
- **Inference Speed**: Time per inference step
- **Memory Usage**: Memory consumption during simulation
- **Animation Generation**: Time to generate visualizations
- **Data Export**: Time to export experimental results

## 🔬 Advanced Features

### Multi-Objective Optimization
The autonomous car agent supports multi-objective optimization with:
- **Safety**: Collision avoidance and conservative actions
- **Efficiency**: Energy-efficient control policies
- **Comfort**: Smooth acceleration and velocity profiles
- **Goal Reaching**: Effective navigation to target destinations

### Adaptive Inference
The race car agent features adaptive inference that:
- Adjusts precision parameters based on prediction errors
- Modifies planning horizons based on uncertainty
- Learns optimal control strategies online
- Adapts to changing track conditions

### Extensible Physics
The physics engine supports:
- Custom force functions
- User-defined dynamics models
- Environmental interactions
- Stochastic elements

## 🤝 Contributing

### Adding New Car Types
1. Define physics model in `src/physics.jl`
2. Create world environment in `src/world.jl`
3. Implement agent behavior in `src/agent.jl`
4. Add visualization theme in `src/visualization.jl`
5. Update configuration in `config.jl`
6. Add tests in `test/`

### Adding New Inference Algorithms
1. Implement inference struct in `src/agent.jl`
2. Add creation function to agent factory
3. Update configuration parameters
4. Add performance tests
5. Document new algorithm

## 📚 References

### Core References
- **RxInfer Documentation**: [https://rxinfer.ml](https://rxinfer.ml)
- **Active Inference Theory**: [Friston et al.](https://www.nature.com/articles/nrn2787)
- **Variational Inference**: [Blei et al.](https://www.jmlr.org/papers/v18/16-107.html)
- **Mountain Car Problem**: [Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html)

### Advanced Topics
- **Free Energy Principle**: [Friston](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20-%20a%20unified%20brain%20theory.pdf)
- **Probabilistic Programming**: [van de Meent et al.](https://arxiv.org/abs/1809.10756)
- **Control Theory**: [Kirk](https://www.doverpublications.com/9780486442781/optimal-control-theory/)

## 🐛 Troubleshooting

### Common Issues

**Julia Package Installation Issues:**
```bash
# Clear package cache and reinstall
rm -rf ~/.julia/registries
julia -e "using Pkg; Pkg.update()"
```

**Memory Issues During Long Simulations:**
```bash
# Reduce time steps or planning horizon
julia run.jl mountain_car --time_steps 50 --planning_horizon 15
```

**Animation Generation Issues:**
```bash
# Check display capabilities
julia -e "using Plots; plot(1:10, 1:10)"  # Test plotting
```

**Performance Issues:**
```bash
# Enable performance logging
julia run.jl mountain_car --performance --verbose
```

### Getting Help
- Check the test suite: `julia test/runtests.jl --verbose`
- Enable detailed logging: `julia run.jl mountain_car --verbose`
- Review configuration: `julia -e "include('config.jl'); Config.print_configuration()"`

## 📄 License

This project is based on the RxInferExamples.jl repository and follows the same licensing terms. See the main repository LICENSE file for details.

## 🙏 Acknowledgments

- **RxInfer Team**: For the excellent probabilistic programming framework
- **Active Inference Community**: For advancing the theoretical foundations
- **Julia Community**: For the high-performance scientific computing ecosystem

---

**Built with ❤️ using Julia and RxInfer**

*This framework demonstrates the power of active inference for complex control problems and serves as both a research tool and educational resource for the active inference community.*

