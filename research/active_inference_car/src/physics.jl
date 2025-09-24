# Generalized Physics Module for Active Inference Car Examples
# Supports multiple car types with extensible dynamics models

@doc """
Generalized physics module supporting multiple car types and dynamics models.

This module provides a comprehensive physics simulation framework that can handle
different types of cars (mountain car, race car, autonomous car) with their
specific dynamics, forces, and environmental interactions.

## Supported Dynamics Models
- **Mountain Car**: Gravitational dynamics with friction
- **Race Car**: Aerodynamic forces, tire grip, high-speed dynamics
- **Autonomous Car**: Sensor-based dynamics, obstacle avoidance, path planning
- **Custom**: User-defined physics with extensible interface

## Key Features
- **Modular Design**: Separate physics components for different car types
- **Extensible Forces**: Easy to add new force types and dynamics
- **Numerical Integration**: Multiple integration methods (Euler, RK4, adaptive)
- **Real-time Physics**: Support for stochastic environments and dynamic conditions
- **Performance Optimized**: Efficient computation for real-time simulation
"""
module Physics

using LinearAlgebra
using Statistics
using HypergeometricFunctions: _₂F₁
import ..Config: PHYSICS, WORLD, get_config_value

# ==================== ABSTRACT INTERFACES ====================

@doc """
Abstract base type for all physics models.

This provides a common interface for different physics implementations,
ensuring consistency across car types.
"""
abstract type AbstractPhysicsModel end

@doc """
Abstract type for force functions.

Force functions compute the forces acting on the car at each time step.
"""
abstract type AbstractForce end

@doc """
Abstract type for dynamics integrators.

Handles numerical integration of the equations of motion.
"""
abstract type AbstractIntegrator end

# ==================== PHYSICS MODEL IMPLEMENTATIONS ====================

@doc """
Mountain car physics model with gravitational dynamics.

Classic mountain car problem where the car must build momentum to overcome
gravity and reach the goal on the opposite hill.
"""
struct MountainCarPhysics <: AbstractPhysicsModel
    # Physics parameters
    engine_force_limit::Float64
    friction_coefficient::Float64
    gravity_factor::Float64
    mass::Float64
    time_step::Float64

    # Force functions
    engine_force::Function
    friction_force::Function
    gravitational_force::Function
    total_force::Function

    function MountainCarPhysics(;
        engine_force_limit::Float64 = 0.04,
        friction_coefficient::Float64 = 0.1,
        gravity_factor::Float64 = 0.0025,
        mass::Float64 = 1.0,
        time_step::Float64 = 0.1
    )
        # Create force functions
        engine_force = (action::Float64) -> engine_force_limit * tanh(action)
        friction_force = (velocity::Float64) -> -friction_coefficient * velocity
        gravitational_force = create_mountain_gravity_function()

        # Total force function
        total_force = (position::Float64, velocity::Float64, action::Float64; kwargs...) ->
            engine_force(action) + friction_force(velocity) + gravitational_force(position)

        new(
            engine_force_limit,
            friction_coefficient,
            gravity_factor,
            mass,
            time_step,
            engine_force,
            friction_force,
            gravitational_force,
            total_force
        )
    end
end

@doc """
Race car physics model with aerodynamic and tire dynamics.

High-performance racing car with advanced physics including air resistance,
downforce, tire grip modeling, and track temperature effects.
"""
struct RaceCarPhysics <: AbstractPhysicsModel
    # Physics parameters
    engine_force_limit::Float64
    friction_coefficient::Float64
    air_resistance::Float64
    downforce_coefficient::Float64
    mass::Float64
    tire_grip::Float64
    track_temperature::Float64

    # Force functions
    engine_force::Function
    friction_force::Function
    aerodynamic_force::Function
    downforce::Function
    total_force::Function

    function RaceCarPhysics(;
        engine_force_limit::Float64 = 0.15,
        friction_coefficient::Float64 = 0.05,
        air_resistance::Float64 = 0.01,
        downforce_coefficient::Float64 = 0.02,
        mass::Float64 = 1.0,
        tire_grip::Float64 = 0.95,
        track_temperature::Float64 = 25.0
    )
        # Create force functions
        engine_force = (action::Float64) -> engine_force_limit * tanh(action)
        friction_force = (velocity::Float64, grip_factor::Float64) ->
            -friction_coefficient * velocity * grip_factor

        # Temperature-dependent grip factor
        grip_factor = (temperature::Float64) ->
            tire_grip * (1.0 + 0.02 * (temperature - 25.0))

        # Aerodynamic forces
        aerodynamic_force = (velocity::Float64) ->
            -air_resistance * velocity^2 * sign(velocity)

        downforce = (velocity::Float64) ->
            downforce_coefficient * velocity^2

        # Total force function
        total_force = (position::Float64, velocity::Float64, action::Float64, temperature::Float64) ->
            engine_force(action) +
            friction_force(velocity, grip_factor(temperature)) +
            aerodynamic_force(velocity) +
            downforce(velocity)

        new(
            engine_force_limit,
            friction_coefficient,
            air_resistance,
            downforce_coefficient,
            mass,
            tire_grip,
            track_temperature,
            engine_force,
            friction_force,
            aerodynamic_force,
            downforce,
            total_force
        )
    end
end

@doc """
Autonomous car physics model with sensor-based dynamics.

Urban autonomous vehicle with obstacle detection, path planning,
and safety constraints.
"""
struct AutonomousCarPhysics <: AbstractPhysicsModel
    # Physics parameters
    engine_force_limit::Float64
    friction_coefficient::Float64
    sensor_range::Float64
    reaction_time::Float64
    safety_margin::Float64

    # Control systems
    obstacle_avoidance::Function
    path_following::Function
    emergency_braking::Function
    total_force::Function

    function AutonomousCarPhysics(;
        engine_force_limit::Float64 = 0.08,
        friction_coefficient::Float64 = 0.08,
        sensor_range::Float64 = 10.0,
        reaction_time::Float64 = 0.1,
        safety_margin::Float64 = 2.0
    )
        # Control system functions
        obstacle_avoidance = (position::Float64, obstacles::Vector{Float64}) -> begin
            # Check for obstacles within sensor range
            for obstacle in obstacles
                if abs(position - obstacle) < sensor_range
                    # Emergency obstacle avoidance
                    avoidance_force = -sign(position - obstacle) * engine_force_limit * 0.5
                    return avoidance_force
                end
            end
            return 0.0
        end

        path_following = (current_pos::Float64, target_pos::Float64, current_vel::Float64) -> begin
            # Proportional controller for path following
            position_error = target_pos - current_pos
            velocity_error = 0.0 - current_vel  # Target velocity = 0 at waypoints

            kp = 0.1  # Position gain
            kv = 0.05  # Velocity gain

            control_force = kp * position_error + kv * velocity_error
            return clamp(control_force, -engine_force_limit, engine_force_limit)
        end

        emergency_braking = (velocity::Float64, obstacle_distance::Float64) -> begin
            # Emergency braking if obstacle too close
            if obstacle_distance < safety_margin
                return -engine_force_limit * 0.8  # Strong braking
            end
            return 0.0
        end

        # Total force function
        total_force = (position::Float64, velocity::Float64, action::Float64,
                      target_pos::Float64, obstacles::Vector{Float64}) -> begin
            # Basic friction
            friction = -friction_coefficient * velocity

            # Control systems
            path_control = path_following(position, target_pos, velocity)
            obstacle_avoid = obstacle_avoidance(position, obstacles)

            # Closest obstacle distance
            min_obstacle_dist = minimum([abs(position - obs) for obs in obstacles], init=Inf)

            emergency_brake = emergency_braking(velocity, min_obstacle_dist)

            # Combine forces with safety priorities
            control_force = path_control + obstacle_avoid + emergency_brake
            engine_force = engine_force_limit * tanh(action)

            return friction + control_force + engine_force
        end

        new(
            engine_force_limit,
            friction_coefficient,
            sensor_range,
            reaction_time,
            safety_margin,
            obstacle_avoidance,
            path_following,
            emergency_braking,
            total_force
        )
    end
end

# ==================== FORCE FUNCTION LIBRARY ====================

@doc """
Create gravitational force function for mountain car.

Uses hypergeometric functions to model the complex mountain landscape
and compute gravitational forces.
"""
function create_mountain_gravity_function()
    return (position::Float64) -> begin
        if position < 0
            # Left valley: simple gravitational force
            return -0.0025 * (2 * position + 1)
        else
            # Right hill: complex gravitational landscape
            # Using hypergeometric function for accurate modeling
            h = position * _₂F₁(0.5, 0.5, 1.5, -5 * position^2) +
                position^3 * _₂F₁(1.5, 1.5, 2.5, -5 * position^2) / 3 +
                position^5 / 80
            return -0.0025 * h
        end
    end
end

@doc """
Create aerodynamic drag force function.

Computes air resistance based on velocity squared, with optional
cross-sectional area and drag coefficient parameters.
"""
function create_aerodynamic_force(drag_coefficient::Float64 = 0.47, area::Float64 = 1.0)
    return (velocity::Float64) -> -drag_coefficient * area * velocity^2 * sign(velocity)
end

@doc """
Create tire friction force function with grip modeling.

Includes temperature-dependent tire grip and surface adhesion effects.
"""
function create_tire_friction(base_friction::Float64, tire_compound::Float64 = 1.0)
    return (velocity::Float64, normal_force::Float64, temperature::Float64) -> begin
        # Temperature effect on grip (optimal range 20-30°C)
        temp_factor = 1.0 - 0.02 * abs(temperature - 25.0)
        grip_factor = max(0.5, temp_factor) * tire_compound

        # Dynamic friction based on velocity and normal force
        friction_force = -base_friction * normal_force * velocity * grip_factor
        return friction_force
    end
end

# ==================== DYNAMICS INTEGRATORS ====================

@doc """
Euler integrator for simple forward integration.

Basic first-order integration method, fast but less accurate for
complex dynamics.
"""
struct EulerIntegrator <: AbstractIntegrator
    time_step::Float64

    function EulerIntegrator(time_step::Float64 = 0.1)
        new(time_step)
    end
end

@doc """
Runge-Kutta 4th order integrator for higher accuracy.

More accurate integration method, suitable for racing and high-speed
dynamics.
"""
struct RK4Integrator <: AbstractIntegrator
    time_step::Float64

    function RK4Integrator(time_step::Float64 = 0.1)
        new(time_step)
    end
end

@doc """
Adaptive integrator with error control.

Automatically adjusts time step based on integration error,
optimal for complex autonomous systems.
"""
struct AdaptiveIntegrator <: AbstractIntegrator
    base_time_step::Float64
    error_tolerance::Float64
    min_step::Float64
    max_step::Float64

    function AdaptiveIntegrator(
        base_time_step::Float64 = 0.1,
        error_tolerance::Float64 = 1e-6,
        min_step::Float64 = 0.01,
        max_step::Float64 = 0.5
    )
        new(base_time_step, error_tolerance, min_step, max_step)
    end
end

# ==================== INTEGRATION METHODS ====================

@doc """
Perform Euler integration step.

Args:
- physics: Physics model
- state: Current state [position, velocity]
- action: Control input
- kwargs: Additional parameters for physics model

Returns:
- Next state after integration step
"""
function integrate(integrator::EulerIntegrator, physics::AbstractPhysicsModel,
                  state::Vector{Float64}, action::Float64; kwargs...)
    dt = integrator.time_step

    # Compute acceleration from total force
    acceleration = physics.total_force(state[1], state[2], action; kwargs...) / get_mass(physics)

    # Update state using Euler method
    new_position = state[1] + state[2] * dt
    new_velocity = state[2] + acceleration * dt

    return [new_position, new_velocity]
end

@doc """
Perform Runge-Kutta 4th order integration.

More accurate integration suitable for complex dynamics.
"""
function integrate(integrator::RK4Integrator, physics::AbstractPhysicsModel,
                  state::Vector{Float64}, action::Float64; kwargs...)
    dt = integrator.time_step

    # RK4 integration steps
    k1 = physics.total_force(state[1], state[2], action; kwargs...) / get_mass(physics)
    k2 = physics.total_force(state[1] + 0.5 * state[2] * dt,
                           state[2] + 0.5 * k1 * dt, action; kwargs...) / get_mass(physics)
    k3 = physics.total_force(state[1] + 0.5 * state[2] * dt,
                           state[2] + 0.5 * k2 * dt, action; kwargs...) / get_mass(physics)
    k4 = physics.total_force(state[1] + state[2] * dt,
                           state[2] + k3 * dt, action; kwargs...) / get_mass(physics)

    # Update state
    new_position = state[1] + state[2] * dt
    new_velocity = state[2] + (k1 + 2*k2 + 2*k3 + k4) * dt / 6

    return [new_position, new_velocity]
end

@doc """
Perform adaptive integration with error control.

Automatically adjusts step size for accuracy.
"""
function integrate(integrator::AdaptiveIntegrator, physics::AbstractPhysicsModel,
                  state::Vector{Float64}, action::Float64; kwargs...)
    dt = integrator.base_time_step

    # Perform two half-steps for error estimation
    half_dt = dt / 2

    # First half-step
    mid_state = integrate(EulerIntegrator(half_dt), physics, state, action; kwargs...)
    # Second half-step from midpoint
    end_state = integrate(EulerIntegrator(half_dt), physics, mid_state, action; kwargs...)

    # Full step
    full_state = integrate(EulerIntegrator(dt), physics, state, action; kwargs...)

    # Estimate error
    error = norm(end_state - full_state)
    max_error = integrator.error_tolerance

    if error < max_error
        # Step successful, try larger step next time
        return end_state, min(dt * 1.1, integrator.max_step)
    else
        # Step failed, try smaller step
        return integrate(AdaptiveIntegrator(dt * 0.9, max_error, integrator.min_step, integrator.max_step),
                        physics, state, action; kwargs...)
    end
end

# ==================== PHYSICS FACTORY FUNCTIONS ====================

@doc """
Create physics model based on car type.

Args:
- car_type: Symbol specifying car type (:mountain_car, :race_car, :autonomous_car)
- custom_params: Optional custom parameters to override defaults

Returns:
- Physics model instance
"""
function create_physics(car_type::Symbol = :mountain_car; custom_params::Dict{Symbol, Any} = Dict{Symbol, Any}())
    base_params = get_physics_params(car_type)

    # Merge custom parameters
    params = merge(base_params, custom_params)

    if car_type == :mountain_car
        return MountainCarPhysics(; params...)
    elseif car_type == :race_car
        return RaceCarPhysics(; params...)
    elseif car_type == :autonomous_car
        return AutonomousCarPhysics(; params...)
    else
        throw(ArgumentError("Unknown car type: $car_type"))
    end
end

@doc """
Get default physics parameters for a car type.

Args:
- car_type: Symbol specifying car type

Returns:
- Dictionary of default parameters
"""
function get_physics_params(car_type::Symbol)
    if car_type == :mountain_car
        return Dict(
            :engine_force_limit => get_config_value(:physics, :engine_force_limit, 0.04),
            :friction_coefficient => get_config_value(:physics, :friction_coefficient, 0.1),
            :gravity_factor => get_config_value(:physics, :gravity_factor, 0.0025),
            :mass => get_config_value(:physics, :mass, 1.0),
            :time_step => get_config_value(:physics, :time_step, 0.1)
        )
    elseif car_type == :race_car
        return Dict(
            :engine_force_limit => get_config_value(:physics, :engine_force_limit, 0.15),
            :friction_coefficient => get_config_value(:physics, :friction_coefficient, 0.05),
            :air_resistance => get_config_value(:physics, :air_resistance, 0.01),
            :downforce_coefficient => get_config_value(:physics, :downforce_coefficient, 0.02),
            :mass => get_config_value(:physics, :mass, 1.0),
            :tire_grip => get_config_value(:physics, :tire_grip, 0.95),
            :track_temperature => get_config_value(:physics, :track_temperature, 25.0)
        )
    elseif car_type == :autonomous_car
        return Dict(
            :engine_force_limit => get_config_value(:physics, :engine_force_limit, 0.08),
            :friction_coefficient => get_config_value(:physics, :friction_coefficient, 0.08),
            :sensor_range => get_config_value(:physics, :sensor_range, 10.0),
            :reaction_time => get_config_value(:physics, :reaction_time, 0.1),
            :safety_margin => get_config_value(:physics, :safety_margin, 2.0)
        )
    else
        throw(ArgumentError("Unknown car type: $car_type"))
    end
end

@doc """
Create integrator based on configuration.

Args:
- method: Integration method (:euler, :rk4, :adaptive)
- time_step: Base time step for integration

Returns:
- Integrator instance
"""
function create_integrator(method::Symbol = :euler, time_step::Float64 = 0.1)
    if method == :euler
        return EulerIntegrator(time_step)
    elseif method == :rk4
        return RK4Integrator(time_step)
    elseif method == :adaptive
        return AdaptiveIntegrator(time_step)
    else
        throw(ArgumentError("Unknown integration method: $method"))
    end
end

# ==================== UTILITY FUNCTIONS ====================

@doc """
Get mass of the physics model.

Args:
- physics: Physics model instance

Returns:
- Mass value
"""
function get_mass(physics::AbstractPhysicsModel)
    if physics isa MountainCarPhysics
        return physics.mass
    elseif physics isa RaceCarPhysics
        return physics.mass
    elseif physics isa AutonomousCarPhysics
        return 1.0  # Default mass for autonomous car
    else
        return 1.0
    end
end

@doc """
Get landscape height function for a car type.

Args:
- car_type: Car type symbol

Returns:
- Function that computes landscape height at position
"""
function get_landscape_function(car_type::Symbol)
    if car_type == :mountain_car
        return (x::Float64) -> begin
            if x < 0
                h = x^2 + x
            else
                h = x * _₂F₁(0.5, 0.5, 1.5, -5*x^2) +
                    x^3 * _₂F₁(1.5, 1.5, 2.5, -5*x^2) / 3 +
                    x^5 / 80
            end
            return 0.05 * h  # Scale factor for mountain car
        end
    elseif car_type == :race_car
        return (x::Float64) -> begin
            # Racing track: gentle curves with banking
            track_slope = sin(2π * x / 100) * 0.01  # Banking
            return track_slope * x
        end
    elseif car_type == :autonomous_car
        return (x::Float64) -> begin
            # Urban environment: mostly flat with some hills
            return 0.001 * sin(x / 10) + 0.0001 * x  # Very gentle slope
        end
    else
        return (x::Float64) -> 0.0  # Flat landscape for unknown types
    end
end

@doc """
Get landscape coordinates for plotting.

Args:
- car_type: Car type symbol
- points: Number of points for landscape
- range: (min, max) range for landscape

Returns:
- Tuple of (x_coords, y_coords) arrays
"""
function get_landscape_coordinates(car_type::Symbol = :mountain_car;
                                  points::Int = 400,
                                  range::Tuple{Float64, Float64} = (-2.0, 2.0))
    x_coords = range(range[1], range[2], length = points)
    height_func = get_landscape_function(car_type)
    y_coords = height_func.(x_coords)
    return x_coords, y_coords
end

@doc """
Compute next state using physics model and integrator.

Args:
- physics: Physics model
- integrator: Integration method
- state: Current state [position, velocity]
- action: Control input
- kwargs: Additional parameters

Returns:
- Next state after physics update
"""
function next_state(physics::AbstractPhysicsModel, integrator::AbstractIntegrator,
                   state::Vector{Float64}, action::Float64; kwargs...)
    return integrate(integrator, physics, state, action; kwargs...)
end

# ==================== MODULE EXPORTS ====================

export
    # Physics models
    AbstractPhysicsModel,
    MountainCarPhysics,
    RaceCarPhysics,
    AutonomousCarPhysics,

    # Integrators
    AbstractIntegrator,
    EulerIntegrator,
    RK4Integrator,
    AdaptiveIntegrator,

    # Factory functions
    create_physics,
    create_integrator,
    get_physics_params,

    # Utility functions
    get_mass,
    get_landscape_function,
    get_landscape_coordinates,
    next_state,

    # Force functions
    create_mountain_gravity_function,
    create_aerodynamic_force,
    create_tire_friction,

    # Integration methods
    integrate

end # module Physics
