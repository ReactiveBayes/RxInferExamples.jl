# Generalized World Module for Active Inference Car Examples
# Supports multiple environments, obstacle management, and dynamic conditions

@doc """
Generalized world module for active inference car environments.

This module provides a comprehensive framework for managing different types
of car environments, including obstacle placement, boundary conditions,
dynamic elements, and event handling.

## Supported Environment Types
- **Mountain Environment**: Classic mountain car with gravitational landscape
- **Race Track**: Circuit racing with lap timing and track features
- **Urban Environment**: City driving with traffic, intersections, and navigation
- **Custom Environment**: User-defined environments with extensible features

## Key Features
- **Multi-environment Support**: Different world types with specific behaviors
- **Dynamic Obstacles**: Moving obstacles with collision detection
- **Event System**: Time-based events and environmental changes
- **Stochastic Elements**: Random elements for realistic simulation
- **Boundary Management**: Flexible boundary conditions and wrapping
- **State Management**: Clean state handling with reset capabilities
"""
module World

using LinearAlgebra
using Statistics
using Random
using Distributions
import ..Config: WORLD, get_config_value
import ..Physics: AbstractPhysicsModel, get_landscape_function, get_landscape_coordinates, get_mass

# ==================== ABSTRACT INTERFACES ====================

@doc """
Abstract base type for world environments.

Provides common interface for different environment implementations.
"""
abstract type AbstractWorld end

@doc """
Abstract type for obstacles in the environment.

Obstacles can be static or dynamic, with different collision behaviors.
"""
abstract type AbstractObstacle end

@doc """
Abstract type for environmental events.

Events trigger changes in the environment at specific times or conditions.
"""
abstract type AbstractEvent end

# ==================== OBSTACLE IMPLEMENTATIONS ====================

@doc """
Static obstacle with fixed position.

Simple obstacle that doesn't move over time.
"""
struct StaticObstacle <: AbstractObstacle
    position::Float64
    radius::Float64
    collision_type::Symbol  # :stop, :bounce, :avoid

    function StaticObstacle(position::Float64, radius::Float64 = 1.0, collision_type::Symbol = :avoid)
        new(position, radius, collision_type)
    end
end

@doc """
Dynamic obstacle that moves over time.

Obstacle with position that changes according to a motion pattern.
"""
struct DynamicObstacle <: AbstractObstacle
    position_function::Function  # Function of time -> position
    radius::Float64
    collision_type::Symbol
    velocity::Float64

    function DynamicObstacle(position_function::Function, radius::Float64 = 1.0,
                           collision_type::Symbol = :avoid, velocity::Float64 = 0.0)
        new(position_function, radius, collision_type, velocity)
    end
end

@doc """
Traffic obstacle for urban environments.

Represents other vehicles or traffic elements with realistic behavior.
"""
struct TrafficObstacle <: AbstractObstacle
    start_position::Float64
    target_position::Float64
    speed::Float64
    radius::Float64
    collision_type::Symbol

    function TrafficObstacle(start_pos::Float64, target_pos::Float64,
                           speed::Float64, radius::Float64 = 2.0, collision_type::Symbol = :avoid)
        new(start_pos, target_pos, speed, radius, collision_type)
    end
end

# ==================== EVENT IMPLEMENTATIONS ====================

@doc """
Time-based event that triggers at a specific time.

Event that executes when the simulation time reaches a threshold.
"""
struct TimeEvent <: AbstractEvent
    trigger_time::Float64
    action::Function  # Function to execute when triggered

    function TimeEvent(trigger_time::Float64, action::Function)
        new(trigger_time, action)
    end
end

@doc """
State-based event that triggers on condition.

Event that executes when the world state meets certain conditions.
"""
struct StateEvent <: AbstractEvent
    condition::Function  # Function(state) -> Bool
    action::Function     # Function to execute when triggered

    function StateEvent(condition::Function, action::Function)
        new(condition, action)
    end
end

# ==================== WORLD ENVIRONMENT IMPLEMENTATIONS ====================

@doc """
Mountain world environment for mountain car scenarios.

Classic environment with gravitational landscape, static obstacles,
and goal-oriented navigation.
"""
mutable struct MountainWorld <: AbstractWorld
    # Environment bounds
    x_min::Float64
    x_max::Float64
    v_min::Float64
    v_max::Float64

    # Obstacles and events
    obstacles::Vector{AbstractObstacle}
    events::Vector{AbstractEvent}

    # State variables
    current_time::Float64
    current_state::Vector{Float64}
    goal_position::Float64
    goal_tolerance::Float64

    # Environment properties
    landscape_function::Function
    stochastic::Bool

    function MountainWorld(;
        x_min::Float64 = -2.0,
        x_max::Float64 = 2.0,
        v_min::Float64 = -1.0,
        v_max::Float64 = 1.0,
        goal_position::Float64 = 0.5,
        goal_tolerance::Float64 = 0.1,
        obstacles::Vector{AbstractObstacle} = AbstractObstacle[],
        events::Vector{AbstractEvent} = AbstractEvent[],
        stochastic::Bool = false
    )
        landscape_function = get_landscape_function(:mountain_car)
        current_state = [x_min + (x_max - x_min) / 2, 0.0]  # Start at center
        current_time = 0.0

        new(
            x_min, x_max, v_min, v_max,
            obstacles, events,
            current_time, current_state, goal_position, goal_tolerance,
            landscape_function, stochastic
        )
    end
end

@doc """
Race track world environment for racing scenarios.

Circuit-based environment with lap timing, multiple checkpoints,
and racing-specific features.
"""
mutable struct RaceWorld <: AbstractWorld
    # Track properties
    track_length::Float64
    lap_count::Int
    checkpoints::Vector{Float64}

    # Environment bounds (circular track)
    x_min::Float64
    x_max::Float64
    v_min::Float64
    v_max::Float64

    # Racing state
    current_lap::Int
    lap_times::Vector{Float64}
    current_time::Float64
    current_state::Vector{Float64}

    # Track features
    obstacles::Vector{AbstractObstacle}
    events::Vector{AbstractEvent}
    landscape_function::Function
    stochastic::Bool

    function RaceWorld(;
        track_length::Float64 = 100.0,
        lap_count::Int = 3,
        checkpoints::Vector{Float64} = [25.0, 50.0, 75.0],
        x_min::Float64 = 0.0,
        x_max::Float64 = 100.0,
        v_min::Float64 = 0.0,
        v_max::Float64 = 5.0,
        obstacles::Vector{AbstractObstacle} = AbstractObstacle[],
        events::Vector{AbstractEvent} = AbstractEvent[],
        stochastic::Bool = true
    )
        landscape_function = get_landscape_function(:race_car)
        current_state = [0.0, 1.0]  # Start at beginning with some speed
        current_time = 0.0

        new(
            track_length, lap_count, checkpoints,
            x_min, x_max, v_min, v_max,
            1, Float64[], current_time, current_state,
            obstacles, events, landscape_function, stochastic
        )
    end
end

@doc """
Urban world environment for autonomous driving scenarios.

Complex urban environment with traffic, intersections, dynamic obstacles,
and navigation challenges.
"""
mutable struct UrbanWorld <: AbstractWorld
    # City layout
    intersections::Vector{Tuple{Float64, Float64}}  # (x, y) positions
    traffic_lights::Dict{Tuple{Float64, Float64}, Symbol}  # :red, :yellow, :green
    road_segments::Vector{Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}}

    # Environment bounds
    x_min::Float64
    x_max::Float64
    v_min::Float64
    v_max::Float64

    # Dynamic elements
    traffic_obstacles::Vector{TrafficObstacle}
    pedestrians::Vector{DynamicObstacle}

    # State management
    current_time::Float64
    current_state::Vector{Float64}
    goal_position::Float64
    goal_tolerance::Float64

    # Events and obstacles
    obstacles::Vector{AbstractObstacle}
    events::Vector{AbstractEvent}
    landscape_function::Function
    stochastic::Bool

    function UrbanWorld(;
        x_min::Float64 = -50.0,
        x_max::Float64 = 50.0,
        v_min::Float64 = -2.0,
        v_max::Float64 = 3.0,
        goal_position::Float64 = 25.0,
        goal_tolerance::Float64 = 2.0,
        intersections::Vector{Tuple{Float64, Float64}} = [(0.0, 0.0), (20.0, 0.0), (0.0, 20.0)],
        traffic_lights::Dict{Tuple{Float64, Float64}, Symbol} = Dict((0.0, 0.0) => :green),
        road_segments::Vector{Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}} = [
            ((-50.0, 0.0), (50.0, 0.0)),
            ((0.0, -50.0), (0.0, 50.0))
        ],
        traffic_obstacles::Vector{TrafficObstacle} = TrafficObstacle[],
        pedestrians::Vector{DynamicObstacle} = DynamicObstacle[],
        obstacles::Vector{AbstractObstacle} = AbstractObstacle[],
        events::Vector{AbstractEvent} = AbstractEvent[],
        stochastic::Bool = true
    )
        landscape_function = get_landscape_function(:autonomous_car)
        current_state = [0.0, 1.5]  # Start at origin with urban speed
        current_time = 0.0

        new(
            intersections, traffic_lights, road_segments,
            x_min, x_max, v_min, v_max,
            traffic_obstacles, pedestrians,
            current_time, current_state, goal_position, goal_tolerance,
            obstacles, events, landscape_function, stochastic
        )
    end
end

# ==================== WORLD STATE MANAGEMENT ====================

@doc """
Execute action in the world and update state.

Args:
- world: World instance
- action: Control action to execute
- physics: Physics model for state update
- time_step: Time step for integration

Returns:
- Success status and any collision information
"""
function execute_action!(world::AbstractWorld, action::Float64,
                        physics::AbstractPhysicsModel, time_step::Float64 = 0.1)

    # Get obstacle information for physics
    obstacle_info = get_obstacle_info(world)

    # Update world time
    world.current_time += time_step

    # Get current state
    current_pos = world.current_state[1]
    current_vel = world.current_state[2]

    # Check for collisions before physics update
    collision_info = check_collisions(world, current_pos)

    if collision_info[:collision_detected]
        # Handle collision based on obstacle type
        action = handle_collision(world, collision_info, action)
    end

    # Update physics state using proper integration
    # For now, use simple Euler integration
    dt = 0.1  # Default time step
    acceleration = physics.total_force(current_pos, current_vel, action) / get_mass(physics)
    new_position = current_pos + current_vel * dt
    new_velocity = current_vel + acceleration * dt

    # Apply boundary conditions
    new_state = [new_position, new_velocity]
    new_state = apply_boundaries(world, new_state)

    # Update world state
    world.current_state = new_state

    # Process events
    process_events!(world, time_step)

    return true, collision_info
end

@doc """
Get current observation from the world.

Args:
- world: World instance

Returns:
- Current state observation
"""
function observe(world::AbstractWorld)
    return copy(world.current_state)
end

@doc """
Reset world to initial state.

Args:
- world: World instance

Returns:
- Initial state
"""
function reset!(world::AbstractWorld)
    if world isa MountainWorld
        world.current_state = [world.x_min + (world.x_max - world.x_min) / 2, 0.0]
        world.current_time = 0.0
    elseif world isa RaceWorld
        world.current_state = [0.0, 1.0]
        world.current_time = 0.0
        world.current_lap = 1
        world.lap_times = Float64[]
    elseif world isa UrbanWorld
        world.current_state = [0.0, 1.5]
        world.current_time = 0.0
    end
    return world.current_state
end

@doc """
Get current state of the world.

Args:
- world: World instance

Returns:
- Current state vector
"""
function get_state(world::AbstractWorld)
    return copy(world.current_state)
end

@doc """
Set world state to specific values.

Args:
- world: World instance
- position: New position
- velocity: New velocity
"""
function set_state!(world::AbstractWorld, position::Float64, velocity::Float64)
    world.current_state = [position, velocity]
end

# ==================== COLLISION DETECTION ====================

@doc """
Check for collisions with obstacles.

Args:
- world: World instance
- position: Current position

Returns:
- Collision information dictionary
"""
function check_collisions(world::AbstractWorld, position::Float64)
    collision_info = Dict(
        :collision_detected => false,
        :obstacle_position => NaN,
        :obstacle_type => :none,
        :collision_force => 0.0,
        :avoidance_action => 0.0
    )

    for obstacle in world.obstacles
        if obstacle isa StaticObstacle
            distance = abs(position - obstacle.position)
            if distance < obstacle.radius
                collision_info[:collision_detected] = true
                collision_info[:obstacle_position] = obstacle.position
                collision_info[:obstacle_type] = obstacle.collision_type
                collision_info[:collision_force] = 1.0 / (distance + 0.001)  # Inverse distance force
                break
            end
        elseif obstacle isa DynamicObstacle
            obstacle_pos = obstacle.position_function(world.current_time)
            distance = abs(position - obstacle_pos)
            if distance < obstacle.radius
                collision_info[:collision_detected] = true
                collision_info[:obstacle_position] = obstacle_pos
                collision_info[:obstacle_type] = obstacle.collision_type
                collision_info[:collision_force] = 1.0 / (distance + 0.001)
                break
            end
        end
    end

    return collision_info
end

@doc """
Handle collision based on obstacle type.

Args:
- world: World instance
- collision_info: Collision information
- action: Current action

Returns:
- Modified action to handle collision
"""
function handle_collision(world::AbstractWorld, collision_info::Dict, action::Float64)
    if collision_info[:obstacle_type] == :stop
        return 0.0  # Emergency stop
    elseif collision_info[:obstacle_type] == :bounce
        # Reverse direction with some energy loss
        return -action * 0.5
    elseif collision_info[:obstacle_type] == :avoid
        # Gentle avoidance maneuver
        obstacle_pos = collision_info[:obstacle_position]
        avoidance_force = -sign(world.current_state[1] - obstacle_pos) * 0.02
        return action + avoidance_force
    else
        return action  # No collision handling
    end
end

# ==================== BOUNDARY CONDITIONS ====================

@doc """
Apply boundary conditions to state.

Args:
- world: World instance
- state: Current state

Returns:
- State with boundary conditions applied
"""
function apply_boundaries(world::AbstractWorld, state::Vector{Float64})
    position, velocity = state

    # Apply position boundaries
    if position < world.x_min
        if world isa RaceWorld
            # Wrap around for race track
            position = world.x_max
        else
            # Bounce off boundary for other worlds
            position = world.x_min + (world.x_min - position)
            velocity = -velocity * 0.8  # Energy loss
        end
    elseif position > world.x_max
        if world isa RaceWorld
            # Wrap around for race track
            position = world.x_min
        else
            # Bounce off boundary
            position = world.x_max - (position - world.x_max)
            velocity = -velocity * 0.8
        end
    end

    # Apply velocity boundaries
    velocity = clamp(velocity, world.v_min, world.v_max)

    return [position, velocity]
end

# ==================== EVENT PROCESSING ====================

@doc """
Process pending events in the world.

Args:
- world: World instance
- time_step: Current time step
"""
function process_events!(world::AbstractWorld, time_step::Float64)
    for event in world.events
        if event isa TimeEvent
            if world.current_time >= event.trigger_time
                event.action(world, time_step)
                # Remove event after triggering (single-use events)
                filter!(e -> e !== event, world.events)
            end
        elseif event isa StateEvent
            if event.condition(world.current_state)
                event.action(world, time_step)
                # Keep state events (they can trigger multiple times)
            end
        end
    end
end

# ==================== OBSTACLE MANAGEMENT ====================

@doc """
Get obstacle information for physics calculations.

Args:
- world: World instance

Returns:
- Dictionary of obstacle information
"""
function get_obstacle_info(world::AbstractWorld)
    obstacle_positions = Float64[]
    obstacle_velocities = Float64[]

    for obstacle in world.obstacles
        if obstacle isa StaticObstacle
            push!(obstacle_positions, obstacle.position)
            push!(obstacle_velocities, 0.0)
        elseif obstacle isa DynamicObstacle
            pos = obstacle.position_function(world.current_time)
            push!(obstacle_positions, pos)
            push!(obstacle_velocities, obstacle.velocity)
        end
    end

    return Dict(
        :obstacle_positions => obstacle_positions,
        :obstacle_velocities => obstacle_velocities,
        :obstacle_count => length(obstacle_positions)
    )
end

@doc """
Add obstacle to world.

Args:
- world: World instance
- obstacle: Obstacle to add
"""
function add_obstacle!(world::AbstractWorld, obstacle::AbstractObstacle)
    push!(world.obstacles, obstacle)
end

@doc """
Add event to world.

Args:
- world: World instance
- event: Event to add
"""
function add_event!(world::AbstractWorld, event::AbstractEvent)
    push!(world.events, event)
end

# ==================== WORLD FACTORY FUNCTIONS ====================

@doc """
Create world environment based on car type.

Args:
- car_type: Symbol specifying car type (:mountain_car, :race_car, :autonomous_car)
- custom_params: Optional custom parameters

Returns:
- World instance
"""
function create_world(car_type::Symbol = :mountain_car; custom_params::Dict{Symbol, Any} = Dict{Symbol, Any}())
    base_params = get_world_params(car_type)

    # Merge custom parameters
    params = merge(base_params, custom_params)

    if car_type == :mountain_car
        return MountainWorld(; params...)
    elseif car_type == :race_car
        return RaceWorld(; params...)
    elseif car_type == :autonomous_car
        return UrbanWorld(; params...)
    else
        throw(ArgumentError("Unknown car type: $car_type"))
    end
end

@doc """
Get default world parameters for a car type.

Args:
- car_type: Symbol specifying car type

Returns:
- Dictionary of default parameters
"""
function get_world_params(car_type::Symbol)
    if car_type == :mountain_car
        return Dict(
            :x_min => get_config_value(:world, :x_min, -2.0),
            :x_max => get_config_value(:world, :x_max, 2.0),
            :v_min => get_config_value(:world, :v_min, -1.0),
            :v_max => get_config_value(:world, :v_max, 1.0),
            :goal_position => get_config_value(:world, :target_position, 0.5),
            :goal_tolerance => get_config_value(:world, :goal_tolerance, 0.1),
            :stochastic => get_config_value(:world, :stochastic, false)
        )
    elseif car_type == :race_car
        return Dict(
            :track_length => 100.0,
            :lap_count => 3,
            :checkpoints => [25.0, 50.0, 75.0],
            :x_min => 0.0,
            :x_max => 100.0,
            :v_min => 0.0,
            :v_max => 5.0,
            :stochastic => get_config_value(:world, :stochastic, true)
        )
    elseif car_type == :autonomous_car
        return Dict(
            :x_min => get_config_value(:world, :x_min, -50.0),
            :x_max => get_config_value(:world, :x_max, 50.0),
            :v_min => get_config_value(:world, :v_min, -2.0),
            :v_max => get_config_value(:world, :v_max, 3.0),
            :goal_position => get_config_value(:world, :target_position, 25.0),
            :goal_tolerance => get_config_value(:world, :goal_tolerance, 2.0),
            :stochastic => get_config_value(:world, :stochastic, true)
        )
    else
        throw(ArgumentError("Unknown car type: $car_type"))
    end
end

# ==================== SIMULATION HELPERS ====================

@doc """
Simulate a complete trajectory in the world.

Args:
- world: World instance
- physics: Physics model
- actions: Vector of actions to execute
- time_step: Time step for simulation

Returns:
- Vector of states over the trajectory
"""
function simulate_trajectory(world::AbstractWorld, physics::AbstractPhysicsModel,
                           actions::Vector{Float64}, time_step::Float64 = 0.1)
    states = Vector{Vector{Float64}}(undef, length(actions) + 1)
    states[1] = get_state(world)

    for i in 1:length(actions)
        execute_action!(world, actions[i], physics, time_step)
        states[i + 1] = get_state(world)
    end

    return states
end

@doc """
Check if the goal has been reached.

Args:
- world: World instance
- position: Current position (optional, uses current state if not provided)

Returns:
- Boolean indicating goal achievement
"""
function is_goal_reached(world::AbstractWorld, position::Float64 = world.current_state[1])
    return abs(position - world.goal_position) <= world.goal_tolerance
end

@doc """
Calculate distance to goal.

Args:
- world: World instance
- position: Current position (optional)

Returns:
- Distance to goal
"""
function distance_to_goal(world::AbstractWorld, position::Float64 = world.current_state[1])
    return abs(position - world.goal_position)
end

# ==================== MODULE EXPORTS ====================

export
    # Abstract types
    AbstractWorld,
    AbstractObstacle,
    AbstractEvent,

    # Obstacle types
    StaticObstacle,
    DynamicObstacle,
    TrafficObstacle,

    # Event types
    TimeEvent,
    StateEvent,

    # World types
    MountainWorld,
    RaceWorld,
    UrbanWorld,

    # World management
    execute_action!,
    observe,
    reset!,
    get_state,
    set_state!,

    # Collision handling
    check_collisions,
    handle_collision,
    apply_boundaries,

    # Event processing
    process_events!,
    add_obstacle!,
    add_event!,

    # Factory functions
    create_world,
    get_world_params,

    # Simulation helpers
    simulate_trajectory,
    is_goal_reached,
    distance_to_goal,
    get_obstacle_info

end # module World
