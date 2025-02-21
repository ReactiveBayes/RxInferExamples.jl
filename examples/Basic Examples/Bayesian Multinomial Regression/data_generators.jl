
function generate_switching_data(T, A1, A2, c, Q, R, x_0)
    # Initialize arrays to store states and observations
    x = zeros(2, T)  # State matrix: 2 dimensions × T timesteps
    y = zeros(2, T)  # Observation matrix: 2 dimensions × T timesteps
    
    # Set initial state
    x[:,1] = x_0
    
    # Generate state transitions and observations
    for t in 2:T
        # Switch dynamics multiple times through the sequence
        if t < T/3 || (t >= T/2 && t < 3T/4)
            x[:,t] = A2 * x[:,t-1] + rand(MvNormal(zeros(2), Q))  # First regime
        else
            x[:,t] = A1 * x[:,t-1] + rand(MvNormal(zeros(2), Q))  # Second regime
        end
        
        # Generate observation from current state
        y[:,t] = c * x[:,t] + rand(MvNormal(zeros(2), R))
    end

    return x, y
end
        

# System parameters
T = 1000  # Time horizon
θ = π / 15  # Rotation angle

# Define system matrices
A1 = [cos(θ) -sin(θ); sin(θ) cos(θ)]    # Rotation matrix
A2 = [0.3 -0.1; -0.1 0.5]         
c = [0.6 0.0; 0.0 0.2]                   # Observation/distortion matrix

# Noise parameters
Q = [1.0 0.0; 0.0 1.0]                   # State noise covariance
R = 0.1 * [1.0 0.0; 0.0 1.0]            # Observation noise variance
x_0 = [0.0, 0.0]                         # Initial state vector

# Generate synthetic data
x, y = generate_switching_data(T, A1, A2, c, Q, R, x_0)
y = [y[:,i] for i in 1:T]
x = [x[:,i] for i in 1:T]




using Distributions
using LinearAlgebra
using Plots
import StatsFuns: log2π
using Random
using RxInfer

Random.seed!(42)

# First, let's define the missing structures and functions
"""
    State(x, y, vx, vy, θ, ω)

Contains the state of the drone: position (x,y), velocity (vx,vy), and orientation (θ,ω).
"""
struct State
    x::Float64  # x position
    y::Float64  # y position
    vx::Float64 # x velocity
    vy::Float64 # y velocity
    θ::Float64  # angle
    ω::Float64  # angular velocity
end

get_state(state::State) = (state.x, state.y, state.vx, state.vy, state.θ, state.ω)

"""
    Drone(mass, inertia, radius, limit)

Physical parameters of the drone.
"""
struct Drone
    mass::Float64    # Mass of the drone
    inertia::Float64 # Moment of inertia
    radius::Float64  # Distance from center to motors
    limit::Float64   # Maximum force per motor
end

"""
    Environment(gravity)

Environmental parameters.
"""
struct Environment
    gravity::Float64
end

get_properties(drone::Drone) = (drone.mass, drone.inertia, drone.radius, drone.limit)
get_gravity(env::Environment) = env.gravity

function generate_drone_data(T, dt; noise_level=0.005)
    states = zeros(6, T)
    observations = zeros(2, T)
    
    # System setup
    drone = Drone(1.0, 0.1, 0.2, 10.0)
    env = Environment(9.81)
    hover_force = drone.mass * env.gravity / 2
    
    # Initial state (hovering at y=1m)
    current_state = State(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    states[:, 1] = collect(get_state(current_state))
    
    # Control parameters
    BASE_FORCE_DIFF = 0.2  # Increased base force differential
    NOISE_SCALE_HOVERING = 0.05
    FORCE_BOUNDS = (0.5 * hover_force, 1.5 * hover_force)  # Wider force bounds
    
    # Generate trajectory
    for t in 2:T
        if t < T/2
            # Moving phase: dynamic oscillating forces
            time_factor = 2π * t / (T/4)  # Complete two oscillations during moving phase
            
            # Combine multiple oscillations for more complex movement
            oscillation = sin(time_factor) + 0.5sin(2time_factor) + 0.25sin(3time_factor)
            force_diff = BASE_FORCE_DIFF * oscillation
            
            # Add a forward drift component
            drift = 0.15 * (t / (T/2))  # Gradually increasing forward force
            
            left_force = hover_force + force_diff + drift
            right_force = hover_force - force_diff + drift
        else
            # Hovering phase: small random corrections
            left_force = hover_force + NOISE_SCALE_HOVERING * randn()
            right_force = hover_force + NOISE_SCALE_HOVERING * randn()
        end
        
        # Apply force limits
        actions = clamp.([left_force, right_force], FORCE_BOUNDS...)
        
        # Update state
        next_state = state_transition(collect(get_state(current_state)), 
                                    actions, drone, env, dt)
        
        # Apply damping and constraints
        next_state[3:4] .*= 0.999  # Position velocity damping
        next_state[6] *= 0.95      # Angular velocity damping
        
        # State constraints
        next_state[1] = clamp(next_state[1], -5.0, 5.0)    # x position
        next_state[2] = clamp(next_state[2], 0.0, 5.0)     # y position
        next_state[3:4] = clamp.(next_state[3:4], -2.0, 2.0) # velocities
        next_state[5] = clamp(next_state[5], -π/2, π/2)    # angle (increased range)
        next_state[6] = clamp(next_state[6], -1.5, 1.5)    # angular velocity (increased range)
        
        # Store results
        states[:, t] = next_state
        current_state = State(next_state...)
        
        # Add observation noise to positions
        observations[:, t] = next_state[5:6] + noise_level * randn(2)
    end
    
    # Add noise to first observation
    observations[:, 1] = states[1:2, 1] + noise_level * randn(2)
    
    return states, observations
end

# System parameters
T = 1000  # Time horizon
dt = 0.1   # Time step

"""
    state_transition(state, actions, drone, environment, dt)

This function computes the next state of the drone given the current state, the actions, the drone properties and the environment properties.
"""
function state_transition(state, actions, drone::Drone, environment::Environment, dt)
    # extract drone properties
    m, I, r, limit  = get_properties(drone)

    # extract environment properties
    g = get_gravity(environment)

    # extract feasible actions
    Fl, Fr   = clamp.(actions, 0, limit)
        
    # extract state properties
    x, y, vx, vy, θ, ω = state

    # compute forces and torques
    Fg = m * g
    Fy = (Fl + Fr) * cos(θ) - Fg
    Fx = (Fl - Fr) * sin(θ)
    τ  = (Fl - Fr) * r

    # compute movements
    ax = Fx / m
    ay = Fy / m
    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    x_new  = x + vx * dt + ax * dt^2 / 2
    y_new  = y + vy * dt + ay * dt^2 / 2
        
    # compute rotations
    α = τ / I
    ω_new = ω + α * dt
    θ_new = θ + ω * dt + α * dt^2 / 2
	
    return [x_new, y_new, vx_new, vy_new, θ_new, ω_new]
end

function plot_drone!(p, drone::Drone, state::State; color = :black)
    x, y, x_a, y_a, θ, ω = get_state(state)
    _, _, radius, _ = get_properties(drone)
    dx = radius * cos(θ)
    dy = radius * sin(θ)

    drone_position = [ x ], [ y ]
    drone_engines  = [ x - dx, x + dx ], [ y + dy, y - dy ]
    drone_coordinates = [ x - dx, x, x + dx ], [ y + dy, y, y - dy ]

    rotation_matrix = [ cos(-θ) -sin(-θ); sin(-θ) cos(-θ) ]
    engine_shape = [ -1 0 1; 1 -1 1 ]
    drone_shape  = [ -2 -2 2 2 ; -1 1 1 -1 ]
    
    engine_shape = rotation_matrix * engine_shape
    drone_shape  = rotation_matrix * drone_shape
    engine_marker = Shape(engine_shape[1, :], engine_shape[2, :])
    drone_marker  = Shape(drone_shape[1, :], drone_shape[2, :])
    
    scatter!(p, drone_position[1], drone_position[2]; color = color, label = false, marker = drone_marker)
    scatter!(p, drone_engines[1], drone_engines[2]; color = color, label = false, marker = engine_marker, ms = 10)
    plot!(p, drone_coordinates; color = color, label = false)

    return p
end

# After generating data
x_drone, y_drone = generate_drone_data(T, dt)