# Physics module for the Active Inference Mountain Car example
# Handles all physics-related calculations including forces and landscape geometry

@doc """
Physics module for mountain car environment.

This module implements the physical model of the mountain car environment,
including gravitational forces, friction, engine forces, and landscape geometry.
All functions are designed to be modular and testable.
"""
module Physics

using HypergeometricFunctions: _₂F₁
import ..Config: PHYSICS, VISUALIZATION

@doc """
Create physics functions for the mountain car environment.

Returns:
- Fa: Engine force function based on action
- Ff: Friction force function based on velocity
- Fg: Gravitational force function based on position
- height: Landscape height function based on position
"""
function create_physics(; engine_force_limit::Float64 = PHYSICS.engine_force_limit,
                        friction_coefficient::Float64 = PHYSICS.friction_coefficient)
    # Engine force as function of action (limited by battery/tanh function)
    Fa = (a::Real) -> engine_force_limit * tanh(a)

    # Friction force as function of velocity
    Ff = (y_dot::Real) -> -friction_coefficient * y_dot

    # Gravitational force (horizontal component) as function of position
    Fg = (y::Real) -> begin
        if y < 0
            return -0.05*(2*y + 1)
        else
            return -0.05*((1 + 5*y^2)^(-0.5) + (y^2)*(1 + 5*y^2)^(-3/2) + (y^4)/16)
        end
    end

    # The height of the landscape as a function of the horizontal coordinate
    height = (x::Float64) -> begin
        if x < 0
            h = x^2 + x
        else
            h = x * _₂F₁(0.5, 0.5, 1.5, -5*x^2) +
                x^3 * _₂F₁(1.5, 1.5, 2.5, -5*x^2) / 3 +
                x^5 / 80
        end
        return 0.05*h
    end

    return (Fa, Ff, Fg, height)
end

@doc """
Calculate the next state of the car given current state and action.

Args:
- s_t_min: Current state [position, velocity]
- a_t: Current action
- Fa: Engine force function
- Ff: Friction force function
- Fg: Gravitational force function

Returns:
- Next state [position, velocity]
"""
function next_state(s_t_min::Vector{Float64}, a_t::Float64,
                   Fa::Function, Ff::Function, Fg::Function)
    # Update velocity: v_t = v_{t-1} + F_g(x_{t-1}) + F_f(v_{t-1}) + F_a(a_t)
    v_t = s_t_min[2] + Fg(s_t_min[1]) + Ff(s_t_min[2]) + Fa(a_t)

    # Update position: x_t = x_{t-1} + v_t
    x_t = s_t_min[1] + v_t

    return [x_t, v_t]
end

@doc """
Get landscape coordinates for plotting.

Returns:
- x_coords: Array of x coordinates
- y_coords: Array of corresponding height values
"""
function get_landscape_coordinates()
    x_coords = range(VISUALIZATION.landscape_range[1], VISUALIZATION.landscape_range[2],
                    length=VISUALIZATION.landscape_points)
    _, _, _, height = create_physics()
    y_coords = [height(x) for x in x_coords]
    return x_coords, y_coords
end

# Export public functions
export create_physics, next_state, get_landscape_coordinates

end # module Physics
