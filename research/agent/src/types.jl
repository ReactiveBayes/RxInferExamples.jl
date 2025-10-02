# Type System for Agent-Environment Framework
# Provides strongly-typed wrappers around StaticArrays for compile-time safety

using StaticArrays
using LinearAlgebra

"""
StateVector{N} - Strongly typed state vector with N dimensions.

Wraps SVector{N, Float64} for compile-time dimension checking.
"""
struct StateVector{N} <: AbstractVector{Float64}
    data::SVector{N, Float64}
end

# Constructors - avoid infinite recursion by being explicit
StateVector{N}(v::Vector{<:Real}) where {N} = StateVector{N}(SVector{N, Float64}(v...))
StateVector{N}(v::NTuple{N, <:Real}) where {N} = StateVector{N}(SVector{N, Float64}(v))
StateVector(data::SVector{N, Float64}) where {N} = StateVector{N}(data)
StateVector(v::Vector{<:Real}) = StateVector(SVector{length(v), Float64}(v...))

# AbstractVector interface
Base.size(s::StateVector{N}) where {N} = (N,)
Base.length(s::StateVector{N}) where {N} = N
Base.getindex(s::StateVector, i::Int) = s.data[i]
Base.IndexStyle(::Type{<:StateVector}) = IndexLinear()

# Iteration interface
Base.iterate(s::StateVector) = iterate(s.data)
Base.iterate(s::StateVector, state) = iterate(s.data, state)

# Conversion
Base.convert(::Type{Vector{Float64}}, s::StateVector) = Vector(s.data)
Base.convert(::Type{SVector{N, Float64}}, s::StateVector{N}) where {N} = s.data
Base.Vector(s::StateVector) = Vector(s.data)

# Mathematical operations
Base.:+(s1::StateVector{N}, s2::StateVector{N}) where {N} = StateVector(s1.data + s2.data)
Base.:-(s1::StateVector{N}, s2::StateVector{N}) where {N} = StateVector(s1.data - s2.data)
Base.:*(scalar::Real, s::StateVector{N}) where {N} = StateVector(scalar * s.data)
Base.:*(s::StateVector{N}, scalar::Real) where {N} = StateVector(s.data * scalar)
LinearAlgebra.norm(s::StateVector) = norm(s.data)

# Display
Base.show(io::IO, s::StateVector{N}) where {N} = print(io, "StateVector{$N}(", s.data, ")")

"""
ActionVector{M} - Strongly typed action vector with M dimensions.

Wraps SVector{M, Float64} for compile-time dimension checking.
"""
struct ActionVector{M} <: AbstractVector{Float64}
    data::SVector{M, Float64}
end

# Constructors
ActionVector{M}(v::Vector{<:Real}) where {M} = ActionVector{M}(SVector{M, Float64}(v...))
ActionVector{M}(v::NTuple{M, <:Real}) where {M} = ActionVector{M}(SVector{M, Float64}(v))
ActionVector(data::SVector{M, Float64}) where {M} = ActionVector{M}(data)
ActionVector(v::Vector{<:Real}) = ActionVector(SVector{length(v), Float64}(v...))

# AbstractVector interface
Base.size(a::ActionVector{M}) where {M} = (M,)
Base.length(a::ActionVector{M}) where {M} = M
Base.getindex(a::ActionVector, i::Int) = a.data[i]
Base.IndexStyle(::Type{<:ActionVector}) = IndexLinear()

# Iteration interface
Base.iterate(a::ActionVector) = iterate(a.data)
Base.iterate(a::ActionVector, state) = iterate(a.data, state)

# Conversion
Base.convert(::Type{Vector{Float64}}, a::ActionVector) = Vector(a.data)
Base.convert(::Type{SVector{M, Float64}}, a::ActionVector{M}) where {M} = a.data
Base.Vector(a::ActionVector) = Vector(a.data)

# Mathematical operations
Base.:+(a1::ActionVector{M}, a2::ActionVector{M}) where {M} = ActionVector(a1.data + a2.data)
Base.:-(a1::ActionVector{M}, a2::ActionVector{M}) where {M} = ActionVector(a1.data - a2.data)
Base.:*(scalar::Real, a::ActionVector{M}) where {M} = ActionVector(scalar * a.data)
Base.:*(a::ActionVector{M}, scalar::Real) where {M} = ActionVector(a.data * scalar)
LinearAlgebra.norm(a::ActionVector) = norm(a.data)

# Display
Base.show(io::IO, a::ActionVector{M}) where {M} = print(io, "ActionVector{$M}(", a.data, ")")

"""
ObservationVector{K} - Strongly typed observation vector with K dimensions.

Wraps SVector{K, Float64} for compile-time dimension checking.
"""
struct ObservationVector{K} <: AbstractVector{Float64}
    data::SVector{K, Float64}
end

# Constructors
ObservationVector{K}(v::Vector{<:Real}) where {K} = ObservationVector{K}(SVector{K, Float64}(v...))
ObservationVector{K}(v::NTuple{K, <:Real}) where {K} = ObservationVector{K}(SVector{K, Float64}(v))
ObservationVector(data::SVector{K, Float64}) where {K} = ObservationVector{K}(data)
ObservationVector(v::Vector{<:Real}) = ObservationVector(SVector{length(v), Float64}(v...))

# AbstractVector interface
Base.size(o::ObservationVector{K}) where {K} = (K,)
Base.length(o::ObservationVector{K}) where {K} = K
Base.getindex(o::ObservationVector, i::Int) = o.data[i]
Base.IndexStyle(::Type{<:ObservationVector}) = IndexLinear()

# Iteration interface
Base.iterate(o::ObservationVector) = iterate(o.data)
Base.iterate(o::ObservationVector, state) = iterate(o.data, state)

# Conversion
Base.convert(::Type{Vector{Float64}}, o::ObservationVector) = Vector(o.data)
Base.convert(::Type{SVector{K, Float64}}, o::ObservationVector{K}) where {K} = o.data
Base.Vector(o::ObservationVector) = Vector(o.data)

# Mathematical operations
Base.:+(o1::ObservationVector{K}, o2::ObservationVector{K}) where {K} = ObservationVector(o1.data + o2.data)
Base.:-(o1::ObservationVector{K}, o2::ObservationVector{K}) where {K} = ObservationVector(o1.data - o2.data)
Base.:*(scalar::Real, o::ObservationVector{K}) where {K} = ObservationVector(scalar * o.data)
Base.:*(o::ObservationVector{K}, scalar::Real) where {K} = ObservationVector(o.data * scalar)
LinearAlgebra.norm(o::ObservationVector) = norm(o.data)

# Display
Base.show(io::IO, o::ObservationVector{K}) where {K} = print(io, "ObservationVector{$K}(", o.data, ")")

# Export
export StateVector, ActionVector, ObservationVector
