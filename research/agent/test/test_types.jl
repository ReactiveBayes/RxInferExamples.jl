# Type System Tests

using Test
include("../src/types.jl")
using .Main: StateVector, ActionVector, ObservationVector

@testset "StateVector Tests" begin
    # Construction
    @test StateVector{2}([1.0, 2.0]) isa StateVector{2}
    @test StateVector([1.0, 2.0]) isa StateVector{2}
    
    # Dimensions
    s = StateVector{3}([1.0, 2.0, 3.0])
    @test length(s) == 3
    @test size(s) == (3,)
    
    # Indexing
    @test s[1] == 1.0
    @test s[2] == 2.0
    @test s[3] == 3.0
    
    # Conversion
    @test Vector(s) == [1.0, 2.0, 3.0]
    
    # Mathematical operations
    s1 = StateVector{2}([1.0, 2.0])
    s2 = StateVector{2}([3.0, 4.0])
    @test (s1 + s2) isa StateVector{2}
    @test Vector(s1 + s2) == [4.0, 6.0]
    @test Vector(s2 - s1) == [2.0, 2.0]
    @test Vector(2.0 * s1) == [2.0, 4.0]
    
    # Iteration
    collected = collect(s1)
    @test collected == [1.0, 2.0]
end

@testset "ActionVector Tests" begin
    # Construction
    @test ActionVector{1}([1.0]) isa ActionVector{1}
    @test ActionVector([1.0]) isa ActionVector{1}
    
    # Dimensions
    a = ActionVector{2}([1.0, 2.0])
    @test length(a) == 2
    @test size(a) == (2,)
    
    # Indexing
    @test a[1] == 1.0
    @test a[2] == 2.0
    
    # Conversion
    @test Vector(a) == [1.0, 2.0]
    
    # Mathematical operations
    a1 = ActionVector{1}([1.0])
    a2 = ActionVector{1}([3.0])
    @test (a1 + a2) isa ActionVector{1}
    @test Vector(a1 + a2) == [4.0]
end

@testset "ObservationVector Tests" begin
    # Construction
    @test ObservationVector{2}([1.0, 2.0]) isa ObservationVector{2}
    @test ObservationVector([1.0, 2.0]) isa ObservationVector{2}
    
    # Dimensions
    o = ObservationVector{3}([1.0, 2.0, 3.0])
    @test length(o) == 3
    @test size(o) == (3,)
    
    # Indexing
    @test o[1] == 1.0
    @test o[2] == 2.0
    @test o[3] == 3.0
    
    # Conversion
    @test Vector(o) == [1.0, 2.0, 3.0]
    
    # Mathematical operations
    o1 = ObservationVector{2}([1.0, 2.0])
    o2 = ObservationVector{2}([3.0, 4.0])
    @test (o1 + o2) isa ObservationVector{2}
    @test Vector(o1 + o2) == [4.0, 6.0]
end

@testset "Type Safety" begin
    # These should all compile without errors
    s = StateVector{2}([1.0, 2.0])
    a = ActionVector{1}([0.5])
    o = ObservationVector{2}([1.1, 2.1])
    
    # Type parameters are preserved
    @test typeof(s).parameters[1] == 2
    @test typeof(a).parameters[1] == 1
    @test typeof(o).parameters[1] == 2
end

