@testitem "Distributions defined here hold numbers that are not `Real`" begin
    using ExponentialFamily, LinearAlgebra

    # Numbers a compiler such as Reactant traces are `Number`s but not `Real`s; complex numbers stand
    # in for them here. The distributions defined in this package hold them, and convert between
    # their parametrisations without leaving the element type.
    m = ComplexF64[1.0, 2.0]
    V = ComplexF64[2.0 0.5; 0.5 1.0]

    for T in (MvNormalMeanCovariance, MvNormalMeanPrecision, MvNormalWeightedMeanPrecision)
        @test T(m, V) isa T{ComplexF64}
    end
    @test convert(MvNormalMeanCovariance{ComplexF64}, MvNormalMeanCovariance(m, V)) isa MvNormalMeanCovariance{ComplexF64}
    @test MvNormalMeanScalePrecision(m, 2.0 + 0.0im) isa MvNormalMeanScalePrecision{ComplexF64}

    for T in (NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision, GammaShapeRate)
        @test T(1.0 + 0.0im, 2.0 + 0.0im) isa T{ComplexF64}
    end
    @test ExponentialFamily.WishartFast(3.0 + 0.0im, V) isa ExponentialFamily.WishartFast{ComplexF64}
    @test ExponentialFamily.InverseWishartFast(3.0 + 0.0im, V) isa ExponentialFamily.InverseWishartFast{ComplexF64}
    @test NormalGamma(1.0 + 0.0im, 1.0, 1.0, 1.0) isa NormalGamma{ComplexF64}
end

@testitem "Relaxing the element bounds introduced no ambiguity" begin
    using ExponentialFamily, Test

    relaxed = [
        "NormalMeanVariance", "NormalMeanPrecision", "NormalWeightedMeanPrecision", "MvNormalMeanCovariance",
        "MvNormalMeanPrecision", "MvNormalWeightedMeanPrecision", "MvNormalMeanScalePrecision",
        "MvNormalMeanScaleMatrixPrecision", "GammaShapeRate", "WishartFast", "InverseWishartFast", "NormalGamma",
        "MvNormalGamma", "MvNormalWishart", "MatrixNormalWishart"
    ]
    ambiguities = Test.detect_ambiguities(ExponentialFamily; recursive = true)
    offending = filter(pair -> any(m -> any(name -> occursin(name, string(m.sig)), relaxed), pair), ambiguities)
    @test isempty(offending)
end
