using RxInfer, StaticArrays, LinearAlgebra, Random, StableRNGs, BenchmarkTools

println("RxInfer ", pkgversion(RxInfer), " on ReactiveMP ", pkgversion(RxInfer.ReactiveMP), " (", pathof(RxInfer.ReactiveMP), ")")

function generate_data(rng, A, B, P, Q, x0, n)
    x, y = Vector{Vector{Float64}}(undef, n), Vector{Vector{Float64}}(undef, n)
    prev = x0
    for i in 1:n
        x[i] = rand(rng, MvNormal(A * prev, P))
        y[i] = rand(rng, MvNormal(B * x[i], Q))
        prev = x[i]
    end
    return x, y
end

@model function rotate_ssm(y, x0, A, B, P, Q)
    x_prior ~ x0
    x_prev = x_prior
    for i in 1:length(y)
        x[i] ~ MvNormalMeanCovariance(A * x_prev, P)
        y[i] ~ MvNormalMeanCovariance(B * x[i], Q)
        x_prev = x[i]
    end
end

θ = π / 35
A = [cos(θ) -sin(θ); sin(θ) cos(θ)]
B = diageye(2)
P = diageye(2)
Q = 25.0 * diageye(2)
n = 300
_, y = generate_data(StableRNG(1234), A, B, P, Q, zeros(2), n)

run(y, x0, A, B, P, Q) = infer(model = rotate_ssm(x0 = x0, A = A, B = B, P = P, Q = Q), data = (y = y,))

# --- baseline -------------------------------------------------------------------------------
x0 = MvNormalMeanCovariance(zeros(2), 100.0 * diageye(2))
base = run(y, x0, A, B, P, Q)
println("baseline posterior type: ", typeof(base.posteriors[:x][end]))

sy = [SVector{2}(yi) for yi in y]
sA, sB, sP, sQ = SMatrix{2, 2}(A), SMatrix{2, 2}(B), SMatrix{2, 2}(P), SMatrix{2, 2}(Q)
sx0 = MvNormalMeanCovariance(SVector{2}(zeros(2)), SMatrix{2, 2}(100.0 * diageye(2)))
