using RxInfer, StaticArrays, LinearAlgebra, Random, StableRNGs, BenchmarkTools

@model function ssm(y, x0, A, B, P, Q)
    x_prior ~ x0
    x_prev = x_prior
    for i in 1:length(y)
        x[i] ~ MvNormalMeanCovariance(A * x_prev, P)
        y[i] ~ MvNormalMeanCovariance(B * x[i], Q)
        x_prev = x[i]
    end
end

run(y, x0, A, B, P, Q) = infer(model = ssm(x0 = x0, A = A, B = B, P = P, Q = Q), data = (y = y,))

# A random rotation, non-diagonal, so `A Σ Aᵀ` is symmetric only up to rounding
function problem(d, n; rng = StableRNG(d))
    A = Matrix(qr(randn(rng, d, d)).Q)
    B, P, Q = Matrix(1.0I, d, d), Matrix(1.0I, d, d), 25.0 * Matrix(1.0I, d, d)
    x, y = zeros(d), Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        x = rand(rng, MvNormal(A * x, P))
        y[i] = rand(rng, MvNormal(B * x, Q))
    end
    x0 = MvNormalMeanCovariance(zeros(d), 100.0 * Matrix(1.0I, d, d))
    return (y, x0, A, B, P, Q)
end

staticize(d, (y, x0, A, B, P, Q)) = (
    [SVector{d}(yi) for yi in y],
    MvNormalMeanCovariance(SVector{d}(mean(x0)), SMatrix{d, d}(cov(x0))),
    SMatrix{d, d}(A), SMatrix{d, d}(B), SMatrix{d, d}(P), SMatrix{d, d}(Q),
)

n = 300
dims = parse.(Int, split(get(ENV, "DIMS", "2,4,8,12,16"), ","))
println(rpad("d", 4), rpad("dense median", 16), rpad("allocs", 10), rpad("static median", 16), rpad("allocs", 10), rpad("speedup", 9), rpad("1st call dense/static", 24), "max|Δ|   static")
for d in dims
    p = problem(d, n)
    sp = staticize(d, p)
    t1d = @elapsed rd = run(p...)
    t1s = @elapsed rs = run(sp...)
    static = all(q -> mean(q) isa SVector && cov(q) isa SMatrix, rs.posteriors[:x])
    Δ = maximum(max(maximum(abs, mean(a) - mean(b)), maximum(abs, cov(a) - cov(b))) for (a, b) in zip(rd.posteriors[:x], rs.posteriors[:x]))
    bd = @benchmark run($(p)...) seconds = 5
    bs = @benchmark run($(sp)...) seconds = 5
    md, ms = median(bd), median(bs)
    println(
        rpad(d, 4),
        rpad(string(round(md.time / 1e6, digits = 2), " ms"), 16), rpad(md.allocs, 10),
        rpad(string(round(ms.time / 1e6, digits = 2), " ms"), 16), rpad(ms.allocs, 10),
        rpad(string(round(md.time / ms.time, digits = 2), "×"), 9),
        rpad(string(round(t1d, digits = 1), " s / ", round(t1s, digits = 1), " s"), 24),
        rpad(string(round(Δ, sigdigits = 2)), 9), static,
    )
end
