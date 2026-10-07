using RxInfer, Reactant, LinearAlgebra, StableRNGs, MatrixCorrectionTools, BenchmarkTools

println("RxInfer ", pkgversion(RxInfer), ", Reactant ", pkgversion(Reactant), " (", pathof(Reactant), "), ExponentialFamily ", pkgversion(RxInfer.ExponentialFamily), " (", pathof(RxInfer.ExponentialFamily), "), BayesBase ", pkgversion(RxInfer.BayesBase), ", Julia ", VERSION)
println("TracedRNumber{Float64} <: Real: ", Reactant.TracedRNumber{Float64} <: Real, "; eltype of a traced Float64 array <: Real: ", eltype(Reactant.TracedRArray{Float64, 1}) <: Real)

@model function rotate_ssm(y, m0, V0, A, B, P, Q)
    x_prior ~ MvNormalMeanCovariance(m0, V0)
    x_prev = x_prior
    for i in 1:length(y)
        x[i] ~ MvNormalMeanCovariance(A * x_prev, P)
        y[i] ~ MvNormalMeanCovariance(B * x[i], Q)
        x_prev = x[i]
    end
end

# Probe overrides, script-local, each one a finding:
# 1. Reactant overlays `any(f, ::AbstractArray)`, so `any(ismissing, ::Vector{TracedRArray})` is a traced Bool
RxInfer.inference_check_dataismissing(d::AbstractVector{<:Reactant.AnyTracedRArray}) = false

# The whole inference as one function of arrays: data as a d × n matrix, the posteriors back as arrays
function run_ssm(Y, A, B, P, Q, m0, V0)
    ys = [Y[:, i] for i in 1:size(Y, 2)]
    # The default correction of `*` towards `in` reads each diagonal entry (finding 6)
    result = infer(
        model = rotate_ssm(m0 = m0, V0 = V0, A = A, B = B, P = P, Q = Q), data = (y = ys,),
        options = (context = (matrix_correction = MatrixCorrectionTools.NoCorrection(),),),
    )
    qs = result.posteriors[:x]
    return [mean(q) for q in qs], [cov(q) for q in qs]
end

n = parse(Int, get(ENV, "N", "5"))
θ = π / 35
A = [cos(θ) -sin(θ); sin(θ) cos(θ)]
B, P, Q = Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2), 25.0 * Matrix(1.0I, 2, 2)
rng = StableRNG(1234)
x, Y = zeros(2), zeros(2, n)
for i in 1:n
    global x = rand(rng, MvNormal(A * x, P))
    Y[:, i] = rand(rng, MvNormal(B * x, Q))
end
m0, V0 = zeros(2), 100.0 * Matrix(1.0I, 2, 2)

means, covs = run_ssm(Y, A, B, P, Q, m0, V0)
println("plain arrays: ok, last mean ", means[end])

ours = r"ReactiveMP|RxInfer|MessagePassingRules|ExponentialFamily|BayesBase|FastCholesky|GraphPPL|Rocket"
function report(label, e, bt)
    println("\n", label, " FAILED: ", typeof(e))
    msg = sprint(showerror, e)
    println("  ", first(split(msg, '\n'), 6) |> l -> join(l, "\n  "))
    frames = filter(sf -> occursin(ours, string(sf.file)) || occursin(ours, string(sf.func)), stacktrace(bt))
    for sf in first(frames, 25)
        println("    ", sf.func, " @ ", replace(string(sf.file), homedir() => "~"), ":", sf.line)
    end
end

rargs = Reactant.to_rarray.((Y, A, B, P, Q, m0, V0))

if get(ENV, "EAGER", "false") == "true"
    try
        means, covs = run_ssm(rargs...)
        println("eager ConcreteRArray: ok, ", typeof(means[end]))
    catch e
        report("eager ConcreteRArray", e, catch_backtrace())
    end
end

compiled = try
    t = @elapsed c = @compile sync = true run_ssm(rargs...)
    println("\n@compile: ok in ", round(t, digits = 1), " s")
    c
catch e
    report("@compile", e, catch_backtrace())
    nothing
end

if compiled !== nothing
    rmeans, rcovs = compiled(rargs...)
    Δ = maximum(max(maximum(abs, Array(a) - b), maximum(abs, Array(c) - d)) for (a, b, c, d) in zip(rmeans, means, rcovs, covs))
    println("compiled vs plain: max|Δ| = ", Δ)
end

if compiled !== nothing && get(ENV, "BENCH", "true") == "true"
    bp = @benchmark run_ssm($Y, $A, $B, $P, $Q, $m0, $V0)
    bc = @benchmark $compiled($(rargs)...)
    for (name, b) in (("plain infer", bp), ("compiled", bc))
        m = median(b)
        println(rpad(name, 14), round(m.time / 1e3, digits = 1), " μs, ", m.allocs, " allocs, ", m.memory, " bytes")
    end
    println("speedup: ", round(median(bp).time / median(bc).time, digits = 1), "×")
    # the compiled program itself: count operations in the optimised module
    hlo = string(@code_hlo run_ssm(rargs...))
    println("optimised HLO: ", count(==('\n'), hlo), " lines, ", length(collect(eachmatch(r"stablehlo\.\w+", hlo))), " stablehlo ops")
end
