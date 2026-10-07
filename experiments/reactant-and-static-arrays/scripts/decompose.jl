include("ssm_static_defs.jl")
using BenchmarkTools
const SMR = RxInfer.StandardMessagePassingRules
const ctx = SMR.MessagePassingRulesBase.RuleContext((;))
show_b(name, b) = println(rpad(name, 44), lpad(string(round(median(b).time / 1e3, digits = 2)), 10), " μs  ", lpad(string(median(b).allocs), 8), " allocs  ", lpad(string(median(b).memory), 10), " bytes")

for (label, yy, xx0, AA, BB, PP, QQ) in (("dense", y, x0, A, B, P, Q), ("static", sy, sx0, sA, sB, sP, sQ))
    println("== ", label)
    show_b("infer (total)", @benchmark run($yy, $xx0, $AA, $BB, $PP, $QQ))
    show_b("create_model only", @benchmark RxInfer.create_model(rotate_ssm(x0 = $xx0, A = $AA, B = $BB, P = $PP, Q = $QQ) | (y = $yy,)))
    m = MvNormalMeanCovariance(xx0 isa MvNormalMeanCovariance ? mean(xx0) : mean(xx0), 3.0 * cov(xx0) / 100 + PP)
    l = MvNormalMeanCovariance(yy[1], QQ)
    show_b("scaled(A, m)            (* to out)", @benchmark SMR.scaled($AA, $m))
    show_b("unscaled(ctx, l, B)     (* to in)", @benchmark SMR.unscaled($ctx, $l, $BB))
    show_b("prod(GenericProd, m, l) (equality)", @benchmark prod(GenericProd(), $m, $l))
end
