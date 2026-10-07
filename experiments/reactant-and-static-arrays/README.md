# RxInfer with StaticArrays and with Reactant: findings

An experiment from 2026-10-07, on RxInfer's `refactor/reactivemp-v7` branch, ReactiveMP v7 and Julia
1.13.1. Not an example: the Reactant notebook is on hold until the upstream changes below land.

## The model

The rotating 2-D linear Gaussian state-space model of *Kalman filtering and smoothing*, with exact
belief propagation:

```julia
@model function rotate_ssm(y, x0, A, B, P, Q)
    x_prior ~ x0
    x_prev = x_prior
    for i in 1:length(y)
        x[i] ~ MvNormalMeanCovariance(A * x_prev, P)
        y[i] ~ MvNormalMeanCovariance(B * x[i], Q)
        x_prev = x[i]
    end
end
```

## StaticArrays: works, rarely faster

With `SVector` data and an `SMatrix` prior and constants, every message and posterior stays static and
agrees with the dense run to 2e-14. The only blocker was FastCholesky: StaticArrays' `cholesky`
rejects a matrix that is not exactly symmetric, and `A Σ Aᵀ` with a rotation is off by 5e-16.
That is fixed in ReactiveBayes/FastCholesky.jl#38.

Each rule is 7–20× faster and allocation-free with static arrays, but `infer` is not: the engine
costs about 1 750 allocations per time step whatever the arrays (`scripts/decompose.jl`). Timings of
`infer` at n = 300 (`scripts/sweep.jl`):

| d | dense | static | speedup | first call, static |
|---|---|---|---|---|
| 2 | 24.4 ms | 23.8 ms | 1.03× | 1.7 s |
| 8 | 24.1 ms | 23.1 ms | 1.04× | 1.8 s |
| 16 | 35.8 ms | 28.7 ms | 1.25× | 6.1 s |
| 24 | 57.3 ms | 41.8 ms | 1.37× | 21.5 s |
| 32 | 85.5 ms | 91.1 ms | 0.94× | 57.2 s |

Static allocations are 574 125 at every d up to 24: the engine's floor.

## Reactant: the whole `infer` compiles

`@compile` of a function calling `infer` traces the reactive engine (Rocket's subscriptions, the
equality chain, rule lookup) at trace time and leaves one straight-line StableHLO program
(`scripts/ssm_reactant_local_patches.jl`):

| n | plain `infer` | compiled | speedup | compile | program |
|---|---|---|---|---|---|
| 5 | 424 μs, 10 655 allocs | 36.7 μs, 48 allocs | 11.6× | 80 s | 249 ops |
| 50 | 3.83 ms, 100 321 allocs | 487 μs, 408 allocs | 7.9× | 175 s | 2 544 ops |

Posteriors agree with plain `infer` to 1e-14. Compile time grows with n since the graph is unrolled;
for an example on a slow CI, compiling one filtering step and looping it, each posterior the next
prior, keeps the compile small.

### What it needed

| # | Where | What | Status |
|---|---|---|---|
| 1 | ExponentialFamily, BayesBase, Distributions | Reactant's traced numbers are `Number`, not `Real`, and every distribution bounds its parameters by `Real` | **EnzymeAD/Reactant.jl#2170** (split number types, traced floats `<: AbstractFloat`) fixes it with every package unchanged: checked on its head e3c22f69 (Julia 1.12.7), 11.5×, max\|Δ\| 3.6e-15 (`scripts/ssm_reactant_pr2170.jl`). Relaxing to `Number` locally also worked (`patches/`) but is not proposed: it admits complex and other non-real numbers (EnzymeAD/Reactant.jl#1570) |
| 2 | BayesBase | `mean(PointMass(x))` of a point no specialised method matches recursed until the stack overflowed | ReactiveBayes/BayesBase.jl#65 |
| 3 | FastCholesky | the dense path cannot trace, and Reactant's `cholesky` supports only `\ vector`; on CPU it is also 50–170× slower than LAPACK, while `lu` calls LAPACK (EnzymeAD/Reactant.jl#3411) | ReactiveBayes/FastCholesky.jl#39 (`TracedCholesky`, through LU; with Cholesky a compiled step at d = 64 was 7× slower than RxInfer, with LU 1.5× faster) |
| 4 | GraphPPL | `x ~ x0` hashed the distribution object; `is_nodelabel` iterated number arrays | ReactiveBayes/GraphPPL.jl#334 |
| 5 | RxInfer | the data check iterated every data array; `UnfactorizedData` with `predictvars` failed | ReactiveBayes/RxInfer.jl#702 |
| 6 | RxInfer | PrettyTables 2 pinned, Reactant requires 3 | ReactiveBayes/RxInfer.jl#701 |
| 7 | StandardMessagePassingRules | the `*` rule's default `ReplaceZeroDiagonalEntries` reads values | pass `options = (context = (matrix_correction = MatrixCorrectionTools.NoCorrection(),),)` |
| 8 | Reactant (EnzymeAD/Reactant.jl#3410) | inside a trace, `any(f, ::AbstractArray)` gives a traced `Bool` or throws for plain-`Bool` `f`, and `sum` of plain `Bool`s is a `Bool`, not an `Int` | reported on EnzymeAD/Reactant.jl#2170; `sum` makes GraphPPL's constraint check reject `x ~ x0` with a traced prior, so the prior is written inside the model (`x_prior ~ MvNormalMeanCovariance(m0, V0)`) |
| 9 | Reactant | `one(::TracedRArray)` and `Matrix(UpperTriangular(traced))` index scalars | avoided |

ReactiveMP's engine and the rule bodies needed no change.

## Scripts

Run in an environment with RxInfer, ReactiveMP and the rule packages developed by path, plus the
PR branches above:

- `scripts/ssm_static.jl`, `scripts/sweep.jl`, `scripts/decompose.jl`: StaticArrays.
- `scripts/ssm_reactant_local_patches.jl`: Reactant 0.2.291 with the local `Number` patches.
- `scripts/ssm_reactant_pr2170.jl`: Reactant from #2170 (Julia 1.12) with ExponentialFamily and
  BayesBase from the registry.
- `scripts/fc_reactant_check.jl`: FastCholesky's operations on traced matrices against dense ones.

## The notebook (on hold)

`examples/Advanced Examples/Compiling RxInfer with Reactant` compiles one filtering step and loops it, each
posterior the next prior: 78 s to compile once (Reactant's warm-up; later sizes compile in under a
second), then per step:

| d | RxInfer | compiled | speed-up |
|---|---|---|---|
| 2 | 125 μs | 10.6 μs | 11.7× |
| 8 | 130 μs | 21.8 μs | 6.0× |
| 32 | 225 μs | 106 μs | 2.1× |
| 64 | 675 μs | 437 μs | 1.5× |

It builds only on EnzymeAD/Reactant.jl#2170 (Julia 1.12, `JULIAUP_CHANNEL=1.12 make examples-env …`), with
the PR branches above developed; CPU only here, no GPU backend on macOS.

## Recreating the environment

Every piece is on a pushed branch:

| Package | Branch |
|---|---|
| RxInfer | `experiment/reactant-support` (`refactor/reactivemp-v7` with #701 and #702) |
| FastCholesky | merge `static-arrays-tolerance` (#38) and `reactant-extension` (#39); they conflict only in `Project.toml`'s version (take 1.5.0) and `docs/src/index.md` (keep both sections) |
| GraphPPL | `traced-arguments` (#334) |
| BayesBase | `pointmass-number-elements` (#65) |
| Reactant | `pull/2170/head`, on Julia 1.12 (its base predates Reactant's Julia 1.13 support) |

Develop these, ReactiveMP and its `lib/*MessagePassingRules*` packages by path into one environment, with
ExponentialFamily and BayesBase otherwise from the registry, then build with
`JULIAUP_CHANNEL=1.12 make examples-env ENVIRONMENT=<env> FILTER="Reactant"`. The StaticArrays example
needs only FastCholesky #38 on RxInfer's v7 branch, on Julia 1.13.
