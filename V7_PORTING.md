# Porting the examples to ReactiveMP v7

> [!WARNING]
> **Delete this file before merging this branch.** It tracks the port while ReactiveMP v7 and
> RxInfer's `refactor/reactivemp-v7` are unregistered, and it does not belong on `main`.

This branch runs the examples on ReactiveMP v7 and RxInfer's branch `refactor/reactivemp-v7`
(RxInfer 6.0.0-DEV), which are not registered yet. To build them, prepare one environment with
every example's dependencies, and with RxInfer, ReactiveMP and ReactiveMP's rule packages
(`ReactiveMP.jl/lib/*MessagePassingRules*`) developed by path, then:

```bash
make examples-env ENVIRONMENT=/path/to/that/environment
```

Checked on Julia 1.13.0, 2026-09-30, against the same notebooks on RxInfer 5.5.2 and
ReactiveMP 6.6.0, where all 47 buildable notebooks pass (Large Language Models needs
`OPENAI_KEY` and is skipped).

## Status

| Status | Notebooks |
|---|---|
| **Unchanged, pass** (14) | Coin Toss Model, Feature Functions in Bayesian Regression, Forgetting Factors for Online Inference, Kalman filtering and smoothing, Drone Dynamics, GP Regression by SSM, Infinite Data Stream, Integrating Neural Networks with Flux.jl and Lux.jl, Parameter Optimisation with Optim.jl, Robotic Arm, Gamma Mixture, Gaussian Mixture, Litter Model, Structural Dynamics with Augmented Kalman Filter |
| **Ported, pass** (28) | Active Inference Mountain car, Assessing People Skills, Bayesian Structured Time Series, Chance Constraints, Conjugate-Computational Variational Message Passing, Multi-agent Trajectory Planning, Nonlinear Sensor Fusion, Solving Linear Systems with Message Passing, Bayesian Binomial Regression, Bayesian Linear Regression, Bayesian Multinomial Regression, Bayesian Networks, Contextual Bandits, Hidden Markov Model, Incomplete Data, POMDP Control, Predicting Bike Rental Demand, Bayesian Trust Learning, Latent Vector Autoregressive Model, Autoregressive Models, Hierarchical Gaussian Filter, Invertible Neural Network Tutorial, Ising Model, ODE Parameter Estimation, Probit Model, RTS vs BIFM Smoothing, Simple Nonlinear Node, Universal Mixtures |
| **Need their owners** (6) | Learning Dynamics with VAEs, Large Language Models, EFE Minimization via Message Passing, T-Maze Active Inference, Autoregressive Active Inference, Recurrent Switching Linear Dynamical System. Plans below. |

Every ported notebook's results were compared with v6's. They are identical, or differ where
ReactiveMP v7 changed a rule on purpose (its migration guide, *Behaviour that changed*), as each
commit on this branch says. Porting found four problems in v7, all fixed with tests in
ReactiveMP and RxInfer: a stale free-energy term in loopy graphs, `DiscreteTransition` with one
control, v6's per-node initial message (now `where { initial_messages = … }`), and the error for
a function node named by its type.

## Plans for the six notebooks that need their owners

Each maps every v6 construct to its v7 counterpart, marks what the migration guide says to ask
the owner about, and lists what ReactiveMP could add to make the port simpler (P1–P8, at the end).

## 1. Learning Dynamics with VAEs (S)

| v6 | v7 |
|---|---|
| `@node VAENode Stochastic [out, x]` | `@define_factor_node(node = VAENode, type = Stochastic, interfaces = [:out, :x])` (Nodes) |
| `struct VAEMeta{F}; vae::F; end` | `struct VAEModel{F} <: AbstractAlgorithm; vae::F; end` (`meta`: what a rule computes is an algorithm) |
| `@rule VAENode(:x, Marginalisation) (q_out::PointMass, meta::VAEMeta)` | `@define_message_update_rule(node = VAENode, target = :x, algorithm = VAEModel, args = (q[:out]::PointMass,), body = (algo, args) -> …algo.vae.encoder…)` |
| `@rule VAENode(:out, …) (q_x::MultivariateNormalDistributionsFamily, meta)` | the same shape, target `:out` |
| `@meta begin ContinuousTransition() -> CTMeta(transition); VAENode() -> VAEMeta(vae) end` | `@algorithm begin ContinuousTransition() -> CTVMP(transition); VAENode() -> VAEModel(vae) end` |
| `ContinuousTransition` | `using ContinuousTransitionMessagePassingRules` + Project.toml (Node packages) |

- The algorithm is parametric, so every rule names `algorithm = VAEModel` explicitly; the node
  declares no algorithm (CLAUDE.md gotcha: a parametric default binds `T{Nothing}`).
- Mean field `q(x)q(Hₛ)q(Λₛ)`, so CT's rules read marginals. The known broken CT rule towards `y`
  from `m[:x]` is not reached, and for `reshape` CT's offset change is a no-op.
- The notebook runs with `free_energy = false`, so CT's corrected average energies don't matter.

**Stop-and-ask:** none. VAEMeta is constant and the Flux calls are pure.
**ReactiveMP changes:** none.
**Risk:** MLDatasets downloads MNIST during the build; this was not a v7 issue, and the v6 build
passed.

## 2. Large Language Models (S)

| v6 | v7 |
|---|---|
| `@node LLMPrior Stochastic [(b, aliases = [belief]), (c, aliases = [context]), (t, aliases = [task])]` | `@define_factor_node(node = LLMPrior, type = Stochastic, interfaces = [(:b, aliases = [:belief]), (:c, aliases = [:context]), (:t, aliases = [:task])])` |
| `@node LLMObservation …` | the same, `[:out, (:b, …), (:t, …)]` |
| `@rule LLMPrior(:b, Marginalisation) (q_c::PointMass{<:String}, q_t::PointMass{<:String})` | `@define_message_update_rule(node = LLMPrior, target = :b, args = (q[:c]::PointMass{<:String}, q[:t]::PointMass{<:String}), body = (args) -> …)` |
| `@rule LLMObservation(:b, …)` | the same |
| `import ReactiveMP: rule_nm_switch_k, softmax!` | delete: neither is used, and v7 has neither |
| `NormalMixture(switch, m, p)` with `MeanField()` | unchanged; v7's NormalMixture declares `factorisation = :meanfield` itself |

- The rules call the OpenAI API, so they are not pure. Declare `pure = false` on both, so the
  opt-in purity audit tells the truth.

**Stop-and-ask:** none, only an owner question. The notebook is skipped in every build (`env_required
= ["OPENAI_KEY"]`), so no build verifies it. Should it gain a canned-response path, keyed on the
env var, so that CI checks the graph and the rules without the network?
**ReactiveMP changes:** none.

## 3. EFE Minimization via Message Passing (M)

Constructs:

| v6 | v7 |
|---|---|
| `DiscreteTransition` | `using DiscreteTransitionMessagePassingRules` + Project.toml |
| `mutable struct JointMarginalMeta{C}; joint_marginal::C; end` + `where { meta = state_meta }` on an `Exploration` node **and** on a `DiscreteTransition` node | **stop-and-ask ("meta used as mutable workspace")**. The guide says state belongs to an impure algorithm: `mutable struct JointMarginalStore{C} <: DefaultAlgorithmExtension; joint_marginal::C; end` and `MessagePassingRulesBase.ispure(::Type{<:JointMarginalStore}) = false`, given to both nodes with `where { algorithm = store }`. As an extension, DiscreteTransition keeps every packaged rule and adds only the override below. |
| `@marginalrule DiscreteTransition(:out_in) (m_out::Categorical, m_in::Categorical, q_a::PointMass{<:AbstractArray{T,2}}, meta::JointMarginalMeta)` | `@define_marginal_update_rule(node = DiscreteTransition, target = (:out, :in), algorithm = JointMarginalStore, pure = false, args = (m[:out]::Categorical, m[:in]::Categorical, q[:a]::PointMass{<:AbstractMatrix}), body = (algo, args) -> …)` (Marginal rules; `:out_in` → `(:out, :in)`) |
| `@marginalrule DiscreteTransition(:out_in_T1) (… m_T1 …, meta::JointMarginalMeta)` | target `(:out, :in, (:T, 1))`, reading `m[:T][1]` (Groups; v6's `T1` is the member `(:T, 1)`) |
| `@marginalrule DiscreteTransition(:out_in_T1) (…, meta::Any)`, the Tullio contraction | delete: v7's packaged generic marginal rule computes the same joint for any cluster |
| `@call_marginalrule DiscreteTransition(:out_in) (…, meta = nothing)` inside the override | **no usable counterpart.** The guide says "a rule calling another rule → both calling a plain helper function", but DT's joint marginal is an anonymous closure in `lib/DiscreteTransitionMessagePassingRules/src/rules.jl`, with no public helper. Workarounds: `getresult(call_marginal_update_rule(DiscreteTransition, (:out, :in, (:T, 1)); m = …, q = …, algorithm = DefaultAlgorithm()))` inside the body (public, but not sanctioned for use inside a rule), or keep the notebook's own Tullio contraction. See P1. |
| `@node Exploration Stochastic [out, in]`, `@rule Exploration(:out, …) (q_in::Any, meta::JointMarginalMeta)` | `@define_factor_node(node = Exploration, type = Stochastic, interfaces = [:out, :in])`, and the rule with `algorithm = JointMarginalStore`, `args = (q[:in]::Any,)`, `body = (algo, args) -> … algo.joint_marginal …`, `pure = false` (it reads shared mutable state) |
| `@node Ambiguity …`, its rule | the same as Exploration |
| `RxInfer.ReactiveMP.sdtype(::StandaloneDistributionNode) = Stochastic()` | delete. v7's `StandaloneDistribution` (`s_0 ~ p_s_0`, `s[end] ~ goal`) is declared `Stochastic` (`lib/StandardMessagePassingRules/src/nodes/standalone_distribution.jl:28`). The guide covers the node but not this hack (P4). |
| `options = (force_marginal_computation = true,)` | unchanged; RxInfer's branch keeps it |
| `components(joint_marginal)` on a `Contingency` | unchanged (ExponentialFamily). v7's generic marginal returns a `Contingency` for a joint of messages. |

**Stop-and-ask, for the owner:**

1. The side channel works only if `DiscreteTransition`'s joint marginal is written before
   `Exploration`/`Ambiguity` read it in each iteration. In v6 this held by the order streams were
   subscribed, not by construction.
   - v7 subscribes to a target's inputs in declaration order, but the two nodes are independent,
     so the order across them is not guaranteed.
   - Is a one-iteration lag acceptable? At the first iteration the rules read the
     `Contingency(ones(…))` start.
   - If not, the intended information flow has to be modelled: an explicit edge carrying the
     joint, or computing the EFE term inside the DT override.
2. Is the store meant to be shared between the planning step `t` and nothing else? v6 created one
   meta per `t`, and v7 must do the same.

**ReactiveMP changes:** P1 (a sanctioned way to delegate to a packaged rule), P2 (a guide pair for a
state shared by two nodes), P4 (a guide row for the `sdtype` hack).

## 4. T-Maze Active Inference (M)

This is the EFE pattern, with differences:

- `JointMarginalStorage` is the same as `JointMarginalMeta` above.
- There is only one override, `(:out, :in, (:T, 1))`, used by both the transition node
  `DiscreteTransition(previous_location, B, u[t])` and the observation node
  `DiscreteTransition(location[t], reward_observation_tensor, reward_location)`.
- `location_observation ~ DiscreteTransition(current_location, diageye(5))`: `diageye` is
  re-exported by Standard.
- `location[end] ~ DiscreteTransition(reward_location, reward_to_location_mapping)`: a two-edge DT
  with an empty `T` group, which v7 supports.
- The same `sdtype(StandaloneDistributionNode)` line, to delete.
- `force_marginal_computation = true` stays.

**Stop-and-ask:** the same two questions as EFE. Port EFE first and reuse its algorithm and
override verbatim.
**ReactiveMP changes:** P1, P2, P4.

## 5. Autoregressive Active Inference / MARX (L; S–M with P5)

| v6 | v7 |
|---|---|
| `@node MARX Stochastic [out, outprev1, outprev2, in, inprev1, inprev2, Φ]` | `@define_factor_node(node = MARX, type = Stochastic, interfaces = [:out, :outprev1, :outprev2, :in, :inprev1, :inprev2, :Φ])` |
| `struct MARXMeta{T}; Dy::Int; u_lims::Tuple{T,T}; end`, `@meta marx_meta(Dy, u_lims) = MARX() -> MARXMeta(Dy, u_lims)` | `struct MARXPlanner{T} <: AbstractAlgorithm; Dy::Int; u_lims::Tuple{T,T}; end`; `@algorithm function marx_algorithm(Dy, u_lims) MARX() -> MARXPlanner(Dy, u_lims) end`; `infer(…; algorithm = marx_algorithm(Dy, u_lims))`. Parametric, so every rule names `algorithm = MARXPlanner`. |
| 43 × `@rule MARX(:target, Marginalisation) (m_x::…, q_y::…, meta::MARXMeta) = body` | 43 × `@define_message_update_rule(node = MARX, target = :target, algorithm = MARXPlanner, args = (m[:x]::…, q[:y]::…), body = (algo, args) -> body′)`, where `body′` renames `m_x` → `args.m[:x]`, `q_y` → `args.q[:y]`, `meta` → `algo`. Unions of types translate as they are. |
| `@rule MvLocationScaleT(:out, …) (q_ν::PointMass, q_μ::PointMass, q_σ::PointMass)` | the model never uses `MvLocationScaleT` as a node, so this rule is dead. Drop it, or declare a node for it if it is meant to be one (owner question). |
| custom `BayesBase.prod`/`default_prod_rule` for `unBoltzmann`, `MvLocationScaleT`, `MatrixNormalWishart` | unchanged (BayesBase) |
| `Φ ~ MatrixNormalWishart(…)` | Standard has the node (`nodes/matrix_normal_wishart.jl`). Its average energy changed, but the notebook computes no free energy. |
| `q(u_) :: PointMassFormConstraint()`, `limit_stack_depth`, `@initialization`, `@constraints` | unchanged (RxInfer) |

- MARXMeta holds only constants, so this is not a stop-and-ask case.
- The v6 rules mix `m_` and `q_` inputs under mean-field and BP factorisations. v7's default
  dependency scheme (messages in the rule's own cluster, marginals of the other clusters) is v6's,
  and RxInfer's factorisation of data and constants is unchanged, so each rule's inputs should
  arrive as before.
- Rules that never match anything are dead code in both versions. Before porting, run the v6
  notebook with RxInfer's `trace = true` and list the rules actually called. Port those first and
  test the rest separately, or ask the owner.
- Verification: the trajectories. `Random.seed!(3)` plus the global RNG make them deterministic.
  The rule bodies do not change, and every standard rule the graph uses has a point-mass or
  constant covariance, so the port should match v6 bit for bit or nearly.

**Stop-and-ask:** none in the guide's sense. Two owner questions:

1. The dead `MvLocationScaleT` rule.
2. Whether the unreached `MARX` rules should be kept.

**ReactiveMP changes:** P5 (a mechanical translator) would turn this into an afternoon's work.

## 6. Recurrent Switching Linear Dynamical System (L)

**The Gate node**, a hand-written `GateNode`, is rebuilt from scratch. It has no piecewise
counterpart: the guide's Groups section says "the node needs no node type, `factornode` or
`activate!` of its own".

| v6 | v7 |
|---|---|
| `struct Gate{N}`, `GateNode{N} <: AbstractFactorNode`, `as_node_symbol`, `interfaces`, `alias_interface`, `is_predefined_node`, `sdtype`, `collect_factorisation`, `factornode`, `interfaceindex(es)`, `getinboundinterfaces`, `clustername`, `activate!` | `struct Gate end`; `@define_factor_node(node = Gate, type = Deterministic, interfaces = [:out, :switch, :inputs...], algorithm = GateMP, dependencies = [...])` |
| `GateNodeFunctionalDependencies`, `functional_dependencies`, `collect_latest_messages`/`collect_latest_marginals` with `ManyOf` | `struct GateMP <: AbstractAlgorithm end`, declared dependencies: `:out => (m[:inputs...], q[:switch])`, `:switch => (m[:out], m[:inputs...])`, `(:inputs, k) => (m[:out], q[:switch])` (Functional dependencies: a node with its own dependencies → its own algorithm). In a declaration, `q[:switch]` is the variable's marginal (`ReactiveMP.declared_dependencies`), which is what v6 read with `get_stream_of_marginals(getvariable(switch))`, although `switch` sits in the deterministic node's joint cluster. |
| `ReactiveMP.marginalrule(::Type{<:Gate}, ::Val{:switch_inputs}, …)` → `FactorizedJoint((m_inputs..., m_switch))` | `@define_marginal_update_rule(node = Gate, target = members, algorithm = GateMP, args = (m[:switch]::Any, m[:inputs...]::Any), body = (args) -> FactorizedCluster(...))`, with blocks labelled `(:switch,)` and `((:inputs, k),)`. A deterministic node's clusters are `out` and the joint over its inputs, whatever the constraints say. |
| `@rule Gate(:out) (q_switch::Any, m_inputs::ManyOf{N, Any})` | `args = (q[:switch]::Any, m[:inputs...]::Any)`; the body reads `args.m[:inputs]` as a tuple (Groups) |
| `@rule Gate(:switch) (m_out, m_inputs::ManyOf)` with `compute_logscale(prod(GenericProd(), m_out, input), m_out, input)` | the same body; BayesBase's `compute_logscale` is the guide's sanctioned way to take a product's log scale |
| `@rule Gate((:inputs, k)) (m_out, q_switch)` | `target = (:inputs, k)`; the body reads `k` as before |

**Overrides of packaged rules for mixture inputs**, about 15 rules:

| v6 | v7 |
|---|---|
| `@rule typeof(*)(:out) (m_A::PointMass, m_in::MixtureDistribution, meta)` delegating per component via `@call_rule typeof(*)(:out)` | a rule on `*` must name `algorithm = MultiplicationSampling`, the node's default algorithm, not `DefaultAlgorithm`; the per-component delegation → P1 |
| `@rule typeof(dot)(:out / :in1)` with a Mixture, delegating | Standard's `dot` under its default algorithm, plus P1 |
| `@rule typeof(+)(:out / :in1)` with a Mixture, converting to a normal and delegating | P1, or reimplement the Gaussian sum inline, which is a few lines |
| `@rule ContinuousTransition(:W / :y / :a / :x)` and `@marginalrule ContinuousTransition(:y_x)` with a Mixture `q_a` or `m_x`, moment-matching to a normal and delegating | `algorithm = CTVMP` (CT has no default); the joint target is `(:y, :x)`; P1 for the delegation |
| `@rule DiscreteTransition(:out / :in)` and `@marginalrule DiscreteTransition(:out_in)` for `Multinomial` inputs, converting to `Categorical` and delegating | v7's DT rules take `default` inputs and call `discrete_transition_weights`, which has no `Multinomial` method, so a `Multinomial(1, p)` input throws a MethodError. P3 (DT accepting `Multinomial(1, p)`) makes all three overrides unnecessary. Without it, it is not clear whether a user's typed rule for DT under `DefaultAlgorithm` beats the package's `default`-args rule, so they must go under a `DefaultAlgorithmExtension` given to the DT nodes (P7). |
| `prod(::UnspecifiedProd / ::GenericProd, …)` for Mixture, Multinomial and Categorical; `compute_logscale(::Multinomial…)`; `BayesBase.mean/cov/var(::MixtureDistribution)` | unchanged (BayesBase), piracy as before |

**Model:**

| v6 | v7 |
|---|---|
| `where { meta = CTMeta(transformation) }` | `where { algorithm = CTVMP(transformation) }`, with `using ContinuousTransitionMessagePassingRules` |
| `MultinomialPolya(1, u[t]) where { dependencies = RequireMessageFunctionalDependencies(ψ = …) }` | drop `dependencies` (an error in RxInfer v7) and add `μ(u) = convert(promote_variate_type(typeof(η), NormalWeightedMeanPrecision), η, Ψ)` to `@initialization`; `using PolyaMessagePassingRules` (GPL-3; the node reads its weights' message itself) |
| `softdot(ϕ, x[t], w)` | `using SoftDotMessagePassingRules` |
| `DiscreteTransition(s[t], P)` | `using DiscreteTransitionMessagePassingRules` |
| `import ReactiveMP: AbstractFactorNode, …, ManyOf, …` | delete all of it |

**Stop-and-ask:** it hits every case in the guide: raw message tuples (`getdata.(messages[3:end])`),
engine internals, and rules calling rules. Owner questions:

1. Is Gate meant to be belief propagation towards `out` with `q(switch)` (a VMP–BP hybrid), as
   written? The v7 declaration reproduces exactly that; confirm it is intended.
2. The Mixture-to-Gaussian moment matching in the CT and `+` overrides changes the model. Should it
   stay in the notebook, or become a documented approximation (an algorithm) in the CT package?
3. The notebook fixes `StableRNG(42)` for initialisation. Is agreement with v6 up to Polya sampling
   noise good enough? PolyaGamma sampling now draws from the engine's generator, so results change.

**ReactiveMP changes:** P1, P3, P7, and P8 (a guide example of a Gate-like deterministic node with a
group and declared dependencies).

---

## Proposed ReactiveMP changes

| id | change | kind | justification |
|---|---|---|---|
| P1 | **A sanctioned way for a rule to delegate to another packaged rule** with the same context: either document that `getresult(call_message_update_rule(Node, target; m, q, algorithm, ctx))` may be used inside a rule body, forwarding `ctx`, or add a `delegate_*` helper, or make the packaged rules' bodies public helper functions (DT's joint marginal, CT's rules, `*`, `dot`, `+`) | needs ReactiveMP change | The guide's "a rule calling another rule → both calling a plain helper" can't be followed from user code, because the packaged helpers are private closures. EFE, T-Maze and rSLDS have about 18 delegating rules. |
| P2 | Guide pair and worked example for **state shared by two nodes**: an impure `DefaultAlgorithmExtension` with mutable fields, given to both nodes with `where { algorithm = … }`, and the ordering caveat | needs ReactiveMP change (docs) | EFE and T-Maze. The guide only says "belongs to an algorithm that declares itself impure", without showing how two nodes share one algorithm or what ordering holds. |
| P3 | DiscreteTransition accepts `Multinomial(1, p)` as a categorical input (a `discrete_transition_weights` method), and a `Multinomial` result where the inputs were Multinomial | needs ReactiveMP change | Today it is a MethodError deep inside the rule. rSLDS feeds `MultinomialPolya`'s `Multinomial(1, p)` into DT; this would remove three overrides. |
| P4 | Guide row: `sdtype(::StandaloneDistributionNode) = Stochastic()` → delete, since `StandaloneDistribution` is Stochastic in v7 | needs ReactiveMP change (docs) | It appears verbatim in EFE and T-Maze, and the guide covers the node but not the hack. |
| P5 | A mechanical `@rule`/`@marginalrule` → `@define_*_update_rule` translator script (for example under `compat/`, used during the migration and deleted at release like the rest) | optional ReactiveMP change | MARX has 43 rules of pure syntax translation. Doing it by hand risks exactly the errors the guide warns about (swapped `m`/`q`). |
| P6 | Guide pair: `CVI(rng, n, iters, opt)` → `DeltaApproximation(method = CVIProjection(; outsamples, out_prjparams, in_prjparams, sampling_strategy))` + `context = (rng = …,)`; `cvi_setup`/`cvi_update!` have no counterpart | needs ReactiveMP change (docs) | CCVMP and Nonlinear Sensor Fusion. The guide lists CVI as removed but gives no path forward. |
| P7 | Decide and test the **precedence of a user's typed rule against a package's `default`-args rule** for the same node, target and algorithm, and document it | needs ReactiveMP decision + test | rSLDS without P3, and any user who adds a rule for a packaged node. Today the outcome is undocumented and untested: an ambiguity, the default rule winning silently, or the user's rule winning. |
| P8 | Guide example: a Gate/Mixture-style **deterministic node with a group and its own dependencies**, including its joint marginal rule returning a `FactorizedCluster` | needs ReactiveMP change (docs) | rSLDS's `GateNode` is ~150 lines of engine internals with no v7 pair in the guide. The Groups section shows only a sum node. |
