# Active Inference Theory and Implementation

Theoretical foundations and practical implementation of Active Inference agents.

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [Mathematical Formulation](#mathematical-formulation)
3. [Computational Implementation](#computational-implementation)
4. [Practical Considerations](#practical-considerations)

## Theoretical Background

### The Free Energy Principle

The Free Energy Principle (FEP) is a unifying theory proposing that biological systems minimize variational free energy to maintain their existence.

**Key Insight**: To survive, organisms must:
1. Maintain homeostasis (stay in viable states)
2. Predict sensory inputs
3. Minimize surprise (unexpected observations)

**Free Energy**: An upper bound on surprise (negative log evidence):
```
F = ⟨log q(s) - log p(x, s)⟩_q
  ≥ -log p(x)  (surprise)
```

Where:
- `q(s)`: Approximate posterior (agent's beliefs)
- `p(x, s)`: Joint distribution (generative model)
- `p(x)`: Model evidence (how likely observations are)

### Active Inference

Active Inference extends FEP to action selection:

**Principle**: Agents minimize expected free energy by:
1. **Perception**: Updating beliefs to fit observations
2. **Action**: Selecting actions that bring expected observations closer to preferred states

**Expected Free Energy** (for future time):
```
G = E_q[log q(s) - log p(x, s) - log p(x)]
```

Decomposition:
- **Pragmatic value**: Preference satisfaction
- **Epistemic value**: Information gain
- **Novelty**: Exploration bonus

### Comparison to Other Frameworks

| Framework | Objective | Planning |
|-----------|-----------|----------|
| Reinforcement Learning | Maximize reward | Value iteration |
| Optimal Control | Minimize cost | Bellman equation |
| Active Inference | Minimize free energy | Variational inference |

**Key Difference**: Active Inference agents don't maximize rewards—they minimize surprise by making preferred states unsurprising.

## Mathematical Formulation

### Generative Model

The agent's model of how the world works:

```
p(x_{1:T}, s_{1:T}, u_{1:T}) = 
    p(s_0) ∏_{t=1}^T p(s_t | s_{t-1}, u_t) p(x_t | s_t)
```

Components:
- **States**: Hidden variables `s_t`
- **Observations**: Sensory data `x_t`
- **Actions**: Control variables `u_t`

**Transition model**:
```
p(s_t | s_{t-1}, u_t) = N(s_t | g(s_{t-1}) + h(u_t), Γ^{-1})
```

**Observation model**:
```
p(x_t | s_t) = N(x_t | s_t, Θ^{-1})
```

**Goal prior** (preferences):
```
p(x_t) = N(x_t | goal, Σ^{-1})
```

### Variational Inference

Agent maintains approximate posterior:
```
q(s_{1:T}, u_{1:T}) = ∏_{t=1}^T q(s_t) q(u_t)
```

Mean-field approximation factorizes over time and variables.

**Inference objective**: Minimize KL divergence
```
KL[q(s, u) || p(s, u | x)] = F + const
```

### Message Passing

Variational message passing updates beliefs through local computations:

**Forward messages** (predictions):
```
μ_{s_t→x_t} = E_q[s_t]
```

**Backward messages** (goal information):
```
μ_{x_t→s_t} = Θ (x_t - goal)
```

**Update equations**:
```
∇_q F = 0  ⟹  Belief updates
```

Implemented efficiently in RxInfer using automatic differentiation and conjugate-exponential families.

### Planning as Inference

Action selection reduces to inference:
```
p(u_{1:T} | goal) ∝ exp(-G[u_{1:T}])
```

The most likely actions are those minimizing expected free energy.

**Practical implementation**:
1. Fix observations at goals: `x_t^{goal}`
2. Perform inference over states and actions
3. Extract posterior mean actions: `E_q[u_t]`

## Computational Implementation

### Algorithm Overview

```
1. Initialize beliefs q(s_0), q(u_{1:T})
2. For iteration = 1 to max_iterations:
     a. Forward pass: Update state beliefs
     b. Backward pass: Incorporate goal information  
     c. Update action beliefs
3. Extract: Posterior means and covariances
4. Select: First action u_1
```

### Message Schedule

Efficient message passing order:

```
# Initialization
s_0 ← prior

# Forward sweep (t = 1 to T)
for t in 1:T
    u_t ← control_prior
    s_t ← predict(s_{t-1}, u_t)
    x_t ← observe(s_t)
end

# Backward sweep (t = T to 1)
for t in T:-1:1
    x_t ← goal_constraint
    s_t ← incorporate(x_t)
    u_t ← update_action(s_{t-1}, s_t)
end
```

### Precision Matrices

Critical parameters controlling belief strength:

**Transition Precision (Γ)**:
- High: Trust model dynamics
- Low: More exploratory, adapts to deviations

**Observation Precision (Θ)**:
- High: Trust sensory data
- Low: More robust to noise

**Goal Prior Precision (Σ)**:
- High: Strongly enforce goals
- Low: Satisficing behavior, allows suboptimality

**Relationships**:
```
Γ >> Θ: Model-driven (trust dynamics)
Θ >> Γ: Data-driven (trust observations)
Σ >> Γ, Θ: Goal-driven (strong preferences)
```

### Linearization

For nonlinear dynamics, use local linearization:

**First-order approximation**:
```
g(s) ≈ g(μ_s) + J_g(μ_s) (s - μ_s)
```

Where `J_g` is the Jacobian at mean `μ_s`.

RxInfer uses automatic differentiation via `DeltaMeta` for this.

### Sliding Horizon

Maintain fixed computational cost with receding horizon:

```
Window at step t: [t, t+1, ..., t+T]
```

After each action:
1. Slide window forward
2. Drop past belief
3. Add new future belief
4. Reinitialize action priors

## Practical Considerations

### Hyperparameter Tuning

**Planning Horizon (T)**:
- Start with T = 10-20
- Increase if behavior myopic
- Decrease if slow inference

**Precision Values**:
- Start with balanced: Γ = Θ = 1e4, Σ = 1e4
- Increase Γ if model accurate
- Increase Θ if sensors reliable
- Increase Σ for faster goal achievement

**Inference Iterations**:
- Start with 10 iterations
- Increase if not converging (check free energy)
- Decrease if fast inference needed

### Computational Complexity

**Time complexity**: O(T · d³ · iterations)
- T: Horizon length
- d: State dimensionality
- iterations: Variational iterations

**Space complexity**: O(T · d²)
- Store means and covariances for T time steps

**Scaling strategies**:
1. Reduce horizon
2. Use sparse representations
3. Factorize state space
4. Parallel computation

### Convergence Diagnostics

Monitor variational free energy:

```julia
if free_energy_reduction < tolerance
    convergence = true
end
```

**Warning signs**:
- Free energy increasing
- Oscillating beliefs
- Numerical instability (NaN/Inf)

### Common Issues

**Problem**: Agent doesn't move
- **Cause**: Weak goal prior precision
- **Fix**: Increase Σ

**Problem**: Unstable behavior
- **Cause**: Too high precision
- **Fix**: Decrease Γ, Θ, Σ

**Problem**: Slow convergence
- **Cause**: Too many iterations or long horizon
- **Fix**: Reduce iterations or horizon

**Problem**: Ignores observations
- **Cause**: Low observation precision
- **Fix**: Increase Θ

## Advanced Topics

### Hierarchical Active Inference

Multiple levels of abstraction:
```
Level 2: Plans goals for Level 1
Level 1: Plans actions to reach goals
```

### Meta-Learning

Learning precision matrices:
```
Γ_new = Γ_old + α ∇_Γ log p(x | model)
```

### Multi-Modal Inference

Handle discrete and continuous variables:
```
q(s, z) = q(s) q(z)
```
Where z are discrete modes.

### Model Selection

Compare alternative generative models:
```
p(model | data) ∝ p(data | model) p(model)
```

Use free energy as model evidence approximation.

## References

### Key Papers

1. Friston, K. (2010). "The free-energy principle: a unified brain theory?"
2. Friston, K. et al. (2017). "Active inference: a process theory"
3. Parr, T. & Friston, K. (2019). "Generalised free energy and active inference"

### Implementations

1. RxInfer.jl - Reactive message passing in Julia
2. SPM/DEM - MATLAB toolbox
3. pymdp - Python active inference

### Related Work

- Predictive coding
- Variational autoencoders
- Model-based reinforcement learning
- Optimal control theory

---

**Theory guides practice - deep understanding enables powerful implementations.**

