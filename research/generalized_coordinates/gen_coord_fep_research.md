# Path-Based Formalism of the Free Energy Principle: Mathematical Foundations and Connections with Generalized Filtering, Hierarchical Gaussian Filter, and Bayesian Networks

## Abstract

The Free Energy Principle (FEP) represents a unifying mathematical framework for understanding biological self-organization, perception, and action through variational inference. This comprehensive technical review examines the **path-based formalism** of the FEP, exploring its mathematical foundations in path integral theory and its connections to advanced filtering techniques and Bayesian network architectures. We present detailed mathematical formulations linking the FEP to Generalized Filtering, the Hierarchical Gaussian Filter, Bayesian (Kalman) filtering approaches, and message passing algorithms on Bayesian graphs.

## 1. Introduction

The Free Energy Principle, originally proposed by Karl Friston, provides a mathematical framework for understanding how biological systems maintain their organization through variational inference [](#web:22). The **path-based formalism** extends this principle by incorporating trajectory-level dynamics through path integral formulations, offering a more comprehensive description of system evolution over time [](#web:30).[1][2]

This formalism connects several fundamental areas of computational inference:

- **Path integral formulations** for continuous-time dynamics
- **Generalized filtering** for nonlinear state-space models  
- **Hierarchical Bayesian models** for multi-scale inference
- **Message passing algorithms** for distributed computation on graphical models

## 2. Path Integral Formulation of the Free Energy Principle

### 2.1 Mathematical Foundation

The path integral formulation of the FEP expresses the evolution of a system through trajectories in state space. Following the formulation in Da Costa et al. [](#web:30), the **action functional** for a path $$\mathbf{x}[\tau]$$ is defined as:[2]

$$
A[\mathbf{x}[\tau]] = -\ln p(\mathbf{x}[\tau] | \mathbf{x}_0) = \frac{\tau}{2} \ln |(4\pi)^n\Gamma | + \int_0^\tau dt L(\mathbf{x}, \dot{\mathbf{x}})
$$

where the **Lagrangian** takes the form:

$$
L(\mathbf{x}, \dot{\mathbf{x}}) = \frac{1}{2}\left[(\dot{\mathbf{x}} - \mathbf{f}) \cdot \frac{1}{2\Gamma} (\dot{\mathbf{x}} - \mathbf{f}) + \nabla \cdot \mathbf{f}\right]
$$

Here, $$\mathbf{f}$$ represents the flow field and $$\Gamma$$ the noise covariance matrix [](#web:22).[1]

### 2.2 Principle of Least Action

The **most likely path** corresponds to the trajectory that minimizes the action functional:

$$
\mathbf{x}[\tau] = \arg \min_{\mathbf{x}[\tau]} A[\mathbf{x}[\tau]]
$$

This variational principle yields:

$$
\delta_{\mathbf{x}}A[\mathbf{x}[\tau]] = 0 \Leftrightarrow \dot{\mathbf{x}}(\tau) = \mathbf{f}(\mathbf{x})
$$

The path of least action represents **deterministic flow** without random fluctuations, providing the foundation for understanding system dynamics under the FEP [](#web:22).[1]

### 2.3 Markov Blanket Formulation

Systems described by the FEP possess a **particular partition** into:
- **Internal states** ($$\boldsymbol{\mu}$$)
- **External states** ($$\boldsymbol{\eta}$$)  
- **Blanket states** ($$\mathbf{b}$$) comprising:
  - **Sensory states** ($$\mathbf{s}$$)
  - **Active states** ($$\mathbf{a}$$)

The **conditional independence** property of Markov blankets ensures:

$$
p(\boldsymbol{\mu}, \boldsymbol{\eta} | \mathbf{b}) = p(\boldsymbol{\mu} | \mathbf{b})p(\boldsymbol{\eta} | \mathbf{b})
$$

This partition enables the system to be described through **sparse coupling**:

$$
\begin{bmatrix}
\dot{\boldsymbol{\eta}}(\tau) \\
\dot{\mathbf{s}}(\tau) \\
\dot{\mathbf{a}}(\tau) \\
\dot{\boldsymbol{\mu}}(\tau)
\end{bmatrix} = 
\begin{bmatrix}
f_\eta(\boldsymbol{\eta}, \mathbf{s}, \mathbf{a}) + \boldsymbol{\omega}_\eta(\tau) \\
f_s(\boldsymbol{\eta}, \mathbf{s}, \mathbf{a}) + \boldsymbol{\omega}_s(\tau) \\
f_a(\mathbf{s}, \mathbf{a}, \boldsymbol{\mu}) + \boldsymbol{\omega}_a(\tau) \\
f_\mu(\mathbf{s}, \mathbf{a}, \boldsymbol{\mu}) + \boldsymbol{\omega}_\mu(\tau)
\end{bmatrix}
$$

where $$\boldsymbol{\omega}_i(\tau)$$ represent independent noise processes [](#web:22).[1]

## 3. Generalized Filtering

### 3.1 Mathematical Framework

**Generalized Filtering** (GF) provides a Bayesian filtering scheme for nonlinear state-space models in continuous time [](#web:63). The objective is to maximize the **path-integral of free-energy** bound on model log-evidence:[3]

$$
S = \int dt F(t) \geq -\varepsilon
$$

where the **free-energy** at time $$t$$ is:

$$
F(t) = -\ln p(\tilde{\mathbf{s}}(t) | m) + D_{KL}(t)
$$

with $$D_{KL}(t)$$ representing the Kullback-Leibler divergence between the recognition density $$q(\boldsymbol{\vartheta}(t))$$ and true conditional density.

### 3.2 Generalized Coordinates of Motion

GF operates in **generalized coordinates of motion** $$\tilde{\mathbf{u}} = [\mathbf{u}, \mathbf{u}', \mathbf{u}'', \ldots]^T$$, representing instantaneous trajectories. Under the Laplace assumption, the conditional precision becomes an analytic function of the mean:

$$
P(t) = L_{\boldsymbol{\mu}\boldsymbol{\mu}}
$$

where $$L$$ is the energy function. This enables expressing free-energy purely in terms of conditional means:

$$
F = L(\boldsymbol{\mu}) + \frac{1}{2} \ln |L_{\boldsymbol{\mu}\boldsymbol{\mu}}| - \frac{n}{2} \ln 2\pi
$$

### 3.3 Filtering Dynamics

The **recognition dynamics** for states and parameters are governed by:

**For states:**
$$
\dot{\tilde{\boldsymbol{\mu}}}^{(u)} = D\tilde{\boldsymbol{\mu}}^{(u)} - F_u
$$

**For parameters:**
$$
\dot{\boldsymbol{\mu}}^{(\phi)} = \boldsymbol{\mu}'^{(\phi)}, \quad \dot{\boldsymbol{\mu}}'^{(\phi)} = -F_\phi - \kappa\boldsymbol{\mu}'^{(\phi)}
$$

where $$D$$ is the derivative operator and $$\kappa$$ represents prior precision on parameter motion [](#web:63).[3]

### 3.4 Hierarchical Dynamic Models

GF accommodates **hierarchical structures** of the form:

$$
\begin{align}
\mathbf{s} &= f^{(v)}(\mathbf{x}^{(1)}, \mathbf{v}^{(1)}, \boldsymbol{\theta}) + \mathbf{z}^{(1,v)} \\
\dot{\mathbf{x}}^{(1)} &= f^{(x)}(\mathbf{x}^{(1)}, \mathbf{v}^{(1)}, \boldsymbol{\theta}) + \mathbf{z}^{(1,x)} \\
&\vdots \\
\mathbf{v}^{(i-1)} &= f^{(v)}(\mathbf{x}^{(i)}, \mathbf{v}^{(i)}, \boldsymbol{\theta}) + \mathbf{z}^{(i,v)} \\
\dot{\mathbf{x}}^{(i)} &= f^{(x)}(\mathbf{x}^{(i)}, \mathbf{v}^{(i)}, \boldsymbol{\theta}) + \mathbf{z}^{(i,x)}
\end{align}
$$

The energy function decomposes as:

$$
L = \sum_i L^{(i,v)} + \sum_i L^{(i,x)} + L^{(\phi)}
$$

where each term represents prediction errors at different hierarchical levels [](#web:63).[3]

## 4. Hierarchical Gaussian Filter

### 4.1 Model Architecture

The **Hierarchical Gaussian Filter** (HGF) describes learning about uncertain quantities through **coupled Gaussian random walks** [](#web:62). At each level $$i$$, the state evolution follows:[4]

$$
x_i^{(k)} = x_i^{(k-1)} + \varepsilon_i^{(k)}, \quad \varepsilon_i^{(k)} \sim \mathcal{N}(0, f_i(x_{i+1}^{(k-1)}))
$$

The **coupling function** between levels is:

$$
f_i(x_{i+1}) = \exp(\kappa_i x_{i+1} + \omega_i)
$$

where $$\kappa_i$$ represents **coupling strength** and $$\omega_i$$ the **tonic volatility** component.

### 4.2 Variational Update Equations

The HGF derives **closed-form update equations** through variational inversion:

**For levels i > 1:**
$$
\mu_i^{(k)} = \mu_i^{(k-1)} + \frac{1}{2\kappa_{i-1}} \frac{v_{i-1}^{(k)}\pi_{i-1}^{(k)}}{\pi_i^{(k)}} \delta_{i-1}^{(k)}
$$

$$
\pi_i^{(k)} = \hat{\pi}_i^{(k)} + \frac{1}{2}(\kappa_{i-1}v_{i-1}^{(k)}\hat{\pi}_{i-1}^{(k)})^2 \left(1 + \left(1 - \frac{1}{v_{i-1}^{(k)}\pi_{i-1}^{(k-1)}}\right)\delta_{i-1}^{(k)}\right)
$$

**For the first level:**
$$
\mu_1^{(k)} = \hat{\mu}_1^{(k)} + \frac{\hat{\pi}_u}{\pi_1^{(k)}} \delta_u^{(k)}
$$

where:
- $$v_i^{(k)} = t^{(k)} \exp(\kappa_i \mu_{i+1}^{(k-1)} + \omega_i)$$ represents **predicted volatility**
- $$\delta_i^{(k)}$$ are **prediction errors**
- $$\pi_i^{(k)}$$ are **precision estimates** [](#web:62)[4]

### 4.3 Uncertainty Representation

The HGF accommodates two forms of uncertainty:

1. **Informational uncertainty** (estimation uncertainty): $$\sigma_i^{(k)} = 1/\pi_i^{(k)}$$
2. **Environmental uncertainty**: Through time-varying volatilities governed by higher levels

The **learning rate** is dynamically adjusted based on precision ratios:

$$
\alpha_i^{(k)} = \frac{\hat{\pi}_{i-1}^{(k)}}{\pi_i^{(k)}}
$$

This implements **precision-weighted prediction error** updates characteristic of optimal Bayesian learning [](#web:62).[4]

## 5. Bayesian (Kalman) Filtering Connections

### 5.1 Discriminative Kalman Filter

The **Discriminative Kalman Filter** (DKF) modifies standard Kalman filtering by modeling $$p(\text{state}|\text{observation})$$ rather than $$p(\text{observation}|\text{state})$$ [[5]](#web:42). Under the Bernstein-von Mises theorem, this discriminative approach approximates:

$$
p(z_t | x_t) \approx \mathcal{N}(z_t; h(x_t), R_t)
$$

The **DKF update equations** become:

$$
\hat{z}_{t|t-1} = h(\hat{x}_{t|t-1})
$$

$$
\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t^{disc}(z_t - \hat{z}_{t|t-1})
$$

where the **discriminative gain** is:

$$
K_t^{disc} = P_{t|t-1}H_t^T(H_t P_{t|t-1} H_t^T + R_t)^{-1}
$$

### 5.2 Variational Bayesian Kalman Filtering

**Variational Bayesian extensions** of Kalman filtering address unknown noise statistics by treating them as **latent variables**. The joint posterior becomes:

$$
p(\mathbf{x}_{1:T}, \boldsymbol{\Theta} | \mathbf{z}_{1:T}) \approx q(\mathbf{x}_{1:T})q(\boldsymbol{\Theta})
$$

The **variational free energy** objective is:

$$
\mathcal{L} = \mathbb{E}_{q(\mathbf{x}_{1:T})q(\boldsymbol{\Theta})}[\ln p(\mathbf{z}_{1:T}, \mathbf{x}_{1:T}, \boldsymbol{\Theta})] + \mathcal{H}[q(\mathbf{x}_{1:T})] + \mathcal{H}[q(\boldsymbol{\Theta})]
$$

This leads to **coupled update equations** for state estimates and noise parameters [](#web:44).[6]

### 5.3 Connection to Free Energy Principle

The connection between Bayesian filtering and the FEP emerges through the **variational formulation**. Both frameworks minimize bounds on **surprise** (negative log-probability):

$$
F = \mathbb{E}_{q(\mathbf{x})}[\ln q(\mathbf{x}) - \ln p(\mathbf{o}, \mathbf{x})]
$$

In filtering contexts, this becomes the **negative log-marginal likelihood** that Kalman filters implicitly optimize [](#web:41).[7]

## 6. Bayesian Graphs and Message Passing

### 6.1 Belief Propagation Framework

**Message passing algorithms** on Bayesian graphs implement distributed inference through **belief propagation** (BP). The **Bethe free energy approximation** provides a unified framework:

$$
F_{Bethe} = \sum_i \mathbb{E}_{q_i}[\ln q_i(\mathbf{x}_i) - \ln \phi_i(\mathbf{x}_i)] + \sum_{\{i,j\}} \mathbb{E}_{q_{ij}}[\ln \phi_{ij}(\mathbf{x}_i, \mathbf{x}_j) - \ln q_{ij}(\mathbf{x}_i, \mathbf{x}_j)]
$$

subject to **marginalization consistency constraints**:

$$
\int q_{ij}(\mathbf{x}_i, \mathbf{x}_j) d\mathbf{x}_j = q_i(\mathbf{x}_i)
$$

### 6.2 Message Update Rules

**BP message updates** take the form:

$$
m_{i \to j}(\mathbf{x}_j) \leftarrow \int \phi_i(\mathbf{x}_i) \phi_{ij}(\mathbf{x}_i, \mathbf{x}_j) \prod_{k \in \mathcal{N}(i) \setminus j} m_{k \to i}(\mathbf{x}_i) d\mathbf{x}_i
$$

The **beliefs** are computed as:

$$
b_i(\mathbf{x}_i) = \phi_i(\mathbf{x}_i) \prod_{j \in \mathcal{N}(i)} m_{j \to i}(\mathbf{x}_i)
$$

### 6.3 Variational Message Passing

**Variational Message Passing** (VMP) constrains the messages to specific **exponential family** forms [](#web:101). For Gaussian models:[8]

$$
m_{i \to j}(\mathbf{x}_j) = \mathcal{N}(\mathbf{x}_j; \hat{\boldsymbol{\mu}}_{i \to j}, \hat{\boldsymbol{\Sigma}}_{i \to j})
$$

The **natural parameters** $$(\boldsymbol{\eta}_1, \boldsymbol{\eta}_2)$$ are updated through:

$$
\boldsymbol{\eta}_{1,i \to j} = \boldsymbol{\eta}_{1,i} - \sum_{k \neq j} \boldsymbol{\eta}_{1,k \to i}
$$

$$
\boldsymbol{\eta}_{2,i \to j} = \boldsymbol{\eta}_{2,i} - \sum_{k \neq j} \boldsymbol{\eta}_{2,k \to i}
$$

### 6.4 Connection to Free Energy Principle

The FEP can be viewed as implementing **message passing on hierarchical graphs** where:

- **Nodes** represent random variables (states, parameters)
- **Edges** encode probabilistic dependencies  
- **Messages** propagate prediction errors and precisions

The **hierarchical structure** of the FEP corresponds to **deep Bayesian networks** where higher levels modulate the statistics of lower levels [](#web:85).[9]

## 7. Continuous-Time Active Inference

### 7.1 Path Integral Formulation

**Continuous-time active inference** extends the discrete formulation through **path integrals**. The **action functional** for a policy trajectory becomes:

$$
S[\boldsymbol{\pi}] = \int_t^{t+T} \mathcal{L}(\mathbf{s}(\tau), \dot{\mathbf{s}}(\tau), \mathbf{a}(\tau)) d\tau
$$

where the **Lagrangian** incorporates both **epistemic** and **pragmatic** terms:

$$
\mathcal{L} = \mathbb{E}_{q(\mathbf{s})}[\ln q(\mathbf{s}) - \ln p(\mathbf{o}, \mathbf{s})] + \frac{1}{2}\mathbb{E}_{q(\mathbf{s})}[(\dot{\mathbf{s}} - f(\mathbf{s}, \mathbf{a}))^T\boldsymbol{\Gamma}(\dot{\mathbf{s}} - f(\mathbf{s}, \mathbf{a}))]
$$

### 7.2 Generalized Motion

The framework employs **generalized coordinates** $$\tilde{\mathbf{s}} = [\mathbf{s}, \mathbf{s}', \mathbf{s}'', \ldots]^T$$ to represent **instantaneous trajectories**. The **motion dynamics** satisfy:

$$
\dot{\tilde{\boldsymbol{\mu}}} = D\tilde{\boldsymbol{\mu}} - \nabla_{\tilde{\mathbf{s}}} F
$$

where $$D$$ is the **derivative operator** and $$F$$ is the variational free energy [](#web:135).[10]

### 7.3 Action Selection

**Optimal actions** minimize the **expected free energy**:

$$
\mathbf{a}^* = \arg\min_{\mathbf{a}} \int_t^{t+dt} G(\mathbf{s}(\tau), \mathbf{a}) d\tau
$$

where:

$$
G = \underbrace{\mathbb{E}_{q(\mathbf{s})}[D_{KL}[q(\mathbf{s}) || p(\mathbf{s}|\mathbf{o})]]}_{\text{Epistemic value}} + \underbrace{\mathbb{E}_{q(\mathbf{o})}[-\ln p(\mathbf{o})]}_{\text{Pragmatic value}}
$$

This decomposition reveals the **dual nature** of action: exploration for information gain and exploitation for preferred outcomes [](#web:135).[10]

## 8. Mathematical Relationships and Unification

### 8.1 Variational Principles

All the frameworks discussed share a common foundation in **variational principles**:

1. **Path Integral FEP**: Minimizes action functionals over trajectory space
2. **Generalized Filtering**: Minimizes free-energy bounds on log-evidence  
3. **HGF**: Minimizes variational free energy through precision-weighted updates
4. **Bayesian Filtering**: Minimizes posterior uncertainty through optimal estimation
5. **Message Passing**: Minimizes Bethe free energy approximations

### 8.2 Information Geometry

These approaches can be unified through **information geometry** where:

- **States** lie on statistical manifolds
- **Dynamics** follow geodesics of minimum "informational distance"
- **Precision matrices** define Riemannian metrics
- **Free energy gradients** provide natural gradient directions

The **Fisher Information Matrix** emerges as the fundamental metric:

$$
G_{ij} = \mathbb{E}\left[\frac{\partial \ln p}{\partial \theta_i} \frac{\partial \ln p}{\partial \theta_j}\right]
$$

### 8.3 Hierarchy and Scale

**Hierarchical organization** appears across all frameworks:

- **Path integrals**: Multiple temporal scales through generalized coordinates
- **Generalized filtering**: Nested state-spaces with empirical priors  
- **HGF**: Coupled random walks across levels
- **Message passing**: Deep graphical models with layered inference

This suggests that **hierarchy** is a fundamental organizing principle for complex inference systems [](#web:85).[9]

## 9. Computational Implementation

### 9.1 Numerical Methods

**Practical implementation** of these frameworks requires:

1. **Discretization schemes** for continuous-time dynamics
2. **Variational optimization** algorithms (gradient descent, natural gradients)
3. **Message scheduling** strategies for distributed computation
4. **Approximation methods** for intractable integrals

### 9.2 Algorithmic Connections

The mathematical relationships translate to **algorithmic similarities**:

- **Prediction-correction cycles** in filtering correspond to **forward-backward passes** in message passing
- **Precision weighting** in HGF maps to **information-theoretic learning rates**
- **Generalized coordinates** provide **temporal regularization** across frameworks

### 9.3 Software Implementations

Several software packages implement these approaches:

- **SPM12/DEM**: Generalized filtering and hierarchical models
- **HGF Toolbox**: Hierarchical Gaussian Filter implementations  
- **Active Inference packages**: Path integral and continuous-time methods
- **Belief propagation libraries**: Message passing algorithms

## 10. Applications and Extensions

### 10.1 Neuroscience Applications

These frameworks find extensive application in **computational neuroscience**:

- **Neural mass models** through generalized filtering
- **Perceptual learning** via hierarchical Gaussian filtering
- **Motor control** through continuous-time active inference
- **Brain network dynamics** using message passing on neural graphs

### 10.2 Machine Learning Connections

The mathematical foundations connect to modern **machine learning**:

- **Variational autoencoders** implement approximate inference similar to HGF
- **Normalizing flows** provide path integral-like trajectory modeling
- **Graph neural networks** extend message passing to learned representations
- **Continuous normalizing flows** parallel generalized filtering dynamics

### 10.3 Future Directions

**Emerging research directions** include:

1. **Quantum extensions** of path integral formulations
2. **Stochastic partial differential equation** formulations
3. **Multi-agent systems** with distributed Markov blankets
4. **Non-equilibrium steady states** and thermodynamic connections

## 11. Conclusion

The **path-based formalism** of the Free Energy Principle provides a unifying mathematical framework that connects diverse approaches to **probabilistic inference** and **adaptive behavior**. Through path integral formulations, these methods share common foundations in:

1. **Variational principles** for optimal inference
2. **Hierarchical organization** across multiple scales  
3. **Information geometry** for principled dynamics
4. **Precision-weighted learning** for uncertainty quantification

The mathematical connections revealed here suggest that biological intelligence may implement **universal computational principles** that span from microscopic molecular dynamics to macroscopic behavioral patterns. Understanding these connections provides a foundation for developing more sophisticated models of **adaptive systems** and their implementation in **artificial intelligence**.

The technical depth of these mathematical relationships demonstrates that the Free Energy Principle is not merely a conceptual framework, but a **rigorous mathematical theory** with precise computational implementations and measurable predictions. This mathematical rigor provides the foundation for empirical testing and practical applications across neuroscience, machine learning, and autonomous systems.

## References

The references correspond to the web search results identified by their IDs (e.g.,  refers to web:22). Key sources include:[1]

- ****: "The free energy principle made simpler but not too simple" (Physics Reports 2023)[1]
- ****: "Path integrals, particular kinds, and strange things" (arXiv:2210.12761)[2]
- **[41-50]**: IEEE papers on Bayesian and variational Kalman filtering
- **[62-65]**: Hierarchical Gaussian Filter papers (PMC articles)
- ****: Generalized Filtering (Friston et al., Mathematical Problems in Engineering)[3]
- ****: Experimental validation of FEP (Nature Communications)[9]
- ****: Message passing algorithms unified framework[8]
- ****: Active inference continuous time formulation[10]
- **[165-171]**: Markov blankets and Bayesian mechanics papers

This comprehensive technical review demonstrates the mathematical sophistication underlying the Free Energy Principle and its connections to established frameworks in Bayesian inference, filtering theory, and probabilistic machine learning.

[1] https://discovery.ucl.ac.uk/id/eprint/10175520/1/1-s2.0-S037015732300203X-main.pdf
[2] https://arxiv.org/abs/2210.12761
[3] https://www.fil.ion.ucl.ac.uk/~karl/Generalised%20Filtering.pdf
[4] https://pubmed.ncbi.nlm.nih.gov/25477800/
[5] https://direct.mit.edu/neco/article/32/5/969/95592/The-Discriminative-Kalman-Filter-for-Bayesian
[6] https://ieeexplore.ieee.org/document/9611293/
[7] https://ieeexplore.ieee.org/document/9581918/
[8] https://arxiv.org/pdf/1703.10932.pdf
[9] https://www.nature.com/articles/s41467-023-40141-z
[10] https://publish.obsidian.md/active-inference/knowledge_base/cognitive/active_inference
[11] https://link.springer.com/10.1007/s10910-024-01671-z
[12] https://link.aps.org/doi/10.1103/PhysRevA.88.043619
[13] http://link.springer.com/10.1007/BF01362785
[14] http://link.springer.com/10.1007/978-94-011-3190-2_20
[15] https://pubs.aip.org/jcp/article/158/17/174105/2887632/Estimation-of-frequency-factors-for-the
[16] https://link.aps.org/doi/10.1103/PhysRevB.111.L041114
[17] https://philosophymindscience.org/index.php/phimisci/article/view/9187
[18] https://pubs.acs.org/doi/10.1021/acs.jctc.2c00151
[19] https://link.springer.com/10.1007/s11229-023-04292-2
[20] https://linkinghub.elsevier.com/retrieve/pii/S0076687916301094
[21] https://arxiv.org/pdf/2210.12761.pdf
[22] http://link.aps.org/pdf/10.1103/PhysRevD.106.106015
[23] https://arxiv.org/vc/arxiv/papers/2201/2201.06387v2.pdf
[24] https://arxiv.org/pdf/2102.06953.pdf
[25] https://arxiv.org/pdf/2401.08873.pdf
[26] https://pubs.acs.org/doi/10.1021/acs.jctc.4c01717
[27] https://arxiv.org/abs/2108.13343
[28] https://arxiv.org/pdf/2303.15546.pdf
[29] https://pmc.ncbi.nlm.nih.gov/articles/PMC9191674/
[30] https://pmc.ncbi.nlm.nih.gov/articles/PMC12079789/
[31] https://web.mit.edu/dvp/www/Work/8.06/dvp-8.06-paper.pdf
[32] https://publish.obsidian.md/active-inference/knowledge_base/mathematics/path_integral_free_energy
[33] https://en.wikipedia.org/wiki/Path_integral_formulation
[34] https://www.dialecticalsystems.eu/contributions/the-free-energy-principle-a-precis/
[35] https://publish.obsidian.md/active-inference/knowledge_base/mathematics/variational_free_energy
[36] https://phoebe.pubpub.org/pub/pf2xw4p6
[37] https://arxiv.org/abs/2201.06387
[38] https://pubs.aip.org/aip/jcp/article/102/10/4151/482190/Variational-upper-and-lower-bounds-on-quantum-free
[39] https://pmc.ncbi.nlm.nih.gov/articles/PMC5857288/
[40] https://pmc.ncbi.nlm.nih.gov/articles/PMC3780612/
[41] https://en.wikipedia.org/wiki/Free_energy_principle
[42] https://arxiv.org/abs/2205.07793
[43] https://link.aps.org/doi/10.1103/PhysRevLett.131.126501
[44] https://www.sciencedirect.com/science/article/pii/S1571064523001094
[45] https://direct.mit.edu/neco/article/36/5/963/119791/An-Overview-of-the-Free-Energy-Principle-and
[46] https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2020.549187/full
[47] https://www.sciencedirect.com/science/article/pii/S037015732300203X
[48] https://jaredtumiel.github.io/blog/2020/08/08/free-energy1.html
[49] https://link.springer.com/10.1007/s00034-023-02436-w
[50] https://www.semanticscholar.org/paper/bb95cf0ab4482a302549f5eb2bad5d14584ef8a4
[51] https://ieeexplore.ieee.org/document/9747986/
[52] https://link.springer.com/10.1007/s12555-021-0467-4
[53] https://www.mdpi.com/1099-4300/24/1/117
[54] https://ieeexplore.ieee.org/document/10078861/
[55] https://ieeexplore.ieee.org/document/9926172/
[56] https://arxiv.org/html/2405.05646v2
[57] http://arxiv.org/pdf/1611.09293.pdf
[58] https://arxiv.org/pdf/1712.01406.pdf
[59] https://pmc.ncbi.nlm.nih.gov/articles/PMC9634992/
[60] https://arxiv.org/pdf/1912.08601.pdf
[61] http://arxiv.org/pdf/1302.0681.pdf
[62] https://www.mdpi.com/1099-4300/19/12/655/pdf?version=1512141144
[63] https://arxiv.org/html/2410.15832v3
[64] http://arxiv.org/pdf/2404.00481.pdf
[65] https://arxiv.org/pdf/1704.06988.pdf
[66] https://www.bme.jhu.edu/ascharles/wp-content/uploads/2020/01/KalmanFilterBayesDerivation.pdf
[67] https://www.jamesstephen.in/papers/8.pdf
[68] https://pmc.ncbi.nlm.nih.gov/articles/PMC4237059/
[69] https://en.wikipedia.org/wiki/Generalized_filtering
[70] https://en.wikipedia.org/wiki/Kalman_filter
[71] https://www.tnu.ethz.ch/fileadmin/user_upload/documents/Publications/2021/2021_Senoz_The_Switching_Hierarchical_Gaussian_Filter.pdf
[72] https://pubs.aip.org/aip/jcp/article/161/14/144106/3316030/Path-filtering-in-path-integral-simulations-of
[73] https://pmc.ncbi.nlm.nih.gov/articles/PMC8259355/
[74] https://biaslab.github.io/pdf/isit2021/shgf.pdf
[75] https://link.aps.org/doi/10.1103/PhysRevB.21.4251
[76] https://people.bordeaux.inria.fr/pierre.delmoral/chen_bayesian.pdf
[77] https://biaslab.github.io/publication/switching-hgf/
[78] https://www.sciencedirect.com/science/article/abs/pii/S0021999117300839
[79] https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
[80] https://examples.rxinfer.com/categories/problem_specific/hierarchical_gaussian_filter/
[81] https://link.aps.org/doi/10.1103/PhysRevA.41.644
[82] https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf
[83] https://www.sciencedirect.com/science/article/abs/pii/S1053811915004747
[84] https://www.mdpi.com/1099-4300/26/11/984
[85] https://arxiv.org/abs/2505.22749
[86] https://www.semanticscholar.org/paper/a874d92dcc0112d75b49824d7b80bdfa7698069b
[87] https://arxiv.org/abs/2310.02946
[88] http://link.springer.com/10.1007/s00354-012-0103-1
[89] http://link.springer.com/10.1007/978-3-642-04180-8_57
[90] https://arxiv.org/abs/2503.13223
[91] https://arxiv.org/abs/2502.12654
[92] http://arxiv.org/pdf/2205.11543.pdf
[93] https://royalsocietypublishing.org/doi/10.1098/rsfs.2022.0029
[94] http://arxiv.org/pdf/1304.1095.pdf
[95] https://pmc.ncbi.nlm.nih.gov/articles/PMC10198254/
[96] http://arxiv.org/pdf/2410.02972.pdf
[97] https://arxiv.org/pdf/2008.09927.pdf
[98] https://pmc.ncbi.nlm.nih.gov/articles/PMC11089901/
[99] http://arxiv.org/pdf/2502.12654.pdf
[100] https://www.frontiersin.org/articles/10.3389/fnhum.2013.00598/pdf
[101] https://pmc.ncbi.nlm.nih.gov/articles/PMC8902446/
[102] https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0067022
[103] https://www.spsc.tugraz.at/phd-theses/phd_leisenberger.html
[104] https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20-%20a%20rough%20guide%20to%20the%20brain.pdf
[105] https://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter8.pdf
[106] https://arxiv.org/pdf/1401.3877.pdf
[107] https://pmc.ncbi.nlm.nih.gov/articles/PMC7714236/
[108] https://proceedings.neurips.cc/paper/2020/file/be53d253d6bc3258a8160556dda3e9b2-Paper.pdf
[109] https://en.wikipedia.org/wiki/Belief_propagation
[110] https://academic.oup.com/nsr/article/11/5/nwae025/7571549
[111] https://arxiv.org/html/2407.11231v1
[112] https://cba.mit.edu/events/03.11.ASE/docs/Yedidia.pdf
[113] https://link.aps.org/doi/10.1103/PhysRevResearch.7.013220
[114] https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2019.00020/full
[115] https://oecs.mit.edu/pub/my8vpqih
[116] https://www.numberanalytics.com/blog/deep-dive-path-integral-formulation
[117] https://www.semanticscholar.org/paper/f2988fe5c853191c0d68820f5133b9b21e68c2a4
[118] https://epubs.siam.org/doi/10.1137/24M1653306
[119] https://ieeexplore.ieee.org/document/10595175/
[120] https://link.aps.org/doi/10.1103/PhysRevD.110.024079
[121] https://link.aps.org/doi/10.1103/PhysRevE.109.014210
[122] https://www.ssrn.com/abstract=2836961
[123] https://link.springer.com/10.1134/S2635167621020142
[124] https://www.semanticscholar.org/paper/fb55578121c91dddeae29ef65fe637a599a5850f
[125] https://ieeexplore.ieee.org/document/8868359/
[126] https://projecteuclid.org/journals/annales-de-linstitut-henri-poincare-probabilites-et-statistiques/volume-58/issue-2/Path-dependent-FeynmanKac-formula-for-forward-backward-stochastic-Volterra-integral/10.1214/21-AIHP1158.full
[127] https://arxiv.org/pdf/2011.11476.pdf
[128] http://arxiv.org/pdf/1310.1824.pdf
[129] http://arxiv.org/pdf/2109.05282.pdf
[130] http://arxiv.org/pdf/2112.12958.pdf
[131] https://arxiv.org/pdf/1209.0623.pdf
[132] https://arxiv.org/pdf/1609.02849.pdf
[133] http://arxiv.org/pdf/1503.01667.pdf
[134] https://pmc.ncbi.nlm.nih.gov/articles/PMC4385267/
[135] http://arxiv.org/pdf/1909.12990.pdf