### VisualizationMethods

Utilities to scan this repository for GraphPPL / RxInfer `@model` definitions and generate:
- Mermaid diagrams (`.mmd`)
- Graphviz DOT (`.dot`)
- PNG renders (if Graphviz `dot` is available)

Outputs are saved under `research/visualization_methods/outputs/<file>__<model>/`.

#### Quick start

- Ensure Julia 1.9+ and optionally Graphviz installed (`dot` on PATH).
- Run:

```bash
julia research/visualization_methods/bin/visualize_repo_models.jl
```

#### API

- `VisualizationMethods.scan_models()` → list `@model` definitions
- `VisualizationMethods.model_to_mermaid(file, name)` → Mermaid text
- `VisualizationMethods.model_to_dot(file, name)` → DOT text
- `VisualizationMethods.render_graph_assets(file, name)` → writes `.mmd`, `.dot`, and optional `.png`
- `VisualizationMethods.visualize_repo_models()` → end-to-end

This uses a lightweight heuristic parser to extract stochastic statements (`x ~ Dist(args...)`).
