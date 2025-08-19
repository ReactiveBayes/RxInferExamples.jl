### Testing: layout and runner

How to run all tests end-to-end:

```bash
GKSwstype=100 julia --project=. tests/runtests.jl
```

What is covered:
- `unit_environment.jl`: RNG-seeded environment constructs and produces observations/history.
- `unit_streams.jl`: static and timer stream helpers compile.
- `unit_model_updates.jl`: `@model`, constraints, autoupdates, initialization; smoke infers on a tiny stream and produces free-energy.
- `unit_visualize.jl`: plot builders and animations construct.
- `unit_meta_project_docs.jl`: presence of `Project.toml`, `Manifest.toml`, `meta.jl`, `README.md`, and `docs/`.
- Integration test in `runtests.jl`: runs `run.jl`, validates artifacts in `static/`, `realtime/`, `comparison/`, and asserts numerical equivalence of static vs realtime estimates.

Notes:
- GIFs are optional in CI. Enable with `IDS_MAKE_GIF=1` to render animation artifacts (`static_free_energy.gif`, `static_composed_estimates_fe.gif`, `realtime_inference.gif`, `realtime_free_energy.gif`, `comparison/overlay_means.gif`).
- Use `IDS_SEED` to fix the RNG for reproducible environment generation in both static and realtime runs.
- Realtime free-energy is written from either a live FE stream (if exposed) or from strict online per-prefix computation controlled by `IDS_RT_FE_EVERY` (fallback: offline per-prefix after the run). Static FE is always computed per-prefix offline.

