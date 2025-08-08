## HGF (Hierarchical Gaussian Filter)

### Run
- **Tests**: `julia --project=research/hgf -e "using Pkg; Pkg.instantiate(); Pkg.test()"`
- **Example**: `julia --project=research/hgf research/hgf/run_hgf.jl`
- **Outputs**: saved under `results/<timestamp>/HGF/` (PNGs, GIFs, MP4s, JSON report, summary.txt).

### Modules
- **Utils.jl**: parameters and synthetic data generation
- **Model.jl**: RxInfer models and constraints/meta
- **Viz.jl**: plotting helpers and animations
- **Run.jl**: filtering and smoothing runners
- **HGF.jl**: convenience re-exports

### Notes
- Headless plotting is handled in `run_hgf.jl` via `ENV["GKSwstype"] = "100"`.
- MP4 generation uses Julia's `FFMPEG` artifact; if system ffmpeg is missing, MP4 is skipped with a warning.
- See `TECHNICAL_README.md` for model equations and variational setup.
