# How the Build System Works

This document explains the build system for RxInfer.jl Examples. The build process involves two main scripts:
`examples/make.jl` and `docs/make.jl`, each serving a different purpose.

!!! tip "Quick Help"
    Run `make help` to see all available build commands and their descriptions:
    ```bash
    make help
    ```

## Development Options

The build system supports using either the released version of RxInfer.jl or a local development version:

```bash
# Build with released version (default)
make examples

# Build with local development version (expects RxInfer.jl next to RxInferExamples.jl)
make examples-dev

# Build with specific RxInfer.jl path
make examples-dev RXINFER=/path/to/RxInfer.jl

# Build single example with development version
make examples-dev FILTER=LinearRegression RXINFER=/path/to/RxInfer.jl
```

When using the development version (`--use-dev`), the build system will:
1. Look for RxInfer.jl in the specified location
2. Add it as a development dependency to each notebook's environment
3. Ensure all notebooks use the same RxInfer version

## Overview

The build process happens in two stages:
1. Converting notebooks to markdown (`examples/make.jl`)
2. Building the documentation (`docs/make.jl`)

## Stage 1: Notebook Processing (`examples/make.jl`)

This script handles the conversion of Jupyter notebooks to markdown files.
At the beginning of the execution the script scans the examples and their 
respective `Project.toml` files and records all the depencies of all the examples 
in one big temporary environment. This environment that is being used to run 
each individual example. This have several consequences that is good to be aware of:
- Running examples in bulk always resolves to the same versions of packages for ALL examples
- Running examples in bulk reuses cached and compiled code across all dependencies, that speeds up the build process
- Running examples individually and in bulk can resolve to difference versions of packages, in case of some conflicts, Julia usually decides to use the older version of the respective packages. That means that running examples via Jupyter notebook may use different versions since Jupyter notebook resolves the dependencies locally.

Optionally, it is possible to start the build process with the development version of RxInfer.jl.
Look at `make help` or script arguments to understand how to enable this option.

After creation of the big temporary environment (temporary because it will be deleted as soon as the build process finishes) the script 
proceeds with building each individual notebook. The notebook processing system:
- Converts `.ipynb` files to `.md` using Weave.jl
- Uses separate `julia` processes for each individual notebook
- Uses the shared big temporary environment
- Generates figures in the same directory as the notebook
- Fixes absolute paths to use relative paths
- Adds contribution notes automatically

!!! warning "Self-Contained Examples"
    Examples must be self-contained and cannot use `include()` statements. All code must be directly in the notebook cells to ensure:
    - Examples are reproducible by copying and pasting
    - The build system can properly process all code
    - Documentation remains consistent across different environments

For auxiliary file handling, the system copies all supporting files like data files while excluding Manifest.toml files. The original directory structure is maintained throughout this process.

The error handling system checks for error blocks in the output, reports any failed conversions, and provides detailed context when errors occur to help with debugging.

### Parallel Processing

The build system leverages Julia's distributed computing capabilities to process multiple notebooks simultaneously. It distributes the workload across available CPU cores. After processing completes, it generates a detailed report showing how many notebooks were processed successfully and which ones failed, if any, together with a performance report listing the total wall-clock time, the total CPU time summed over all notebooks, and the slowest notebooks to build.

Notebooks are dispatched in directory order by default. Because the build finishes only
when its slowest notebook finishes, a long-running notebook that happens to start last
sets the total wall-clock time on its own. An example can opt out of that by setting
`build_priority = true` in its `meta.jl`, which schedules it ahead of the others - see
[Metadata Requirements](@ref) in the contributing guide. This is a scheduling hint only:
it changes when a notebook starts, never its output or the total CPU time of the build.

## Stage 2: Documentation Building (`docs/make.jl`)

This script builds the final documentation and performs several key functions:

First, it collects metadata from all examples by reading their meta.jl files. This includes gathering titles, descriptions, and tags for each example, and organizing them into appropriate categories.

Next, it generates the pages needed for the documentation site. This involves creating a comprehensive list of all examples, setting up the navigation structure between pages, and applying consistent HTML styling to the examples list.

Finally, it handles the actual documentation building process using Documenter.jl. This includes deploying the built documentation to GitHub Pages and ensuring clean builds by removing old artifacts when needed.

### Syntax highlighting

Documenter is configured with `prerender = true`. Instead of shipping code blocks as plain
text and letting `highlight.js` colour them in the reader's browser, Documenter shells out to
`node` for every code block during the build and writes the highlighted markup directly into
the HTML. This makes the highlighting independent of anything that happens at page load - CDN
availability, ad blockers, or a JavaScript error elsewhere on the page can no longer leave the
examples unhighlighted.

!!! warning "`node` is required"
    Pre-rendering needs both `node` and `curl` on the `PATH` (`curl` is used to fetch the
    `highlight.js` bundle at build time). `docs/make.jl` checks for `node` before calling
    `makedocs`:

    - locally, a missing `node` only produces a warning and the build falls back to
      client-side highlighting,
    - in CI (`CI=true`) it is a hard error, so a runner without Node cannot silently deploy
      unhighlighted pages.

    Note that only Julia *input* cells are highlighted. Weave emits the output of each cell as
    an untagged code fence, which Documenter renders as `nohighlight` - that is intentional,
    those blocks are program output rather than source code.

### Deployment

The last thing `docs/make.jl` does in CI is call `deploydocs`, which pushes `docs/build` to the
`gh-pages` branch. The site is deployed *unversioned*, straight at the domain root
(`versions = nothing`), which is why the build also strips Documenter's `versions.js` script tag -
it points above the site root and would 404 on every page load.

!!! warning "A missing token makes the deploy silent, not loud"
    `deploydocs` reads `DOCUMENTER_KEY` or `GITHUB_TOKEN` from the environment. When neither is
    set it decides this is not a deploy build, logs `Deploying: ✘`, and **returns normally**. The
    job goes green and the live site keeps serving whatever it served before.

    `docs/make.jl` refuses to finish such a build: on `refs/heads/main` or a tag, a missing token
    is a hard error. If you touch the `docs` job in `.github/workflows/CI.yml`, keep the
    credentials on the same step that runs `make docs` — not on a step before or after it.
