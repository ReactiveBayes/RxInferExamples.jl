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
make example-dev FILTER=LinearRegression RXINFER=/path/to/RxInfer.jl
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

The notebook processing system:
- Converts `.ipynb` files to `.md` using Weave.jl
- Preserves each notebook's environment through Project.toml
- Optionally integrates development version of RxInfer.jl
- Generates figures in the same directory as the notebook
- Fixes absolute paths to use relative paths
- Adds contribution notes automatically

For auxiliary file handling, the system copies all supporting files like data files while excluding Manifest.toml files. The original directory structure is maintained throughout this process.

The error handling system checks for error blocks in the output, reports any failed conversions, and provides detailed context when errors occur to help with debugging.

### Parallel Processing

The build system leverages Julia's distributed computing capabilities to process multiple notebooks simultaneously. It distributes the workload across available CPU cores. After processing completes, it generates a detailed report showing how many notebooks were processed successfully and which ones failed, if any.

## Stage 2: Documentation Building (`docs/make.jl`)

This script builds the final documentation and performs several key functions:

First, it collects metadata from all examples by reading their meta.jl files. This includes gathering titles, descriptions, and tags for each example, and organizing them into appropriate categories.

Next, it generates the pages needed for the documentation site. This involves creating a comprehensive list of all examples, setting up the navigation structure between pages, and applying consistent HTML styling to the examples list.

Finally, it handles the actual documentation building process using Documenter.jl. This includes deploying the built documentation to GitHub Pages and ensuring clean builds by removing old artifacts when needed.