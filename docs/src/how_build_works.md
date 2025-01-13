# How the Build System Works

This document explains the build system for RxInfer.jl Examples. The build process involves two main scripts:
`examples/make.jl` and `docs/make.jl`, each serving a different purpose.

!!! tip "Quick Help"
    Run `make help` to see all available build commands and their descriptions:
    ```bash
    make help
    ```

## Overview

The build process happens in two stages:
1. Converting notebooks to markdown (`examples/make.jl`)
2. Building the documentation (`docs/make.jl`)

## Stage 1: Notebook Processing (`examples/make.jl`)

This script handles the conversion of Jupyter notebooks to markdown files.

The notebook processing system converts `.ipynb` files to `.md` using Weave.jl, while preserving the notebook's environment through Project.toml. During conversion, it generates figures in the same directory as the notebook and fixes absolute paths to use relative paths instead. Each example also gets contribution notes added automatically.

For auxiliary file handling, the system copies all supporting files like data files while excluding Manifest.toml files. The original directory structure is maintained throughout this process.

The error handling system checks for error blocks in the output, reports any failed conversions, and provides detailed context when errors occur to help with debugging.

### Parallel Processing

The build system leverages Julia's distributed computing capabilities to process multiple notebooks simultaneously. It distributes the workload across available CPU cores. After processing completes, it generates a detailed report showing how many notebooks were processed successfully and which ones failed, if any.

## Stage 2: Documentation Building (`docs/make.jl`)

This script builds the final documentation and performs several key functions:

First, it collects metadata from all examples by reading their meta.jl files. This includes gathering titles, descriptions, and tags for each example, and organizing them into appropriate categories.

Next, it generates the pages needed for the documentation site. This involves creating a comprehensive list of all examples, setting up the navigation structure between pages, and applying consistent HTML styling to the examples list.

Finally, it handles the actual documentation building process using Documenter.jl. This includes deploying the built documentation to GitHub Pages and ensuring clean builds by removing old artifacts when needed.