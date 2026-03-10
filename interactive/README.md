# Reactive inference with RxInfer.jl

The `interactive` folder contains programs where a system is controlled and a user can change operating conditions through sliders. They demonstrate that RxInfer adapts online to the resulting prediction errors.

## Installation instructions

To start the interactive demo, follow these steps:

0. Go to the [RxInfer.jl](https://github.com/biaslab/RxInfer.jl) and click the `Star` button, otherwise the demo will not work.
1. You need the Julia programming language preinstalled. We recommend using the [`juliaup`](https://github.com/JuliaLang/juliaup) Julia version manager to easliy install multiple versions of Julia.
2. Start `julia` and run `] add IJulia`. Note the `]` symbol, its important, it changes the julia mode from execution to package manager mode. After that you may need to run `build` in the package manager mode. Alternatively it is possible to do the same in the execution mode by running `using Pkg; Pkg.add("IJulia"); Pkg.build()`. These commands install Julia jupyter kernel. After you can use either standard `jupyter notebook` command or simply run `using IJulia; notebook(dir = pwd())` in the project's directory.
3. The repository comes with the preconfigured environment written in the `Project.toml` and `Manifest.toml`.
The notebook instantiates and installs all required packages automatically. Note, however, that initial installation and precompilation may take a significant amount of time.
4. Simply click `Cells` -> `Run all` and wait for the instantion of the example. Julia may take several minutes to precompile the code.
