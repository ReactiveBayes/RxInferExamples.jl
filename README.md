# A collection of examples for RxInfer.jl

![](docs/src/assets/biglogo-blacktheme.svg?raw=true&sanitize=true)

[![Official page](https://img.shields.io/badge/official%20page%20-RxInfer-blue)](https://rxinfer.ml)
[![Examples](https://img.shields.io/badge/examples-RxInfer-brightgreen)](https://examples.rxinfer.ml)
[![Contribute a new example](https://img.shields.io/badge/Contribute-%20a%20new%20example-red)](https://examples.rxinfer.ml/how_to_contribute/)
[![Q&A](https://img.shields.io/badge/Q&A-RxInfer-orange)](https://github.com/reactivebayes/RxInfer.jl/discussions)

This repository contains a collection of examples for [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl), a Julia package for reactive message passing and probabilistic programming.

Navigate to the [Examples](https://examples.rxinfer.ml) page to check the pre-rendered examples or clone the repository and run the examples locally. Additionally, explore the official [RxInfer.jl](https://docs.rxinfer.ml) documentation.

## How to run the examples locally

1. Clone the repository:
   ```bash
   git clone https://github.com/ReactiveBayes/RxInferExamples.jl.git
   ```

2. Install required global dependencies:
   ```bash
   julia -e 'using Pkg; Pkg.add("Weave")'
   ```
   
   > **Note**
   > Building examples requires the Weave.jl package to be installed globally.

3. Build the examples:
   ```bash
   make examples
   ```

4. Build and preview the documentation:
   ```bash
   make docs
   make preview
   ```

> [!NOTE]  
> Example builds are cached. If you make changes to an example and still see old errors after rebuilding, try clearing the cache first with the `make clean` command.

All the examples are Jupyter notebooks, which also can be run with [Jupyter](https://jupyter.org/).

## Contributing

We welcome contributions! Please check our [contribution guide](https://examples.rxinfer.ml/how_to_contribute/) for guidelines.

## Resources

- [How to Contribute](https://examples.rxinfer.ml/how_to_contribute/)
- [RxInfer.jl Documentation](https://docs.rxinfer.ml)
- [RxInfer.jl Repository](https://github.com/ReactiveBayes/RxInfer.jl)
- [Examples Documentation](https://examples.rxinfer.ml)

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.