# A collection of examples for RxInfer.jl

![](docs/src/assets/biglogo-blacktheme.svg?raw=true&sanitize=true)

[![Official page](https://img.shields.io/badge/official%20page%20-RxInfer-blue)](https://rxinfer.com)
[![Examples](https://img.shields.io/badge/examples-RxInfer-brightgreen)](https://examples.rxinfer.com)
[![Contribute a new example](https://img.shields.io/badge/Contribute-%20a%20new%20example-red)](https://examples.rxinfer.com/how_to_contribute/)
[![Q&A](https://img.shields.io/badge/Q&A-RxInfer-orange)](https://github.com/reactivebayes/RxInfer.jl/discussions)

This repository contains a collection of examples for [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl), a Julia package for reactive message passing and probabilistic programming.

Navigate to the [Examples](https://examples.rxinfer.com) page to check the pre-rendered examples or clone the repository and run the examples locally. Additionally, explore the official [RxInfer.jl](https://docs.rxinfer.com) documentation.

## How to run the examples locally

1. Clone the repository:
   ```bash
   git clone https://github.com/ReactiveBayes/RxInferExamples.jl.git
   ```

2. Build the examples:
   ```bash
   make examples
   ```

3. Build and preview the documentation:
   ```bash
   make docs
   make preview
   ```

> [!NOTE]  
> If you make changes to an example and still see old errors after rebuilding, try clearing the cache first with the `make clean` command.

All the examples are Jupyter notebooks, which also can be run with [Jupyter](https://jupyter.org/). Note, however, that the automatic build process 
merges the dependencies in all the examples into one single temporary environment. This means that dependencies can be resolved differently when running examples individually in comparison with the automatic build process.

## Interactive examples

Most of the examples are available on the [official website](https://examples.rxinfer.com). Some examples, however, cannot be converted to a static HTML file and thus are placed under `interactive/` folder. Those examples can only be executed inside Jupyter notebook environment (or a plugin like in VSCode) as they may require some features that are not available in pure HTML.

## Contributing

We welcome contributions! Please check our [contribution guide](https://examples.rxinfer.com/how_to_contribute/) for guidelines.

## Resources

- [How to Contribute](https://examples.rxinfer.com/how_to_contribute/)
- [RxInfer.jl Documentation](https://docs.rxinfer.com)
- [RxInfer.jl Repository](https://github.com/ReactiveBayes/RxInfer.jl)
- [Examples Documentation](https://examples.rxinfer.com)

## Python Integration

RxInfer can be used from Python through client-server infrastructure developed by Lazy Dynamics.

- **[RxInferServer.jl](https://github.com/lazydynamics/RxInferServer)** - A RESTful API service for deploying RxInfer models
- **[RxInferClient.py](https://github.com/lazydynamics/RxInferClient.py)** - Python SDK for interacting with RxInferServer

**Note** that the license for the `RxInferServer` is different from Rxinfer and is hosted under a different organization.

The server provides OpenAPI-compliant endpoints for model deployment and inference, while the Python client offers a convenient interface to:
- Create and manage model instances
- Execute inference tasks
- Monitor inference progress
- Handle authentication and API keys
- Process results in a native format

For more information, visit:
- [Server Documentation](https://server.rxinfer.com)
- [Python SDK Documentation](https://lazydynamics.github.io/RxInferClient.py/)

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.
