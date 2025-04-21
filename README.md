# A collection of examples for RxInfer.jl

![](docs/src/assets/biglogo-blacktheme.svg?raw=true&sanitize=true)

[![Official page](https://img.shields.io/badge/official%20page%20-RxInfer-blue)](https://rxinfer.com)
[![Examples](https://img.shields.io/badge/examples-RxInfer-brightgreen)](https://examples.rxinfer.com)
[![Contribute a new example](https://img.shields.io/badge/Contribute-%20a%20new%20example-red)](https://examples.rxinfer.com/how_to_contribute/)
[![Q&A](https://img.shields.io/badge/Q&A-RxInfer-orange)](https://github.com/reactivebayes/RxInfer.jl/discussions)

## Python Integration

RxInfer can be used from Python through our client-server infrastructure:

- **[RxInferServer.jl](https://github.com/lazydynamics/RxInferServer)** - A RESTful API service for deploying RxInfer models
- **[RxInferClient.py](https://github.com/lazydynamics/RxInferClient.py)** - Python SDK for interacting with RxInferServer

The server provides OpenAPI-compliant endpoints for model deployment and inference, while the Python client offers a convenient interface to:
- Create and manage model instances
- Execute inference tasks
- Monitor inference progress
- Handle authentication and API keys
- Process results in a native format

For more information, visit:
- [Server Documentation](https://server.rxinfer.com)
- [Python SDK Documentation](https://lazydynamics.github.io/RxInferClient.py/)

This repository contains a collection of examples for [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl), a Julia package for reactive message passing and probabilistic programming.

Navigate to the [Examples](https://examples.rxinfer.com) page to check the pre-rendered examples or clone the repository and run the examples locally. Additionally, explore the official [RxInfer.jl](https://docs.rxinfer.com) documentation.

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
> If you make changes to an example and still see old errors after rebuilding, try clearing the cache first with the `make clean` command.

All the examples are Jupyter notebooks, which also can be run with [Jupyter](https://jupyter.org/).

## Contributing

We welcome contributions! Please check our [contribution guide](https://examples.rxinfer.com/how_to_contribute/) for guidelines.

## Resources

- [How to Contribute](https://examples.rxinfer.com/how_to_contribute/)
- [RxInfer.jl Documentation](https://docs.rxinfer.com)
- [RxInfer.jl Repository](https://github.com/ReactiveBayes/RxInfer.jl)
- [Examples Documentation](https://examples.rxinfer.com)

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.