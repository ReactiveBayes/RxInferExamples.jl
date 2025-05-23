# RxInfer.jl Examples

Welcome to the examples gallery for [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl), a Julia package for reactive message passing and probabilistic programming.

!!! note
    This documentation is automatically generated from Jupyter notebooks in the repository.
    The examples are regularly tested to ensure they work with the latest version of RxInfer.jl.

## About RxInfer.jl

RxInfer.jl is a Julia package that combines message passing-based inference with reactive programming paradigms. It provides:

- A flexible framework for probabilistic programming
- Reactive message passing for real-time inference
- Efficient and scalable inference algorithms
- Support for both online and offline inference
- Integration with the Julia ecosystem
- Python integration through client-server infrastructure

Read more about RxInfer.jl in the [RxInfer.jl Documentation](https://docs.rxinfer.com).

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

## Examples

Browse our comprehensive collection of examples in the [List of Examples](autogenerated/list_of_examples.md) section.
Each example demonstrates different aspects of RxInfer.jl's capabilities and includes detailed explanations and code.

## Contributing

We welcome contributions from the community! Whether you want to fix bugs, improve existing examples, or add new ones,
please check our [contribution guide](how_to_contribute.md) for detailed instructions and best practices.

## Getting Started

To run these examples locally:

- Clone the repository:
```bash
git clone https://github.com/ReactiveBayes/RxInferExamples.jl.git
```

- Build the examples:
```bash
make examples
```

- Build the documentation:
```bash
make docs
```

- Preview the documentation:
```bash
make preview
```

## Resources

- [How to Contribute](how_to_contribute.md)
- [RxInfer.jl Documentation](https://docs.rxinfer.com)
- [RxInfer.jl GitHub Repository](https://github.com/ReactiveBayes/RxInfer.jl)
- [Julia Documentation](https://docs.julialang.org)

!!! info "For Developers"
    If you're interested in how the examples and documentation are built,
    check out our [Build System Documentation](how_build_works.md).

## License

RxInfer.jl and these examples are licensed under the MIT License. See the LICENSE file in the repository for more details.
