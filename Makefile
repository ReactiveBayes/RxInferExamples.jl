SHELL = /bin/bash
.DEFAULT_GOAL = help

.PHONY: install

install:
	mkdir -p ./packages
	git -C packages/GraphPPL.jl pull || git clone --branch dev-4.0.0 https://github.com/ReactiveBayes/GraphPPL.jl packages/GraphPPL.jl
	git -C packages/ReactiveMP.jl pull || git clone --branch dev-gp-4.0.0 https://github.com/ReactiveBayes/ReactiveMP.jl packages/ReactiveMP.jl
	git -C packages/RxInfer.jl pull || git clone --branch dev-gp-4.0.0 https://github.com/ReactiveBayes/RxInfer.jl packages/RxInfer.jl
	julia -e '\
		import Pkg; \
		Pkg.activate("drone"); \
		Pkg.instantiate(); \
		Pkg.develop(path="packages/GraphPPL.jl"); \
		Pkg.develop(path="packages/ReactiveMP.jl"); \
		Pkg.develop(path="packages/RxInfer.jl"); \
		Pkg.precompile() \
	'
	echo "Installation complete!"