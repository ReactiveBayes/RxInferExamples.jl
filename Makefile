.PHONY: docs clean docs-setup preview examples

# Default target
all: examples docs

# Build examples
examples:
	julia --project=examples examples/make.jl

# Build specific examples (usage: make example FILTER=pattern)
example:
	@if [ "$(FILTER)" = "" ]; then \
		echo "Error: FILTER is required. Usage: make example FILTER=pattern"; \
		exit 1; \
	fi
	julia --project=examples examples/make.jl $(FILTER)

# Build documentation
docs: docs-setup
	julia --project=docs docs/make.jl

# Clean build artifacts
clean:
	rm -rf docs/build
	rm -rf docs/src/examples

# Install documentation dependencies
docs-setup:
	julia --project=docs -e 'using Pkg; Pkg.instantiate()'

# Preview documentation in browser
preview:
	open docs/build/index.html

# Help command
help:
	@echo "Available targets:"
	@echo "  docs       - Build the documentation"
	@echo "  docs-setup - Install documentation dependencies"
	@echo "  preview    - Open documentation in browser (can be run after `docs` target)"
	@echo "  clean      - Remove build artifacts"
	@echo "  example    - Build specific examples (usage: make example FILTER=pattern)"
