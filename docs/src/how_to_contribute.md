# Contributing to RxInfer Examples

We welcome contributions from the community! This guide will help you understand how to add new examples or improve existing ones in the RxInfer Examples collection.

## Adding New Examples

### Location and Structure

1. Create a new Jupyter notebook in the appropriate category folder:
   - `examples/Basic Examples/` for fundamental concepts
   - `examples/Advanced Examples/` for complex applications
   - `examples/Problem Specific/` for domain-specific use cases

2. Each example should have:
   - A clear, descriptive title
   - A `meta.jl` file in the same directory
   - A local `Project.toml` for dependencies
   - Any required data files

### Notebook Guidelines

1. **First Cell Requirements**
   - Must be a markdown cell
   - Should contain ONLY the title as `# <title>`
   - Title should be descriptive and unique (avoid "Overview")

2. **Environment Setup**
   - The notebook will use the environment specified in the `Project.toml` file
   - Add any additional dependencies to the local project

3. **Content Structure**
   - Clear introduction and problem description
   - Model specification with explanations
   - Inference procedure details
   - Results analysis and visualization
   - Comprehensive comments for readability

### Mathematical Content

1. **Equation Formatting**
   ```
   $$\begin{aligned}
   <latex equations here>
   \end{aligned}$$
   ```
   This is important for the documentation to render correctly.
   The automatic rendering of equations is handled by the `make.jl` script and it does not understand the spaces after the `$$` or `$`

2. **Equation Rules**
   - No space after opening `$$` or `$`
   - Separate display equations with empty lines
   - Inline equations use single `$...$`
   - Example: `$$a + b$$` (not `$$ a + b $$`)

### Code Guidelines

1. **Scoping Rules**
   - Use `let ... end` blocks for local scoping
   - Use `global` keyword when needed:
   ```julia
   variable = 0
   for i in 1:10
       global variable = variable + i
   end
   ```

2. **Visualization**
   - All plots should display automatically
   - Save special figures (e.g., GIFs) to `figure-name.gif`
   - Reference saved figures with `![](figure-name.gif)` right after the cell

### Metadata Requirements

Create a `meta.jl` file in your example's directory with:
```julia
return (
    title = "Your Example Title",
    description = """
    A clear description of what the example demonstrates.
    """,
    tags = ["category", "relevant", "tags", "here"]
)
```

## Testing Your Example

1. **Local Testing**
   ```bash
   # Test all examples
   make examples

   # Test specific example
   make example FILTER=YourNotebookName

   # Render the documentation
   make docs

   # Preview the documentation
   make preview
   ```

2. **Common Issues**
   - Check for UndefVarError (scoping issues)
   - Ensure all dependencies are in Project.toml
   - Verify plots display correctly
   - Test with a clean environment

## Important Notes

!!! note "Plotting Package Preference"
    Please use `Plots.jl` instead of `PyPlot`. PyPlot's installation significantly impacts CI build times.

!!! warning "Documentation Generation"
    Ensure your notebook renders correctly in the documentation by:
    - Following equation formatting rules
    - Using proper cell types
    - Including all necessary resources

## Getting Help

If you're unsure about anything:
1. Check existing examples for reference
2. Open an issue for guidance
3. Ask in the discussions section

Your contributions help make RxInfer.jl better for everyone!
