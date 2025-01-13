# Contributing to RxInfer Examples

We welcome contributions from the community! This guide will help you understand how to add new examples or improve existing ones in the RxInfer Examples collection. Here are the steps to follow to add a new example:

## Location and Structure

Create a new Jupyter notebook in the appropriate category folder. Use `examples/Basic Examples/` for fundamental concepts, `examples/Advanced Examples/` for complex applications, or `examples/Problem Specific/` for domain-specific use cases.

!!! note
    You can also introduce a new category by creating a new folder in the `examples/` directory.
    In this case, you should also add a new entry in the `docs/make.jl` file.

Each example at the very least should have a clear, descriptive title, a `meta.jl` file in the same directory, a local `Project.toml` for dependencies, and any required data files.

## Notebook Guidelines

1. **First Cell Requirements**
   The first cell must be a markdown cell. It should contain ONLY the title as `# <title>`. The title should be descriptive and unique (avoid "Overview").

2. **Environment Setup**
   The notebook will use the environment specified in the `Project.toml` file. Add any additional dependencies to the local project.

3. **Content Structure**
   The notebook should have a clear introduction and problem description, model specification with explanations, inference procedure details, results analysis and visualization, and comprehensive comments for readability.

## Mathematical Content

!!! note
    The automatic rendering of equations is handled by the `make.jl` script and it does not understand the spaces after the `$$` or `$`. Below are the rules for formatting equations.

1. **Equation Formatting**
   ```
   Some text

   $$\begin{aligned}
   <latex equations here>
   \end{aligned}$$

   Some other text
   ```
   Do not add spaces before or after the `$$` or `$`

2. **Equation Rules**
   - No space after opening `$$` or `$`
   - Separate display equations with empty lines
   - Inline equations use single `$...$`, e.g. `$$a + b$$` and not `$$ a + b $$`

## Visualization and figures

- All plots rendered with `Plots.jl` should display automatically
- Asset figures can be saved in the same directory as the notebook and referenced with `![](figure-name.png)`
- Special figures (e.g., GIFs) should be saved to `generated-in-the-notebook.gif` in the same directory as the notebook
- Reference saved figures as markdown with `![](generated-in-the-notebook.gif)` right after the cell that generated it

## Metadata Requirements

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
