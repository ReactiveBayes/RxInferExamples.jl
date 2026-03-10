# Contributing to RxInfer Examples

We welcome contributions from the community! This guide will help you understand how to add new examples or improve existing ones in the RxInfer Examples collection. Here are the steps to follow to add a new example:

## Location and Structure

Create a new Jupyter notebook in the appropriate category folder. Use `examples/Basic Examples/` for fundamental concepts, `examples/Advanced Examples/` for complex applications, `examples/Problem Specific/` for domain-specific use cases, or `examples/Experimental Examples` for some (potentially unpolished) experiments.

!!! note
    You can also introduce a new category by creating a new folder in the `examples/` directory.
    In this case, you should also add a new entry in the `docs/make.jl` file.

Each example at the very least should have a clear, descriptive title, a `meta.jl` file in the same directory, a local `Project.toml` for dependencies, and any required data files. 

If your example cannot be statically generated, put it inside the `interactive` folder instead.

## Notebook Guidelines

1. **First Cell Requirements**
   The first cell must be a markdown cell. It should contain ONLY the title as `# <title>`. The title should be descriptive and unique (avoid "Overview").

2. **Environment Setup**
   - The notebook will use the environment specified in the `Project.toml` file. Add any additional dependencies to the local project.
   - Notebook should run regardless of what versions of dependencies are being used 
   - The `[compat]` section inside each individual `Project.toml` will **NOT** be respected, do not rely on it

3. **Content Structure**
   The notebook should have a clear introduction and problem description, model specification with explanations, inference procedure details, results analysis and visualization, and comprehensive comments for readability. It is perfectly fine to use LLMs to come up with a nice narrative and/or story for your example. Please, do **NOT** submit examples with just code. Always add some narrative, which explains _why_ the example is doing what it is doing. If you notice an (old) example without explanations or narrative, please open an issue or (even better!) contribute by opening a PR! The examples in the `examples/Experimental Examples` folder might be less explanatory, but do not specifically put your example in this folder just to avoid writing the explanations!

4. **Self-Contained Code**
   - Examples must be fully self-contained without using `include()` statements. The `include()` statements cannot be injected in the [HTML version](https://examples.rxinfer.com) of the examples.
   - All code should be directly in the notebook cells
   - Do not reference external Julia files
   - Users should be able to reproduce examples by simply copying and pasting from the documentation

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

## Hidden/Collapsible Code Blocks

You can hide complex or supplementary code blocks behind collapsible sections to improve readability while still making all code available to users.

1. **Creating Hidden Blocks**
   To create a collapsible code block, add special marker comments within your code blocks:
   ```julia
   ### EXAMPLE_HIDDEN_BLOCK_START(Custom summary text) ###
   # This code will be hidden by default
   function complex_function()
       # Implementation details
       return result
   end

   nothing # to suppress the output in the notebook
   ### EXAMPLE_HIDDEN_BLOCK_END ###
   ```
   Important to note that these comments must be on the first and the last line of the code block respectively. If you want to suppress the output of the code block entirely, add `nothing` at the end of the code block.

2. **Best Practices**
   - Use hidden blocks for implementation details that would distract from the main tutorial flow
   - Provide a descriptive summary that explains what the hidden code does
   - Ensure the code within hidden blocks still runs correctly - it just gets hidden in the display
   - Use for auxiliary functions, data processing, or complex implementations

3. **Result in Documentation**
   The code block will be rendered as a collapsible "details" element with your custom summary text as the clickable header.

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

!!! note
    Note that building the examples locally requires `Weave.jl` package to be installed globally in your Julia environment. Use `julia -e 'using Pkg; Pkg.add("Weave")'` to install it.

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

2. **Build Caching**
   
   The build system caches the results of example compilation. If you make changes to an example and still see old errors after rebuilding:
   
   ```bash
   # Clear all build caches and artifacts
   make clean
   
   # Then rebuild
   make examples
   ```

3. **Common Issues**
   - Ensure all dependencies are in Project.toml
   - Verify plots display correctly
   - Test with a clean environment
   - If errors persist after fixing, run `make clean` to clear cached builds

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
