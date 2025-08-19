# Claude Code Quick Reference

## Essential Commands

```bash
# Basic usage
claude --help                    # Show help
claude --version                 # Check version
claude chat                      # Start interactive chat
claude init                      # Initialize project

# Code analysis
claude analyze <file>            # Analyze specific file
claude analyze .                 # Analyze entire project
claude analyze "*.jl"           # Analyze by pattern

# Code generation
claude "description"             # Generate code from description
claude "description" --file f    # Generate with file context

# Code refactoring
claude "refactor this" --file f  # Refactor specific file
claude "optimize function"       # Optimize code
```

## Interactive Chat Commands

Once in `claude chat` mode:

```
/help          # Show chat commands
/exit          # Quit chat session
/clear         # Clear conversation
/save          # Save conversation
/context       # Show current context
```

## File Patterns

```bash
# Include specific file types
claude analyze "*.jl"           # Julia files only
claude analyze "*.{jl,md}"      # Julia and markdown
claude analyze "src/**/*.jl"    # Recursive in src/

# Exclude patterns
claude analyze . --exclude "*.git/*"
claude analyze . --exclude "outputs/*"
```

## Context and Focus

```bash
# Analyze with specific focus
claude analyze . --focus "performance"
claude analyze . --focus "security"
claude analyze . --focus "documentation"

# Set project context
claude "help with this" --context "Julia scientific computing"
```

## Model Selection

```bash
# Use specific model
claude --model claude-4-opus-20250514
claude --model claude-4-sonnet-20241022
claude --model claude-3-haiku-20240307

# Set default in config
{
  "env": {
    "CLAUDE_MODEL": "claude-4-opus-20250514"
  }
}
```

## Project Configuration

```bash
# Initialize project
claude init

# Creates .claude/settings.json:
{
  "env": {
    "PROJECT_CONTEXT": "Your project description",
    "LANGUAGE": "julia"
  },
  "context": {
    "include_patterns": ["*.jl", "*.md"],
    "exclude_patterns": ["*.git/*", "outputs/*"]
  }
}
```

## Environment Variables

```bash
# Set API key
export ANTHROPIC_API_KEY="your-key"

# Set model
export CLAUDE_MODEL="claude-4-opus-20250514"

# Set temperature
export CLAUDE_TEMPERATURE="0.1"
```

## Common Use Cases

### Julia Development

```bash
# Analyze Julia project
claude analyze . --focus "Julia best practices"

# Generate Julia module
claude "Create a Julia module for Bayesian inference"

# Optimize Julia code
claude "Optimize this function for performance" --file src/model.jl

# Generate tests
claude "Generate tests for this module" --file src/inference.jl
```

### Documentation

```bash
# Generate docs
claude "Document this function" --file src/function.jl

# Create README
claude "Create a comprehensive README for this project"

# Explain code
claude "Explain how this algorithm works" --file algorithm.jl
```

### Debugging

```bash
# Find bugs
claude "Find potential bugs in this code" --file src/code.jl

# Fix errors
claude "Fix the error in this function" --file src/function.jl

# Improve error handling
claude "Add proper error handling" --file src/function.jl
```

## Configuration Files

### Global Settings (`~/.claude/settings.json`)

```json
{
  "env": {
    "ANTHROPIC_API_KEY": "your-key",
    "CLAUDE_MODEL": "claude-4-opus-20250514",
    "DEFAULT_TEMPERATURE": 0.1
  },
  "permissions": {
    "allow": ["code_execution", "file_operations"],
    "deny": ["system_access"]
  }
}
```

### Project Settings (`.claude/settings.json`)

```json
{
  "env": {
    "PROJECT_CONTEXT": "RxInfer examples",
    "LANGUAGE": "julia"
  },
  "context": {
    "include_patterns": ["*.jl", "*.md", "*.toml"],
    "exclude_patterns": ["*.git/*", "outputs/*"]
  }
}
```

## Troubleshooting

```bash
# Check installation
which claude
claude --version

# Check API key
echo $ANTHROPIC_API_KEY
cat ~/.claude/settings.json

# Re-authenticate
claude auth login

# Reinstall
npm install -g @anthropic-ai/claude-code
```

## Useful Aliases

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
# Quick access
alias cc="claude chat"
alias ca="claude analyze"
alias cg="claude"

# Project-specific
alias ccp="claude chat --cwd ."
alias cap="claude analyze ."
```
