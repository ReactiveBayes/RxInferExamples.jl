# Claude Code Documentation

## Overview

Claude Code is a powerful AI coding assistant that integrates with your development workflow through the command line. It provides intelligent code analysis, generation, refactoring, and debugging capabilities.

## Installation

### Prerequisites

- **macOS 11 (Big Sur) or later**
- **Homebrew** (package manager)
- **Node.js** (JavaScript runtime)

### Installation Methods

#### Method 1: npm (Recommended)

```bash
# Install globally via npm
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version
```

#### Method 2: Homebrew

```bash
# Add Anthropic tap
brew tap anthropic/claude

# Install via Homebrew
brew install claude-code

# Verify installation
claude --version
```

## API Key Setup

### 1. Get Your API Key

1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Sign in or create an account
3. Navigate to "API Keys" section
4. Click "Create API Key"
5. Provide a descriptive name
6. Copy the generated key (it won't be shown again)

### 2. Configure API Key

#### Option A: Interactive Setup (Recommended)

```bash
# Run interactive authentication
claude auth login

# Follow prompts to enter your API key
```

#### Option B: Environment Variable

```bash
# Set for current session
export ANTHROPIC_API_KEY="your-api-key-here"

# Make permanent (add to ~/.zshrc or ~/.bashrc)
echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

#### Option C: Configuration File

Create `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_API_KEY": "your-api-key-here",
    "CLAUDE_MODEL": "claude-4-opus-20250514"
  },
  "permissions": {
    "allow": ["code_execution", "file_operations"],
    "deny": []
  }
}
```

## Usage

### Basic Commands

```bash
# Start interactive chat session
claude chat

# Analyze a specific file
claude analyze filename.jl

# Generate code from description
claude "Create a function to calculate fibonacci numbers"

# Refactor existing code
claude "Refactor this function to improve performance"

# Generate tests
claude "Generate unit tests for this module"
```

### Command Reference

```bash
# Get help
claude --help

# Check version
claude --version

# Initialize project
claude init

# Start chat in specific directory
claude chat --cwd /path/to/project

# Use specific model
claude --model claude-4-sonnet-20241022
```

### Interactive Chat Commands

Once in chat mode:

- **Type your question** and press Enter
- **Use `/help`** for chat-specific commands
- **Use `/exit`** to quit chat session
- **Use `/clear`** to clear conversation history
- **Use `/save`** to save conversation

## Project Integration

### Initialize Project

```bash
# Navigate to your project
cd /path/to/your/project

# Initialize Claude Code
claude init

# This creates .claude/ directory with project settings
```

### Project Configuration

Create `.claude/settings.json` in your project:

```json
{
  "env": {
    "CLAUDE_MODEL": "claude-4-opus-20250514",
    "PROJECT_CONTEXT": "Julia scientific computing with RxInfer"
  },
  "permissions": {
    "allow": ["code_execution", "file_operations", "web_search"],
    "deny": ["system_access"]
  },
  "context": {
    "include_patterns": ["*.jl", "*.md", "*.toml"],
    "exclude_patterns": ["*.git/*", "outputs/*", "*.log"]
  }
}
```

## Advanced Features

### Code Analysis

```bash
# Analyze entire project
claude analyze .

# Analyze specific file types
claude analyze "*.jl"

# Analyze with specific focus
claude analyze filename.jl --focus "performance optimization"
```

### Code Generation

```bash
# Generate from scratch
claude "Create a Julia module for Bayesian inference"

# Generate with context
claude "Add error handling to this function" --file filename.jl

# Generate tests
claude "Generate comprehensive tests for this module" --file filename.jl
```

### Refactoring

```bash
# Refactor specific function
claude "Refactor the optimize function for better readability" --file filename.jl

# Refactor entire file
claude "Refactor this file to follow Julia best practices" --file filename.jl
```

## IDE Integration

### Visual Studio Code

1. Install "Claude Code" extension from VS Code Marketplace
2. Configure API key in extension settings
3. Use Command Palette: `Ctrl+Shift+P` → "Claude Code: Chat"

### JetBrains IDEs

1. Install Claude Code plugin from JetBrains Marketplace
2. Configure API key in plugin settings
3. Access via Tools → Claude Code

### Vim/Neovim

1. Install via plugin manager (e.g., vim-plug)
2. Configure LSP integration
3. Use `:Claude` commands

## Security Best Practices

### API Key Protection

- **Never commit API keys** to version control
- **Use environment variables** or secure configuration files
- **Rotate keys regularly** for enhanced security
- **Monitor usage** through Anthropic Console

### Permission Management

```json
{
  "permissions": {
    "allow": ["code_execution", "file_operations"],
    "deny": ["system_access", "network_access"]
  }
}
```

## Troubleshooting

### Common Issues

#### "Command not found: claude"
```bash
# Reinstall globally
npm install -g @anthropic-ai/claude-code

# Check PATH
echo $PATH
which claude
```

#### "API key not found"
```bash
# Check environment variable
echo $ANTHROPIC_API_KEY

# Re-authenticate
claude auth login
```

#### "Permission denied"
```bash
# Check file permissions
ls -la ~/.claude/

# Fix permissions
chmod 600 ~/.claude/settings.json
```

### Getting Help

- **Documentation**: [Anthropic Docs](https://docs.anthropic.com/)
- **Support**: [Anthropic Help Center](https://support.anthropic.com/)
- **Community**: [Anthropic Discord](https://discord.gg/anthropic)
- **GitHub**: [Claude Code Repository](https://github.com/anthropics/anthropic-claude-code)

## Examples

### Julia-Specific Usage

```bash
# Analyze Julia project structure
claude analyze . --focus "Julia project organization"

# Generate Julia module
claude "Create a Julia module for Gaussian process regression"

# Optimize Julia code
claude "Optimize this Julia function for better performance" --file src/model.jl

# Generate Julia tests
claude "Generate comprehensive tests for this Julia module" --file src/inference.jl
```

### Scientific Computing

```bash
# Analyze mathematical models
claude "Review this mathematical model for correctness" --file model.jl

# Generate documentation
claude "Generate comprehensive documentation for this algorithm" --file algorithm.jl

# Optimize numerical computations
claude "Optimize these numerical computations for speed" --file computations.jl
```

## Configuration Reference

### Global Settings (`~/.claude/settings.json`)

```json
{
  "env": {
    "ANTHROPIC_API_KEY": "your-api-key",
    "CLAUDE_MODEL": "claude-4-opus-20250514",
    "DEFAULT_TEMPERATURE": 0.1,
    "MAX_TOKENS": 4000
  },
  "ui": {
    "theme": "auto",
    "font_size": 14,
    "show_line_numbers": true
  },
  "permissions": {
    "allow": ["code_execution", "file_operations", "web_search"],
    "deny": ["system_access"]
  }
}
```

### Project Settings (`.claude/settings.json`)

```json
{
  "env": {
    "PROJECT_CONTEXT": "RxInfer examples and research",
    "LANGUAGE": "julia",
    "FRAMEWORK": "rxinfer"
  },
  "context": {
    "include_patterns": ["*.jl", "*.md", "*.toml"],
    "exclude_patterns": ["*.git/*", "outputs/*", "*.log"],
    "max_file_size": "1MB"
  }
}
```

## Updates and Maintenance

### Update Claude Code

```bash
# Update via npm
npm update -g @anthropic-ai/claude-code

# Update via Homebrew
brew upgrade claude-code

# Check for updates
claude --version
```

### Clean Installation

```bash
# Remove via npm
npm uninstall -g @anthropic-ai/claude-code

# Remove via Homebrew
brew uninstall claude-code

# Clean configuration
rm -rf ~/.claude/
```

---

*For the latest updates and features, visit [Anthropic's official documentation](https://docs.anthropic.com/).*
