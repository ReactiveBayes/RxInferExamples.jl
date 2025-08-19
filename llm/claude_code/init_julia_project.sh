#!/bin/bash

# Initialize Claude Code for Julia Projects
# This script sets up Claude Code with Julia-specific configuration

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if we're in a Julia project
check_julia_project() {
    if [ -f "Project.toml" ] || [ -f "JuliaProject.toml" ]; then
        print_success "Julia project detected"
        return 0
    else
        print_warning "No Julia project detected (no Project.toml found)"
        print_status "This script will still work, but consider running it from a Julia project root"
        return 1
    fi
}

# Get project information
get_project_info() {
    local PROJECT_NAME=""
    local PROJECT_DESCRIPTION=""
    
    if [ -f "Project.toml" ]; then
        PROJECT_NAME=$(grep "^name" Project.toml | sed 's/name = "\(.*\)"/\1/' 2>/dev/null || echo "julia_project")
        PROJECT_DESCRIPTION=$(grep "^description" Project.toml | sed 's/description = "\(.*\)"/\1/' 2>/dev/null || echo "Julia project")
    else
        PROJECT_NAME=$(basename "$PWD")
        PROJECT_DESCRIPTION="Julia project"
    fi
    
    echo "$PROJECT_NAME|$PROJECT_DESCRIPTION"
}

# Create .claude directory and configuration
create_claude_config() {
    local PROJECT_NAME="$1"
    local PROJECT_DESCRIPTION="$2"
    
    print_status "Creating Claude Code configuration..."
    
    # Create .claude directory
    mkdir -p .claude
    
    # Create settings.json
    cat > .claude/settings.json << EOF
{
  "env": {
    "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY:-your-api-key-here}",
    "CLAUDE_MODEL": "claude-4-opus-20250514",
    "PROJECT_CONTEXT": "${PROJECT_DESCRIPTION}",
    "LANGUAGE": "julia",
    "FRAMEWORK": "julia",
    "DEFAULT_TEMPERATURE": 0.1,
    "MAX_TOKENS": 4000
  },
  "permissions": {
    "allow": [
      "code_execution",
      "file_operations",
      "web_search"
    ],
    "deny": [
      "system_access",
      "network_access"
    ]
  },
  "context": {
    "include_patterns": [
      "*.jl",
      "*.md",
      "*.toml",
      "*.jmd",
      "*.ipynb"
    ],
    "exclude_patterns": [
      "*.git/*",
      "outputs/*",
      "*.log",
      "*.tmp",
      "*.cache",
      "build/*",
      "deps/*",
      ".julia/*"
    ],
    "max_file_size": "2MB",
    "recursive": true
  },
  "ui": {
    "theme": "auto",
    "font_size": 14,
    "show_line_numbers": true,
    "syntax_highlighting": true
  },
  "julia_specific": {
    "package_manager": "Pkg",
    "test_framework": "Test",
    "documentation_tool": "Documenter.jl",
    "code_style": "JuliaFormatter",
    "linting": "JET.jl",
    "profiling": "Profile.jl"
  },
  "analysis_focus": [
    "julia_best_practices",
    "performance_optimization",
    "type_stability",
    "memory_efficiency",
    "scientific_computing"
  ],
  "code_generation": {
    "include_tests": true,
    "include_documentation": true,
    "follow_julia_style": true,
    "use_multiple_dispatch": true,
    "prefer_immutable": true
  },
  "refactoring": {
    "preserve_type_stability": true,
    "maintain_multiple_dispatch": true,
    "optimize_memory_usage": true,
    "improve_readability": true
  }
}
EOF
    
    print_success "Configuration created at .claude/settings.json"
}

# Create .gitignore entry for Claude Code
update_gitignore() {
    if [ -f ".gitignore" ]; then
        if ! grep -q ".claude/" .gitignore; then
            echo "" >> .gitignore
            echo "# Claude Code configuration" >> .gitignore
            echo ".claude/" >> .gitignore
            print_success "Added .claude/ to .gitignore"
        else
            print_status ".claude/ already in .gitignore"
        fi
    else
        print_warning "No .gitignore found - consider creating one"
    fi
}

# Create example usage file
create_example_usage() {
    print_status "Creating example usage file..."
    
    cat > .claude/EXAMPLE_USAGE.md << EOF
# Claude Code Usage Examples for This Project

## Quick Start

\`\`\`bash
# Start interactive chat
claude chat

# Analyze entire project
claude analyze .

# Analyze specific files
claude analyze "*.jl"
claude analyze "src/**/*.jl"

# Generate code
claude "Create a function to calculate the mean of a vector"
claude "Generate tests for this module" --file src/module.jl

# Refactor code
claude "Refactor this function for better performance" --file src/function.jl
\`\`\`

## Project-Specific Commands

\`\`\`bash
# Analyze Julia code quality
claude analyze . --focus "julia_best_practices"

# Optimize performance
claude analyze . --focus "performance_optimization"

# Check type stability
claude analyze . --focus "type_stability"

# Generate documentation
claude "Document this function" --file src/function.jl
\`\`\`

## Common Patterns

\`\`\`bash
# With file context
claude "help with this function" --file src/function.jl

# With specific focus
claude analyze . --focus "memory_efficiency"

# Exclude certain directories
claude analyze . --exclude "outputs/*" --exclude "*.git/*"
\`\`\`
EOF
    
    print_success "Example usage created at .claude/EXAMPLE_USAGE.md"
}

# Test Claude Code configuration
test_configuration() {
    print_status "Testing Claude Code configuration..."
    
    if command -v claude >/dev/null 2>&1; then
        if claude --version >/dev/null 2>&1; then
            print_success "Claude Code is working"
            
            # Test project analysis
            print_status "Testing project analysis..."
            if timeout 30s claude analyze . --help >/dev/null 2>&1; then
                print_success "Project analysis ready"
            else
                print_warning "Project analysis may need API key configuration"
            fi
        else
            print_warning "Claude Code installation may be incomplete"
        fi
    else
        print_warning "Claude Code not found in PATH"
        print_status "Install with: npm install -g @anthropic-ai/claude-code"
    fi
}

# Show next steps
show_next_steps() {
    echo
    print_success "Julia project Claude Code setup complete!"
    echo
    print_status "Next steps:"
    echo "  1. Configure your API key:"
    echo "     - Edit .claude/settings.json"
    echo "     - Or run: claude auth login"
    echo "  2. Start using Claude Code:"
    echo "     - claude chat"
    echo "     - claude analyze ."
    echo "  3. Check examples: .claude/EXAMPLE_USAGE.md"
    echo
    print_status "Get API key at: https://console.anthropic.com/"
    echo
}

# Main execution
main() {
    echo "=========================================="
    echo "  Claude Code Julia Project Setup"
    echo "=========================================="
    echo
    
    # Check if we're in a Julia project
    check_julia_project
    
    # Get project information
    local PROJECT_INFO=$(get_project_info)
    local PROJECT_NAME=$(echo "$PROJECT_INFO" | cut -d'|' -f1)
    local PROJECT_DESCRIPTION=$(echo "$PROJECT_INFO" | cut -d'|' -f2)
    
    print_status "Project: $PROJECT_NAME"
    print_status "Description: $PROJECT_DESCRIPTION"
    echo
    
    # Create configuration
    create_claude_config "$PROJECT_NAME" "$PROJECT_DESCRIPTION"
    
    # Update gitignore
    update_gitignore
    
    # Create example usage
    create_example_usage
    
    # Test configuration
    test_configuration
    
    # Show next steps
    show_next_steps
}

# Run main function
main "$@"
