#!/bin/bash

# Claude Code Setup Script
# This script helps configure Claude Code with proper API key setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Claude Code installation
check_claude_installation() {
    print_status "Checking Claude Code installation..."
    
    if command_exists claude; then
        VERSION=$(claude --version 2>/dev/null || echo "unknown")
        print_success "Claude Code is installed: $VERSION"
        return 0
    else
        print_error "Claude Code is not installed or not in PATH"
        return 1
    fi
}

# Function to install Claude Code
install_claude() {
    print_status "Installing Claude Code..."
    
    if command_exists npm; then
        print_status "Installing via npm..."
        npm install -g @anthropic-ai/claude-code
        print_success "Claude Code installed via npm"
    elif command_exists brew; then
        print_status "Installing via Homebrew..."
        brew tap anthropic/claude
        brew install claude-code
        print_success "Claude Code installed via Homebrew"
    else
        print_error "Neither npm nor Homebrew found. Please install one of them first."
        print_status "npm: https://nodejs.org/"
        print_status "Homebrew: https://brew.sh/"
        exit 1
    fi
}

# Function to check API key
check_api_key() {
    if [ -n "$ANTHROPIC_API_KEY" ]; then
        print_success "API key found in environment variable"
        return 0
    fi
    
    if [ -f "$HOME/.claude/settings.json" ]; then
        if grep -q "ANTHROPIC_API_KEY" "$HOME/.claude/settings.json"; then
            print_success "API key found in configuration file"
            return 0
        fi
    fi
    
    print_warning "No API key found"
    return 1
}

# Function to get API key from user
get_api_key() {
    echo
    print_status "To use Claude Code, you need an API key from Anthropic."
    print_status "Get your API key at: https://console.anthropic.com/"
    echo
    read -p "Enter your Anthropic API key: " -s API_KEY
    echo
    
    if [ -z "$API_KEY" ]; then
        print_error "API key cannot be empty"
        return 1
    fi
    
    # Test the API key
    print_status "Testing API key..."
    if curl -s -H "x-api-key: $API_KEY" \
            -H "content-type: application/json" \
            -d '{"model":"claude-3-haiku-20240307","max_tokens":10,"messages":[{"role":"user","content":"Hello"}]}' \
            https://api.anthropic.com/v1/messages > /dev/null 2>&1; then
        print_success "API key is valid"
        return 0
    else
        print_error "API key validation failed"
        return 1
    fi
}

# Function to configure API key
configure_api_key() {
    local API_KEY="$1"
    
    print_status "Configuring API key..."
    
    # Create .claude directory
    mkdir -p "$HOME/.claude"
    
    # Create or update settings.json
    cat > "$HOME/.claude/settings.json" << EOF
{
  "env": {
    "ANTHROPIC_API_KEY": "$API_KEY",
    "CLAUDE_MODEL": "claude-4-opus-20250514"
  },
  "permissions": {
    "allow": ["code_execution", "file_operations"],
    "deny": []
  }
}
EOF
    
    # Set proper permissions
    chmod 600 "$HOME/.claude/settings.json"
    
    print_success "API key configured in $HOME/.claude/settings.json"
    
    # Also set as environment variable for current session
    export ANTHROPIC_API_KEY="$API_KEY"
    print_success "API key set as environment variable for current session"
}

# Function to add to shell profile
add_to_shell_profile() {
    local PROFILE_FILE=""
    
    if [ -n "$ZSH_VERSION" ]; then
        PROFILE_FILE="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ]; then
        PROFILE_FILE="$HOME/.bashrc"
    else
        PROFILE_FILE="$HOME/.profile"
    fi
    
    if [ -f "$PROFILE_FILE" ]; then
        if ! grep -q "ANTHROPIC_API_KEY" "$PROFILE_FILE"; then
            echo "" >> "$PROFILE_FILE"
            echo "# Claude Code API Key" >> "$PROFILE_FILE"
            echo "export ANTHROPIC_API_KEY=\"$API_KEY\"" >> "$PROFILE_FILE"
            print_success "Added API key to $PROFILE_FILE"
            print_status "Run 'source $PROFILE_FILE' or restart your terminal to apply changes"
        else
            print_warning "API key already exists in $PROFILE_FILE"
        fi
    fi
}

# Function to test Claude Code
test_claude() {
    print_status "Testing Claude Code..."
    
    if timeout 30s claude --version > /dev/null 2>&1; then
        print_success "Claude Code is working correctly"
        return 0
    else
        print_error "Claude Code test failed"
        return 1
    fi
}

# Function to show next steps
show_next_steps() {
    echo
    print_success "Claude Code setup complete!"
    echo
    print_status "Next steps:"
    echo "  1. Start using Claude Code: claude chat"
    echo "  2. Analyze your code: claude analyze ."
    echo "  3. Get help: claude --help"
    echo
    print_status "Documentation: llm/claude_code/README.md"
    print_status "API Console: https://console.anthropic.com/"
    echo
}

# Main execution
main() {
    echo "=========================================="
    echo "    Claude Code Setup Script"
    echo "=========================================="
    echo
    
    # Check if Claude Code is installed
    if ! check_claude_installation; then
        print_status "Installing Claude Code..."
        install_claude
        
        # Verify installation
        if ! check_claude_installation; then
            print_error "Installation failed. Please check the error messages above."
            exit 1
        fi
    fi
    
    # Check if API key is configured
    if ! check_api_key; then
        print_status "API key configuration required..."
        
        # Get API key from user
        if get_api_key; then
            configure_api_key "$API_KEY"
            add_to_shell_profile
        else
            print_error "Failed to configure API key"
            exit 1
        fi
    fi
    
    # Test Claude Code
    if test_claude; then
        show_next_steps
    else
        print_error "Setup incomplete. Please check the configuration."
        exit 1
    fi
}

# Run main function
main "$@"
