#!/usr/bin/env bash
# =============================================================================
# System Dependencies Installation
# =============================================================================
# Installs system packages required by the evaluation framework

set -euo pipefail

echo "Installing system dependencies for Ubuntu/Debian..."

# Check for apt-get (Ubuntu/Debian only)
if ! command -v apt-get &> /dev/null; then
    echo "Error: This script requires Ubuntu/Debian (apt-get not found)"
    echo "Please install dependencies manually:"
    echo "  - yq: https://github.com/mikefarah/yq"
    echo "  - jq, screen, curl, wget, git"
    exit 1
fi

# Install yq for YAML processing in bash scripts
install_yq() {
    echo "Installing yq for YAML processing..."
    
    # Check if already installed
    if command -v yq &> /dev/null; then
        echo "✓ yq already installed: $(yq --version)"
        return 0
    fi
    
    # Try snap first (most reliable)
    if command -v snap &> /dev/null; then
        echo "Installing yq via snap..."
        if sudo snap install yq 2>/dev/null; then
            echo "✓ yq installed via snap"
            return 0
        fi
    fi
    
    # Try apt package manager
    echo "Installing yq via apt..."
    if sudo apt-get update && sudo apt-get install -y yq 2>/dev/null; then
        echo "✓ yq installed via apt"
        return 0
    fi
    
    # Fallback: direct download from GitHub
    echo "Installing yq from GitHub releases..."
    YQ_VERSION="v4.40.5"  # Pin to stable version
    YQ_URL="https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_linux_amd64"
    
    if curl -L "$YQ_URL" -o /tmp/yq && sudo mv /tmp/yq /usr/local/bin/yq && sudo chmod +x /usr/local/bin/yq; then
        echo "✓ yq installed from GitHub to /usr/local/bin/yq"
        return 0
    fi
    
    echo "✗ Failed to install yq"
    echo "Please install manually: https://github.com/mikefarah/yq"
    return 1
}

# Install other system dependencies
install_other_deps() {
    echo "Installing other system dependencies..."
    
    # Update package list and install dependencies
    sudo apt-get update
    sudo apt-get install -y \
        curl \
        wget \
        jq \
        screen \
        git
    
    echo "✓ System dependencies installed"
}

# Main installation
main() {
    install_yq
    install_other_deps
    
    echo ""
    echo "✓ System dependencies installation complete"
    echo ""
    echo "Installed tools:"
    command -v yq &> /dev/null && echo "  yq: $(yq --version)"
    command -v jq &> /dev/null && echo "  jq: $(jq --version)"
    command -v screen &> /dev/null && echo "  screen: $(screen --version | head -1)"
    command -v git &> /dev/null && echo "  git: $(git --version)"
}

main "$@"
