#!/bin/bash

# Install vim if not present
if ! command -v vim &> /dev/null; then
    echo "Installing vim..."
    apt-get update && apt-get install -y vim
fi

# Add custom aliases to bashrc
echo "Setting up custom aliases..."
cat >> /root/.bashrc << 'EOF'

# Custom aliases
alias activate='conda activate math'
alias gohere='cd /home/ubuntu/modfi'
alias ls='ls -lth --color=auto'
alias js='jetson-containers'
EOF

echo "Setup complete! Your custom aliases are now available."
echo "Run 'source /root/.bashrc' or start a new shell to use them."
