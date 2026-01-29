#!/bin/bash
set -euo pipefail

echo "=== Setting up compressdata environment ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync dependencies
echo "Installing dependencies..."
uv sync

# Download enwik8 if not present
mkdir -p src/data
if [ ! -f "src/data/enwik8" ]; then
    echo "Downloading enwik8..."
    curl -L -o /tmp/enwik8.zip "http://mattmahoney.net/dc/enwik8.zip"
    unzip -o /tmp/enwik8.zip -d src/data
    rm /tmp/enwik8.zip
fi

# Configure wandb
echo ""
read -p "Enter your wandb API key (leave blank to skip): " WANDB_KEY
if [ -n "$WANDB_KEY" ]; then
    export WANDB_API_KEY="$WANDB_KEY"
    echo "WANDB_API_KEY set for this session"
    echo ""
    echo "To persist, add to your shell config:"
    echo "  export WANDB_API_KEY=$WANDB_KEY"
fi

echo ""
echo "=== Setup complete ==="
echo "Run: uv run python src/compressor.py"
