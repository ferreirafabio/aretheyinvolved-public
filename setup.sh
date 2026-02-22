#!/bin/bash
# Setup script for aretheyinvolved project
# Creates virtual environment and installs all dependencies
#
# Usage:
#   ./setup.sh          # CPU only (for downloading/basic processing)
#   ./setup.sh --gpu    # With GPU support (for OCR/Whisper on cluster)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
GPU_MODE=false
if [[ "$1" == "--gpu" ]] || [[ "$1" == "-g" ]]; then
    GPU_MODE=true
fi

echo "=========================================="
echo "  AreTheyInvolved - Setup Script"
if $GPU_MODE; then
    echo "  (GPU Mode)"
fi
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.10" | bc -l) -eq 1 ]]; then
    echo "Error: Python 3.10+ required"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install GPU dependencies if requested
if $GPU_MODE; then
    echo ""
    echo "Installing GPU dependencies (PyTorch with CUDA)..."
    pip install -r requirements-gpu.txt
fi

# Download spaCy model
echo ""
echo "Downloading spaCy English model (en_core_web_lg)..."
python -m spacy download en_core_web_lg

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/raw/doj
mkdir -p data/raw/house-oversight
mkdir -p data/processed/text
mkdir -p data/processed/names

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env with your configuration"
fi

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To download files:"
echo "  python scripts/download_doj_direct.py -d 1      # DOJ Dataset 1"
echo "  python scripts/download_house_oversight.py      # House Oversight"
echo ""
echo "To process files:"
echo "  python scripts/process_all.py data/raw/house-oversight/ --dataset 'House Oversight'"
echo ""

if ! $GPU_MODE; then
    echo "NOTE: GPU dependencies not installed. For OCR/Whisper on GPU cluster:"
    echo "  ./setup.sh --gpu"
    echo ""
fi
