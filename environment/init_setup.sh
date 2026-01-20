#!/bin/bash

# File: environment/init_setup.sh
# Usage: Run from project root: ./environment/init_setup.sh

set -e  # Exit on error

# Constants
ENV_NAME="sagemaker_env"
ENV_FILE="environment/environment.yaml"
KERNEL_DISPLAY_NAME="SageMaker (3.10.19)"

echo "Setting up SageMaker environment with pyenv Python 3.10.17..."
echo "Environment name: $ENV_NAME"

# Force conda initialization for this script
CONDA="/opt/anaconda3/bin/conda"
CONDA_BASE=$($CONDA info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Clean up any existing environment
echo "Cleaning up existing environment..."
conda deactivate 2>/dev/null || true
conda env remove -n "$ENV_NAME" 2>/dev/null || true

# Create new environment
echo "Creating conda environment from $ENV_FILE..."
conda env create -f "$ENV_FILE"

# Activate environment
echo "Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Verify Python version
echo "Python version: $(python --version)"

# Install ipykernel if not present
if ! python -c "import ipykernel" 2>/dev/null; then
    echo "Installing ipykernel..."
    conda install ipykernel -y
fi

# Register kernel
echo "Registering Jupyter kernel: $KERNEL_DISPLAY_NAME"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "$KERNEL_DISPLAY_NAME"

echo "Setup complete!"
echo "To activate: conda activate $ENV_NAME"
echo "To check kernel: jupyter kernelspec list"