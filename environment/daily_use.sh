#!/bin/bash

# File: environment/daily_use.sh
# Usage: Run from project root: ./environment/daily_use.sh

# Constants
ENV_NAME="sagemaker_env"

echo "Activating environment: $ENV_NAME"

# Force conda initialization
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate environment
conda activate "$ENV_NAME"

echo "Environment activated!"
echo "Python version: $(python --version)"
echo "To deactivate: conda deactivate"