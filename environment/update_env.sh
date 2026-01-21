#!/bin/bash

# File: environment/update_env.sh
# Usage: Run from project root: ./environment/update_env.sh

set -e

# Constants
ENV_NAME="sagemaker_env"
ENV_FILE="environment/environment.yaml"

echo "Updating environment: $ENV_NAME from $ENV_FILE"

# Force conda initialization
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate environment
conda activate "$ENV_NAME"

# Update from environment.yaml
echo "Applying updates..."
conda env update -f "$ENV_FILE" --prune

echo "Environment update complete!"
echo "To activate: conda activate $ENV_NAME"