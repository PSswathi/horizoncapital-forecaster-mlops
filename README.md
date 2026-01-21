# horizoncapital-forecaster-mlops
Build and deploy an AWS SageMaker MLOps pipeline that forecasts bank capital-planning risk metrics 6–24 months ahead using integrated macroeconomic (FRED), demographic (Census), and financial time-series data, with backtesting and interpretable driver insights.

## PROJECT STATUS -  In Progress

### Project Background:

Banks must allocate capital efficiently across portfolios while meeting regulatory requirements
and maintaining adequate liquidity buffers. Inaccurate long-term forecasts can result in
over-capitalization, which limits growth opportunities, or under-capitalization, which increases
regulatory, credit, and liquidity risk. As economic conditions evolve over time, banks require
1 forecasting approaches that can capture both cyclical macroeconomic changes and 
slower-moving structural trends.
The objective of this project is to develop a machine learning–based forecasting system that
predicts key banking risk or capital-planning proxy metrics over long horizons (6–24 months).
The problem is formulated as a supervised, multivariate time-series forecasting task, where
historical economic, demographic, and financial indicators are used to generate forward-looking
forecasts that support improved capital allocation and planning decisions.



## Local Project Setup Guide

This guide explains how to set up and manage your conda environment for the Assignment using the provided shell scripts.

###  Prerequisites

- Anaconda or Miniconda installed
- pyenv with Python 3.10+ (optional but recommended)
- Terminal/Command line access

### 🛠️ Setup Scripts Overview

We provide 3 shell scripts in the `environment/` folder for different purposes:

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `init_setup.sh` | **Initial environment creation** | First time setup or complete rebuild |
| `update_env.sh` | **Update existing environment** | After modifying `environment.yaml` |
| `daily_use.sh` | **Quick activation** | Every time you start working |

###  1: Create environment.yaml

**First, create your `environment.yaml` file** with project dependencies:

```
# Create environment directory (if not exists)
mkdir -p environment

# Create environment.yaml file
cat > environment/environment.yaml << 'EOF'
    name: sagemaker_env
    channels:
      - conda-forge
      - defaults
    dependencies:
      # Add packages as needed

EOF
```
###  2: Initial Setup (First Time Only)
Run from project root:

```bash
# Make script executable
chmod +x environment/init_setup.sh

# Run initial setup
./environment/init_setup.sh
```
This will:
 - Create new conda environment, Install all dependencies, Register Jupyter kernel, Set up everything for first use

### 3: Update Environment (When Needed)
When you modify environment.yaml (add/remove packages):

```bash
# After editing environment.yaml
./environment/update_env.sh
```

This will:
- Update existing environment, Add new packages, Remove deleted packages, Keep your work intact

###  4: Daily Usage - Every time you start working:
```bash
# Quick activation
./environment/daily_use.sh

# Or manually activate
conda activate sagemaker_env
```

This will:
- Activate your environment, Show Python version, Ready for work
