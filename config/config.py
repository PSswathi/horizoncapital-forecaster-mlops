"""
NFCI Forecasting Project - Configuration
=========================================
Central configuration file for all project constants.
This keeps bucket names, prefixes, and settings in one place.
"""

import boto3

# -----------------------------------------------------------------------------
# AWS ACCOUNT & REGION
# -----------------------------------------------------------------------------
# Get account ID dynamically using STS
def get_account_id():
    """Retrieve AWS account ID from STS."""
    sts_client = boto3.client("sts")
    return sts_client.get_caller_identity()["Account"]

AWS_REGION = "us-east-1"  

# -----------------------------------------------------------------------------
# S3 BUCKET CONFIGURATION
# -----------------------------------------------------------------------------
# Bucket name follows pattern: nfci-forecasting-{account-id}
BUCKET_NAME = f"nfci-forecasting-{get_account_id()}"

# S3 Prefixes (folder structure)
S3_PREFIX = {
    # Data prefixes
    "raw": "data/raw",
    "cleaned": "data/cleaned",
    "splits": "data/splits",
    "train": "data/splits/train",
    "validation": "data/splits/validation",
    "test": "data/splits/test",
    "production": "data/splits/production",
    
    # Feature Store
    "features": "features",
    
    # Model artifacts
    "models": "models/artifacts",
    "baselines": "models/baselines",
    
    # Inference outputs
    "forecasts": "forecasts/nfci_predictions",
    
    # Monitoring
    "data_quality": "monitoring/data_quality",
    "model_quality": "monitoring/model_quality",
    
    # Athena
    "athena_results": "athena-results",
}

# data files
RAW_DATA_FILENAME = "state_month_full.csv"
CLEANED_DATA_FILENAME = "cleaned_dataset.csv"

# feature store
FEATURE_GROUP_NAME = "nfci-feature-group"

# model configuration
TARGET_COLUMN = "NFCI"
MODEL_NAME = "nfci-xgboost"

# Data split ratios
SPLIT_RATIOS = {
    "train": 0.60,
    "validation": 0.10,
    "test": 0.10,
    "production": 0.20,
}

# glue and athena
GLUE_DATABASE_NAME = "nfci_database"
GLUE_CRAWLER_NAME = "nfci-raw-data-crawler"
ATHENA_WORKGROUP = "primary"

# sagemaker configuration
SAGEMAKER_ROLE_NAME = "sagemaker-execution-role"

# Processing job
PROCESSING_INSTANCE_TYPE = "ml.m5.large"
PROCESSING_INSTANCE_COUNT = 1

# Training job
TRAINING_INSTANCE_TYPE = "ml.m5.large"
TRAINING_INSTANCE_COUNT = 1

# Batch transform
TRANSFORM_INSTANCE_TYPE = "ml.m5.large"
TRANSFORM_INSTANCE_COUNT = 1


# helper function to build S3 URIs
def get_s3_uri(prefix_key: str, filename: str = "") -> str:
    """
    Build full S3 URI from prefix key and optional filename.
    
    Example:
        get_s3_uri("raw", "state_month_full.csv")
        -> "s3://nfci-forecasting-123456789/data/raw/state_month_full.csv"
    """
    prefix = S3_PREFIX.get(prefix_key, prefix_key)
    if filename:
        return f"s3://{BUCKET_NAME}/{prefix}/{filename}"
    return f"s3://{BUCKET_NAME}/{prefix}"