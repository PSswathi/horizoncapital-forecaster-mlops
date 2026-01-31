"""
Horizon Capital Forecasting System - SageMaker Pipelines Workflow

Pipeline handles:
- Feature engineering
- Model training
- Batch inference
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.ml import Sagemaker, SagemakerModel, SagemakerTrainingJob
from diagrams.aws.storage import S3
from diagrams.aws.management import Cloudwatch

graph_attr = {
    "fontsize": "18",
    "bgcolor": "white",
    "pad": "0.5",
}

with Diagram(
    "NFCI Forecasting - SageMaker Pipeline",
    filename="sagemaker_pipeline",
    show=False,
    direction="TB",
    graph_attr=graph_attr,
):
    
    # Input
    s3_input = S3("S3: cleaned_dataset.csv")

    with Cluster("Step 1: Feature Engineering"):
        step1 = Sagemaker("Processing Job\n\n- Collapse to national level\n- Create lag features\n  (t-1, t-3, t-6, t-12)\n- Rolling statistics\n- Rate of change\n- Yield curve flags")

    with Cluster("Step 2: Feature Store"):
        step2 = Sagemaker("Feature Store\n(Offline)\n\n- Version features\n- Track lineage")

    with Cluster("Step 3: Train/Test Split"):
        step3 = Sagemaker("Split Data\n\n- Chronological order\n- Train: 2006-2019\n- Test: 2020-2024\n- No shuffle")

    with Cluster("Step 4: Model Training"):
        step4 = SagemakerTrainingJob("XGBoost Training\n\n- Expanding window CV\n- Quantile regression\n- Log to Experiments")

    with Cluster("Step 5: Evaluation"):
        step5 = Sagemaker("Model Evaluation\n\n- MAE/RMSE by horizon\n- Feature importance\n- SHAP values")
        
        condition = Sagemaker("Check:\nMetrics >= Baseline?")

    with Cluster("Step 6: Model Registry"):
        step6 = SagemakerModel("Register Model\n\n- Version\n- Metrics\n- Lineage")

    with Cluster("Step 7: Batch Inference"):
        step7 = Sagemaker("Batch Transform\n\n- Horizons: t+6,12,18,24\n- Point forecasts\n- Prediction intervals")

    # Output
    s3_output = S3("S3: nfci_forecasts.parquet")
    cloudwatch = Cloudwatch("CloudWatch Metrics")

    # Flow
    s3_input >> step1 >> step2 >> step3 >> step4 >> step5 >> condition
    
    condition >> Edge(label="Yes", color="green") >> step6 >> step7
    condition >> Edge(label="No", color="red", style="dashed") >> step4
    
    step7 >> s3_output
    step7 >> cloudwatch


print("Diagram generated: sagemaker_pipeline.png")