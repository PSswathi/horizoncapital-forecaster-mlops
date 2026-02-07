"""
NFCI Forecasting System - AWS Architecture Diagram
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.storage import S3
from diagrams.aws.analytics import GlueCrawlers, GlueDataCatalog, Athena
from diagrams.aws.ml import Sagemaker, SagemakerModel, SagemakerTrainingJob
from diagrams.aws.management import Cloudwatch
from diagrams.aws.general import User

graph_attr = {
    "fontsize": "24",
    "bgcolor": "white",
    "pad": "0.5",
}

with Diagram(
    "NFCI Forecasting System - AWS Architecture",
    show=False,
    direction="TB",
    filename="HorizonCapital_NFCI_Architecture",
    outformat="png",
    graph_attr=graph_attr,
):
    user = User("Data Science\nTeam")

    # s3 data lake cluster
    with Cluster("S3 Data Lake"):
        s3_raw = S3("Raw Data")
        s3_features = S3("Features")
        s3_training = S3("Training Data")
        s3_models = S3("Model Artifacts")
        s3_predictions = S3("Predictions")

    # data catalog cluster
    with Cluster("Data Catalog"):
        glue_crawler = GlueCrawlers("Glue Crawler")
        glue_catalog = GlueDataCatalog("Glue Catalog")
        athena = Athena("Athena")

    # sagemaker ML platform cluster
    with Cluster("SageMaker ML Platform"):
        
        with Cluster("Feature Engineering"):
            processing = Sagemaker("Processing Job")
        
        with Cluster("Model Training"):
            training = SagemakerTrainingJob("Training Job\n(DeepAR)")
            experiments = Sagemaker("Experiments")
        
        with Cluster("Model Management"):
            feature_store = Sagemaker("Feature Store")
            registry = Sagemaker("Model Registry")
        
        with Cluster("Batch Inference"):
            batch_transform = Sagemaker("Batch Transform")
        
        with Cluster("Monitoring"):
            model_monitor = Sagemaker("Model Monitor")

    
    # Model monitoring and alerting (cloudwatch)
    
    cloudwatch = Cloudwatch("CloudWatch")

    # Pipeline orchestration (SageMaker Pipelines)
    pipelines = Sagemaker("SageMaker\nPipelines")

    # connections

    # User interactions
    user >> s3_raw
    user >> athena

    # Data catalog flow
    s3_raw >> glue_crawler >> glue_catalog >> athena

    # ML Pipeline flow
    s3_raw >> processing
    processing >> s3_features
    processing >> feature_store
    s3_features >> training
    training >> experiments
    training >> s3_models
    s3_models >> registry
    registry >> batch_transform
    batch_transform >> s3_predictions
    
    # Monitoring
    batch_transform >> model_monitor
    model_monitor >> cloudwatch

    # Pipeline orchestration (dashed lines)
    pipelines >> Edge(style="dashed") >> processing
    pipelines >> Edge(style="dashed") >> training
    pipelines >> Edge(style="dashed") >> batch_transform
    pipelines >> Edge(style="dashed") >> registry

print("Diagram generated: HorizonCapital_NFCI_Architecture.png")