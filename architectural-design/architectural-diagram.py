"""
Horizon Capital Forecasting System - AWS Architecture Diagram
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.storage import S3
from diagrams.aws.analytics import Glue, Athena, GlueCrawlers, GlueDataCatalog
from diagrams.aws.ml import Sagemaker, SagemakerModel, SagemakerTrainingJob
from diagrams.aws.integration import StepFunctions
from diagrams.aws.management import Cloudwatch
from diagrams.aws.general import GenericDatabase
from diagrams.onprem.client import User
from diagrams.programming.language import Python
from diagrams.onprem.ci import GithubActions
import os

graph_attr = {
    "fontsize": "24",
    "bgcolor": "white",
    "pad": "0.5",
    "splines": "ortho",
}

with Diagram(
    "Horizon Capital Forecasting System - AWS Architecture",
    filename="Horizon_Capital_architecture",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
):
    # Data Ingestion
    with Cluster("Data Ingestion via API"):
        fred_api = GenericDatabase("FRED API")
        census_api = GenericDatabase("Census API")
        ingestion_script = Python("Local Python ETL\nDownload and Clean")
    # S3 Data Lake
    with Cluster("S3 Data Lake"):
        with Cluster("Raw Zone"):
            s3_raw = S3("raw/\nfred/ | census/")
        
        with Cluster("Processed Zone"):
            s3_processed = S3("processed/\ncleaned_dataset.csv")
        
        with Cluster("Feature Zone"):
            s3_features = S3("features/\ntraining_features/")
        
        with Cluster("Model Zone"):
            s3_models = S3("models/\nartifacts/")
        
        with Cluster("Output Zone"):
            s3_forecasts = S3("forecasts/\nHorizon Capital_predictions/")

    # AWS Glue (Data Catalog)
    with Cluster("Data Catalog"):
        glue_crawler = GlueCrawlers("Glue Crawlers")
        glue_catalog = GlueDataCatalog("Glue Data\nCatalog")
    #    glue_etl = Glue("Glue ETL Jobs"), the etl done locally in notebooks

    # Athena (data querying)
    athena = Athena("Athena\n(Ad-hoc Queries)")

    # SageMaker Platform
    with Cluster("SageMaker ML Platform"):
        with Cluster("Feature Engineering"):
            sm_processing = Sagemaker("Processing Job\n(Feature Eng.)")
            sm_feature_store = Sagemaker("Feature Store\n(Offline)")

        with Cluster("Model Development"):
            sm_training = SagemakerTrainingJob("Training Job\n(XGBoost)")
            sm_experiments = Sagemaker("Experiments\n(Tracking)")

        with Cluster("Model Management"):
            sm_registry = SagemakerModel("Model Registry")

        with Cluster("Batch Inference"):
            sm_batch = Sagemaker("Batch Transform\n(Monthly)")

        with Cluster("Monitoring"):
            sm_monitor = Sagemaker("Model Monitor\n(Drift Detection)")

    # Orchestration
    ci_cd = GithubActions("GitHub Actions\n(CI/CD Orchestration)")
    # Monitoring
    cloudwatch = Cloudwatch("CloudWatch\n(Logs & Metrics)")

    # End User
    user = User("Data Science\nTeam")

    # ============================================
    # CONNECTIONS
    # ============================================

      # Local ingestion
    [fred_api, census_api] >> ingestion_script
    ingestion_script >> Edge(label="Upload via boto3") >> s3_raw
    ingestion_script >> Edge(label="Upload via boto3") >> s3_processed


    # Data Catalog Flow
    s3_raw >> glue_crawler >> glue_catalog
    s3_processed >> glue_crawler
    glue_catalog >> athena

    # Feature Engineering Flow
    s3_processed >> sm_processing >> s3_features
    s3_features >> sm_feature_store

    # Training Flow
    sm_feature_store >> sm_training
    sm_training >> sm_experiments
    sm_training >> s3_models
    s3_models >> sm_registry

    # Batch Inference Flow
    sm_registry >> sm_batch
    sm_feature_store >> sm_batch
    sm_batch >> s3_forecasts

    # Monitoring Flow
    sm_batch >> sm_monitor
    sm_monitor >> cloudwatch

    # CI/CD Pipeline
    ci_cd >> Edge(style="dashed", color="blue") >> sm_processing
    ci_cd >> Edge(style="dashed", color="blue") >> sm_training
    ci_cd >> Edge(style="dashed", color="blue") >> sm_batch

    # User Access
    user >> athena
    user >> sm_experiments
    user >> s3_forecasts


print("Diagram generated: Horizon Capital_architecture.png")