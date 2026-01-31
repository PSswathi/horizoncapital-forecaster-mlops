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


graph_attr = {
    "fontsize": "16",
    "bgcolor": "white",
    "pad": "0.4",
    "nodesep": "0.4",
    "ranksep": "0.6",
}

with Diagram(
    "Horizon Capital Forecasting System - AWS Architecture",
    filename="Horizon_Capital_architecture",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
):


# USER
    user = User("Data Science Team")

    # S3 DATA LAKE 
    with Cluster("S3 Data Lake\n(Data Pre-loaded)"):
        s3_cleaned = S3("cleaned_dataset.csv\n11,400 rows × 39 cols")
        s3_features = S3("features/\ntraining_features/")
        s3_models = S3("models/\nartifacts/")
        s3_forecasts = S3("forecasts/\nnfci_predictions/")

    # data catalog
    with Cluster("Data Catalog (Optional)"):
        glue_crawler = GlueCrawlers("Glue Crawler")
        glue_catalog = GlueDataCatalog("Glue Catalog")
        athena = Athena("Athena\n(Ad-hoc Queries)")

    # sagemaker ml platform
    with Cluster("SageMaker ML Platform"):
        
        with Cluster("Feature Engineering"):
            sm_processing = Sagemaker("Processing Job\n- Collapse to national\n- Lag features\n- Rolling stats")

        sm_feature_store = Sagemaker("Feature Store\n(Offline)")
        
        with Cluster("Model Training"):
            sm_training = SagemakerTrainingJob("Training Job\n(XGBoost)")
            sm_experiments = Sagemaker("Experiments")

        with Cluster("Model Management"):
            sm_registry = SagemakerModel("Model Registry")

        with Cluster("Batch Inference"):
            sm_batch = Sagemaker("Batch Transform")

        with Cluster("Monitoring"):
            sm_monitor = Sagemaker("Model Monitor")

    #  orchestration
    sm_pipelines = StepFunctions("SageMaker\nPipelines")
    cloudwatch = Cloudwatch("CloudWatch")

    # ========== CONNECTIONS ==========
    
    # User uploads cleaned data
    user >> s3_cleaned
    
    # Data catalog 
    s3_cleaned >> glue_crawler >> glue_catalog >> athena
    
    # ML Pipeline: Feature Engineering
    s3_cleaned >> sm_processing >> s3_features
    s3_features >> sm_feature_store
    
    # ML Pipeline: Training
    sm_feature_store >> sm_training
    sm_training >> sm_experiments
    sm_training >> s3_models
    s3_models >> sm_registry
    
    # ML Pipeline: Inference
    sm_registry >> sm_batch
    sm_feature_store >> sm_batch
    sm_batch >> s3_forecasts
    
    # Monitoring
    sm_batch >> sm_monitor >> cloudwatch
    
    # Orchestration (dashed blue lines)
    sm_pipelines >> Edge(style="dashed", color="blue") >> sm_processing
    sm_pipelines >> Edge(style="dashed", color="blue") >> sm_training
    sm_pipelines >> Edge(style="dashed", color="blue") >> sm_batch
   
print("Diagram generated: Horizon_Capital_architecture.png")