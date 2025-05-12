import os
import logging
from pathlib import Path
import time
import yaml
import json
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

# Default paths
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CONFIG_PATH = CONFIG_DIR / "mlops_config.yaml"

def load_mlflow_config():
    """Load MLflow configuration from config file"""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('mlflow', {})
    except Exception as e:
        logger.error(f"Error loading MLflow config: {e}")
        return {}

def setup_mlflow():
    """Setup MLflow tracking and return the experiment"""
    config = load_mlflow_config()
    
    # Set the tracking server URI
    tracking_uri = config.get('tracking_uri', os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI set to: {tracking_uri}")
    
    # Get or create the experiment
    experiment_name = config.get('experiment_name', 'object_detection')
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new MLflow experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name}")
    except Exception as e:
        logger.error(f"Error setting up MLflow experiment: {e}")
        experiment_id = None
    
    return experiment_id

def log_model_performance(metrics, model_path=None, input_example=None, model_name="object_detection"):
    """
    Log model performance metrics to MLflow
    
    Args:
        metrics (dict): Dictionary of metrics to log
        model_path (str, optional): Path to the model to log as an artifact
        input_example (numpy.ndarray, optional): Example input for model signature
        model_name (str, optional): Name to register the model as
    
    Returns:
        run_id: ID of the MLflow run
    """
    # Setup MLflow tracking
    experiment_id = setup_mlflow()
    config = load_mlflow_config()
    
    # Start a new run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # Log parameters about the model
        mlflow.log_param("model_type", "YOLOv11")
        mlflow.log_param("timestamp", time.time())
        
        # If a model path is provided, log the model
        if model_path and Path(model_path).exists():
            try:
                # Try to infer the model signature if an input example is provided
                signature = None
                if input_example is not None:
                    # Example output has boxes, scores, classes
                    output_example = (
                        np.array([[10, 10, 100, 100]]),  # Boxes
                        np.array([0.95]),               # Scores
                        np.array([0])                   # Classes
                    )
                    signature = infer_signature(input_example, output_example)
                
                # Log the model
                mlflow.pytorch.log_model(
                    pytorch_model=model_path,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=model_name if config.get('register_model', False) else None
                )
                logger.info(f"Logged model {model_path} to MLflow")
            except Exception as e:
                logger.error(f"Error logging model to MLflow: {e}")
        
        logger.info(f"Logged metrics to MLflow: {metrics}")
        return run.info.run_id

def register_model(run_id, model_name="object_detection", stage="Production"):
    """
    Register a model version in the MLflow Model Registry
    
    Args:
        run_id (str): ID of the MLflow run containing the model
        model_name (str): Name to register the model as
        stage (str): Stage to promote the model to (None, Staging, Production, Archived)
    
    Returns:
        model_version: Version number of the registered model
    """
    try:
        client = MlflowClient()
        
        # Check if model exists in registry
        try:
            client.get_registered_model(model_name)
        except:
            client.create_registered_model(model_name)
        
        # Get the run's artifact URI
        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri
        model_uri = f"{artifact_uri}/model"
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f"Registered model {model_name} version {model_version.version}")
        
        # Transition the model to the specified stage
        if stage:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
            logger.info(f"Transitioned model {model_name} version {model_version.version} to {stage}")
        
        return model_version.version
    
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        return None

def get_latest_model_version(model_name="object_detection", stage="Production"):
    """
    Get the latest version of a model from the MLflow Model Registry
    
    Args:
        model_name (str): Name of the registered model
        stage (str): Stage to filter by (None, Staging, Production, Archived)
    
    Returns:
        model_uri: URI of the latest model version
    """
    try:
        client = MlflowClient()
        
        if stage:
            # Get all versions in the specified stage
            versions = client.get_latest_versions(model_name, stages=[stage])
        else:
            # Get all versions
            versions = client.get_latest_versions(model_name)
        
        if not versions:
            logger.warning(f"No versions found for model {model_name} in stage {stage}")
            return None
        
        # Sort by version number and get the latest
        latest_version = sorted(versions, key=lambda x: x.version, reverse=True)[0]
        model_uri = f"models:/{model_name}/{latest_version.version}"
        
        logger.info(f"Latest version of {model_name} in {stage}: {latest_version.version}")
        return model_uri
    
    except Exception as e:
        logger.error(f"Error getting latest model version: {e}")
        return None

def log_model_drift(current_metrics, baseline_metrics, model_name="object_detection"):
    """
    Log model drift data to MLflow
    
    Args:
        current_metrics (dict): Dictionary of current model metrics
        baseline_metrics (dict): Dictionary of baseline model metrics
        model_name (str): Name of the model
        
    Returns:
        run_id: ID of the MLflow run
    """
    # Calculate drift metrics
    drift_metrics = {}
    for key in current_metrics:
        if key in baseline_metrics:
            drift_metrics[f"{key}_drift"] = current_metrics[key] - baseline_metrics[key]
            drift_metrics[f"{key}_drift_percent"] = (
                (current_metrics[key] - baseline_metrics[key]) / baseline_metrics[key] * 100
                if baseline_metrics[key] != 0 else 0
            )
    
    # Setup MLflow tracking
    experiment_id = setup_mlflow()
    
    # Start a new run
    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Log drift metrics
        for key, value in drift_metrics.items():
            mlflow.log_metric(key, value)
        
        # Log current and baseline metrics
        for key, value in current_metrics.items():
            mlflow.log_metric(f"current_{key}", value)
        
        for key, value in baseline_metrics.items():
            mlflow.log_metric(f"baseline_{key}", value)
        
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("drift_analysis_time", time.time())
        
        logger.info(f"Logged drift metrics to MLflow: {drift_metrics}")
        return run.info.run_id

if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Simple test
    setup_mlflow()
    test_metrics = {
        "mAP": 0.75,
        "precision": 0.82,
        "recall": 0.78,
        "latency_ms": 45.2
    }
    log_model_performance(test_metrics)
    print("MLflow test completed.") 