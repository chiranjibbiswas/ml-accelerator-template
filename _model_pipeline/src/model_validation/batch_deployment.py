"""
Azure ML Batch Deployment & Validation Module

This module provides an entry point `predict_validation_data` which:
1. Registers a model.
2. Creates or retrieves a batch endpoint.
3. Creates validation dataset input.
4. Deploys the model to the batch endpoint.
5. Runs a batch inference job with validation data.
6. Downloads results and returns them as a dataframe.
"""

import os
import pandas as pd
from pathlib import Path
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import (
    Model,
    BatchEndpoint,
    ModelBatchDeployment,
    CodeConfiguration,
    ModelBatchDeploymentSettings,
    BatchRetrySettings,
)
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
from azure.core.exceptions import ResourceNotFoundError

from utils import get_logger
logger = get_logger(__name__)

# ----------------- Core Functions -----------------

def register_model(ml_client: MLClient, best_run_id: str, best_run_env:str, model_name: str) -> Model:
    """Register a trained MLflow model with Azure ML."""
    try:
        model = Model(
            path=f"azureml://jobs/{best_run_id}/outputs/artifacts/outputs/mlflow-model/",
            name=model_name,
            type=AssetTypes.MLFLOW_MODEL,
            tags={
                    "job_run_id": best_run_id,
                    "job_run_env": best_run_env,
                },
        )
        registered_model = ml_client.models.create_or_update(model)
        logger.info(f"Model '{model_name}' registered successfully.")
        return registered_model
    except Exception as e:
        raise RuntimeError(f"Failed to register model '{model_name}': {e}")


def get_or_create_batch_endpoint(ml_client: MLClient, batch_endpoint_name: str) -> BatchEndpoint:
    """Retrieve an existing batch endpoint or create a new one if not found."""
    try:
        endpoint = ml_client.batch_endpoints.get(name=batch_endpoint_name)
        logger.info(f"Batch endpoint '{batch_endpoint_name}' exists; using it.")
        return endpoint
    except ResourceNotFoundError:
        try:
            endpoint = BatchEndpoint(
                name=batch_endpoint_name,
                description="Batch endpoint for model validation",
            )
            endpoint = ml_client.begin_create_or_update(endpoint).result()
            logger.info(f"Created new batch endpoint '{batch_endpoint_name}'.")
            return endpoint
        except Exception as e:
            raise RuntimeError(f"Failed to create batch endpoint '{batch_endpoint_name}': {e}")
    except Exception as e:
        raise RuntimeError(f"Error while retrieving batch endpoint '{batch_endpoint_name}': {e}")


def download_results(ml_client: MLClient, job_name: str, download_path: str = "./predictions") -> pd.DataFrame:
    """Download batch job results and return them as a dataframe."""
    try:
        ml_client.jobs.download(name=job_name, download_path=download_path, output_name="score")
        csv_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(download_path)
            for f in files if f.endswith(".csv")
        ]
        if not csv_files:
            raise FileNotFoundError(f"No CSV results found in {download_path} for job {job_name}")

        df = pd.read_csv(csv_files[0])
        logger.info(f"Results downloaded and parsed from {csv_files[0]}")
        return df
    except Exception as e:
        raise RuntimeError(f" Failed to download results for job '{job_name}': {e}")


# ----------------- Entry Point -----------------

def predict_validation_data(
    ml_client: MLClient,
    best_run_id: str,
    model_name: str,
    validation_data_asset,
    best_run_env: str,
    batch_endpoint_name: str,
    deployment_name: str,
    compute: str,
) -> list[list]:
    """
    Orchestrates model registration, dataset creation, batch deployment,
    job execution, and result retrieval.

    Args:
        ml_client (MLClient): Azure ML client.
        best_run_id (str): ID of best training run.
        model_name (str): Name to register model under.
        validation_data_asset: ML data asset for validation data.
        best_run_env (str): Training run environment string.
        batch_endpoint_name (str): Name of batch endpoint.
        deployment_name (str): Name of deployment.
        compute (str): Compute cluster name.

    Returns:
        list[list]: Batch inference results as a list of lists.
    """
    try:
        # 1. Register model

        registered_model = register_model(ml_client, best_run_id, best_run_env, model_name)
        
        # 2. Validation dataset
        val_input = Input(type=AssetTypes.URI_FILE, path=validation_data_asset.id)
        
        # 3. Get or create batch endpoint
        endpoint = get_or_create_batch_endpoint(ml_client, batch_endpoint_name)
        
        # 4. Environment from run
        # "azureml://registries/azureml/environments/AzureML-ai-ml-automl/versions/22"
        env_parts = best_run_env.strip().split('/')
        env = ml_client.environments.get(name=env_parts[-3], version=env_parts[-1])
        
        # 5. Deploy model

        current_file_path = Path(__file__).resolve()
        # Access the parent directory, then its parent (two levels up)
        two_levels_up_path = current_file_path.parent.parent
        batch_drive_path = os.path.join(str(two_levels_up_path),'utils','batch_score')
        print(f"Using Batch script from: {batch_drive_path}")
        
        try:
            deployment = ModelBatchDeployment(
                name=deployment_name,
                description="Batch deployment to validate the trained model.",
                endpoint_name=endpoint.name,
                model=registered_model,
                environment=env,
                code_configuration=CodeConfiguration(
                    code=batch_drive_path, scoring_script="batch_script.py"
                ),
                compute=compute,
                settings=ModelBatchDeploymentSettings(
                    instance_count=1,
                    max_concurrency_per_instance=1,
                    mini_batch_size=1,
                    output_action=BatchDeploymentOutputAction.SUMMARY_ONLY,
                    retry_settings=BatchRetrySettings(max_retries=1, timeout=300),
                    logging_level="info",
                ),
            )
            ml_client.batch_deployments.begin_create_or_update(deployment).result()
            logger.info(f"Model deployed as '{deployment_name}' to endpoint '{endpoint.name}'.")
        except Exception as e:
            raise RuntimeError(f" Failed to deploy model batch: {e}")

        # 6. Run job
        job = ml_client.batch_endpoints.invoke(
            endpoint_name=endpoint.name,
            deployment_name=deployment_name,
            inputs={"input": val_input},
        )
        logger.info(f"Batch job '{job.name}' submitted.")
        try:
            ml_client.jobs.stream(job.name)
        except Exception as e:
            raise RuntimeError(f"batch predic job failed: {e}")

        # 7. Download results
        return download_results(ml_client, job.name)

    except Exception as e:
        raise RuntimeError(f" predict_validation_data failed: {e}")
