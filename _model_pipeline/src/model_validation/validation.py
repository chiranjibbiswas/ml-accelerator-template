import argparse
import os
import traceback
from datetime import datetime
import pandas as pd

import mlflow
from mlflow.tracking.client import MlflowClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.storage.blob import BlobServiceClient

from .data import btd_score_buckets, generate_report, upload_to_azure_blob, create_dataset
from .batch_deployment import predict_validation_data
from utils import (
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
    RAW_STORAGE_ACCOUNT,
    ML_STORAGE_ACCOUNT,
    RAW_CONTAINER,
    ML_CONTAINER,
    DATASTORE_NAME,
    MODEL_TYPE
)
from utils import get_logger, get_ml_client
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model validation and return the namespace object."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_child_run_id", type=str)
    parser.add_argument("--validation_data_asset", type=str)
    parser.add_argument("--validation_start_date", type=str)
    parser.add_argument("--validation_end_date", type=str)
    parser.add_argument("--compute_name", type=str)
    return parser.parse_args()


def check_data_consistency(ml_client, validation_data_asset) -> bool:
    """Check if the period in the validation data asset matches the provided period."""
    asset_name = validation_data_asset.split(':')[0]
    asset_version = validation_data_asset.split(':')[1]
    try:
        val_dataset = ml_client.data.get(name=asset_name, version=asset_version)
    except Exception as e:
        raise RuntimeError(f"Failed to get validation dataset '{asset_name}:{asset_version}': {e}")
    return val_dataset


def main():
    try:
        args = parse_args()
        if args.validation_start_date and args.validation_end_date:
            args.validation_start_date = datetime.strptime(args.validation_start_date, "%Y-%m-%d").date()
            args.validation_end_date = datetime.strptime(args.validation_end_date, "%Y-%m-%d").date()
            if not args.validation_end_date > args.validation_start_date: #NOSONAR
                raise RuntimeError(f"Validation end date has to be greater than start date. Given:{args.validation_start_date} - {args.validation_start_date}")


        local_dir = "temp"
        os.makedirs(local_dir, exist_ok=True)

        ml_datastore = f"azureml://subscriptions/{SUBSCRIPTION_ID}/resourcegroups/{RESOURCE_GROUP}/workspaces/{WORKSPACE_NAME}/datastores/{DATASTORE_NAME}/paths"
        credential, ml_client = get_ml_client()
        
        raw_blob_service_client = BlobServiceClient(
            account_url=f"https://{RAW_STORAGE_ACCOUNT}.blob.core.windows.net",
            credential=credential,
        )

        ml_blob_service_client = BlobServiceClient(
            account_url=f"https://{ML_STORAGE_ACCOUNT}.blob.core.windows.net",
            credential=credential,
        )

        raw_container_client = raw_blob_service_client.get_container_client(RAW_CONTAINER)
        ml_container_client = ml_blob_service_client.get_container_client(ML_CONTAINER)

        mlflow_client = MlflowClient()
        job = ml_client.jobs.get(name = args.best_child_run_id)
        best_run = mlflow_client.get_run(args.best_child_run_id)

        # Step 1: Validation data features with leadid, sold and btd_flag
        if args.validation_start_date and args.validation_end_date:
            validation_data_path = f"{local_dir}/validation_dataset.csv"
            validation_period = f"{args.validation_start_date.strftime('%Y%m')}_{args.validation_end_date.strftime('%Y%m')}"
            validation_blob_path = f"{MODEL_TYPE}/input_data/validation_data/{validation_period}"
            logger.info(f"Creating validation dataset for the period:{validation_period}.")
            
            validation_data = create_dataset(raw_container_client= raw_container_client, local_dir=local_dir, raw_validation_blob_path=validation_blob_path, args=args)
            validation_data.to_csv(validation_data_path, index=False)
            
            ml_blob_client = ml_container_client.get_blob_client(f"{validation_blob_path}/validation_dataset.csv")
            
            # Upload validation data and register as Data asset
            with open(validation_data_path, "rb") as data:
                ml_blob_client.upload_blob(data, overwrite=True)
                
            validation_asset_name = f"{MODEL_TYPE}_validation_dataset"
            val_dataset = ml_client.data.create_or_update(
                Data(
                    path=f"{ml_datastore}/{validation_blob_path}",
                    type=AssetTypes.URI_FILE,
                    description="Validation dataset.",
                    name=validation_asset_name,
                    tags={"period": validation_period},
                )
            )
            mlflow.log_metric(validation_asset_name, int(val_dataset.version))
        elif args.validation_data_asset:
            logger.info(f"Using the validation data asset: {args.validation_data_asset}")
            val_dataset = check_data_consistency(ml_client=ml_client, 
                                        validation_data_asset=args.validation_data_asset)
            validation_period = val_dataset.tags.get('period')
            validation_blob_path = f"{MODEL_TYPE}/input_data/validation_data/{validation_period}"
            validation_data = pd.read_csv(val_dataset.path)
        else:
            raise RuntimeError("Both validation dates and validation_data_asset cannot be empty.")
        logger.info(f"Validation dataset size: {validation_data.shape}")

        # Step 2: Predict
        pred = predict_validation_data(
            ml_client=ml_client,
            best_run_id=args.best_child_run_id,
            model_name="btd_validation_model",
            validation_data_asset = val_dataset,
            best_run_env=job.environment,
            batch_endpoint_name="renuity-model-validation-ep",
            deployment_name="blue",
            compute=args.compute_name,
        )

        validation_data = validation_data.merge(pred, on='leadid', how='left', copy=True)
        no_score = validation_data.btd_1_score.isnull().sum()
        if no_score>0:
            raise RuntimeError(f"Not all leadids from validation dataset has been scored. Total leads without score: {no_score}")
        
        # Step 3: Buckets
        btd_buckets = btd_score_buckets(validation_data)
        btd_buckets_csv = os.path.join(local_dir, "btd_score_buckets.csv")
        btd_buckets.to_csv(btd_buckets_csv, index=False)
        
        # Step 4: Report
        ts = datetime.today().strftime('%Y%m%d_%H%M')
        score_file = f"{MODEL_TYPE}_lead_prioritization_score_{ts}.csv"
        analysis_file = f"{MODEL_TYPE}_validation_results_{ts}.xlsx"
        generate_report(metrics=best_run.data.metrics, final_df=btd_buckets, report_name=analysis_file)
          # consistent with generate_report

        # Step 5: Upload artifacts
        
        upload_to_azure_blob(analysis_file, f"{validation_blob_path}/{analysis_file}", ml_container_client)
        upload_to_azure_blob(btd_buckets_csv, f"{validation_blob_path}/{score_file}", ml_container_client)
        logger.info(f"Uploaded lead prioritization score to {ML_CONTAINER}/{validation_blob_path}/{score_file}")
        logger.info(f"Uploaded validation result excel file to {ML_CONTAINER}/{validation_blob_path}/{analysis_file}")
    except Exception as e:
        logger.error("Error in validation pipeline: %s", e)
        logger.debug(traceback.format_exc())
        raise


# run script
if __name__ == "__main__":
    # Start MLflow logging
    with mlflow.start_run():
        main()
    logger.info('Validation component completed.')