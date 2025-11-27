"""
data_preprocess

This module handles creation of training and validation datasets from raw blobs in storage,
merges them with label data from Synapse, and registers data assets as MLTable/URI in Azure ML.

Functions:
- create_training_dataset: generate and upload raw training CSV for a two-month window
- create_validation_dataset: generate and upload raw validation CSV for the last two months
- create_dataset: builds merged training & validation data (with label), logging shapes
- create_mltable: converts a CSV into an MLTable asset and registers it
- main: orchestrates argument parsing, dataset creation, and asset registration

Assumes existence of utility modules for blob listing/processing, feature definitions, engine,
ML client, logging, etc.
"""


import argparse
import os
import traceback
from datetime import datetime, date
import pandas as pd
from sqlalchemy import text
from typing import Tuple

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.storage.blob import BlobServiceClient

import mlflow
import mltable
from mltable import DataType

from utils import get_ml_client, get_logger, Config, get_engine, features
from utils import (
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
    RAW_STORAGE_ACCOUNT,
    ML_STORAGE_ACCOUNT,
    RAW_CONTAINER,
    ML_CONTAINER,
    DATASTORE_NAME,
    MODEL_TYPE,
    TARGET_VARIABLE,
)
from utils.parse_json_to_csv import list_blobs_within_date_range, process_blobs_to_csv

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mltable_name", type=str)
    parser.add_argument("--training_start_date", type=str)
    parser.add_argument("--training_end_date", type=str)
    parser.add_argument("--validation_start_date", type=str)
    parser.add_argument("--validation_end_date", type=str)

    parser.add_argument("--step_connection", type=str)
    return parser.parse_args()


def create_validation_dataset(raw_container_client, root_folder: str, blob_path: str, start_date: date, end_date: date, local_dir: str) -> pd.DataFrame:
    """Create and upload validation dataset."""
    try:
        validation_file_path = f"{local_dir}/raw_validation_dataset.csv"
        validation_blob_path = f"{blob_path}/raw_validation_dataset.csv"
        blob_client = raw_container_client.get_blob_client(validation_blob_path)

        blob_info_list = list_blobs_within_date_range(raw_container_client, root_folder, start_date, end_date)
        raw_validation_data = process_blobs_to_csv(raw_container_client, blob_info_list)

        raw_validation_data.columns = [col.lower() for col in raw_validation_data.columns]
        raw_validation_data.to_csv(validation_file_path, index=False)

        with open(validation_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        return raw_validation_data
    except Exception as e:
        logger.error(f"Error creating validation dataset: {e}")
        raise


def create_training_dataset(raw_container_client, root_folder: str, blob_path: str, start_date: date, end_date: date, local_dir: str) -> pd.DataFrame:
    """Create and upload training dataset."""
    try:
        training_file_path = f"{local_dir}/raw_training_dataset.csv"
        training_blob_path = f"{blob_path}/raw_training_dataset.csv"
        blob_client = raw_container_client.get_blob_client(training_blob_path)

        blob_info_list = list_blobs_within_date_range(raw_container_client, root_folder, start_date, end_date)
        raw_training_data = process_blobs_to_csv(raw_container_client, blob_info_list)

        raw_training_data.columns = [col.lower() for col in raw_training_data.columns]
        raw_training_data.to_csv(training_file_path, index=False)

        with open(training_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        return raw_training_data
    except Exception as e:
        logger.error(f"Error creating training dataset: {e}")
        raise


def create_dataset(blob_service_client, raw_container_name: str, local_dir: str, raw_training_blob_path:str, raw_validation_blob_path:str, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create processed training and validation datasets merged with labels."""
    try:
        
        # cols = ["leadid"] + features # for prod #NOSONAR
        train_cols = features + [TARGET_VARIABLE]
        val_cols = ['leadid', 'customerid'] + features + ['sold', TARGET_VARIABLE]
        # raw csv data path
        root_folder = "raw_data"
        synapse_config = Config(os.getenv("KEY_VAULT_NAME"))
        engine = get_engine(synapse_server=synapse_config.synapse_server, #NOSONAR
                            synapse_db=synapse_config.synapse_db,
                            synapse_user=synapse_config.synapse_user,
                            synapse_password=synapse_config.synapse_password) 
        
        container_client = blob_service_client.get_container_client(raw_container_name)

        raw_training_data = create_training_dataset(container_client, root_folder, raw_training_blob_path, args.training_start_date, args.training_end_date, local_dir)
        raw_validation_data = create_validation_dataset(container_client, root_folder, raw_validation_blob_path, args.validation_start_date, args.validation_end_date, local_dir)

        raw_training_data = raw_training_data[(raw_training_data.lead_issued==1) & (raw_training_data.division=='MAC')]
        
        # with engine.connect() as conn: #NOSONAR
        #     query = text(f"""
        #         SELECT leadid, btd_flag
        #         FROM {synapse_config.synapse_db}.dbo.lead_activities 
        #         WHERE entrydate BETWEEN '{start_date}' AND '{cur_date}'
        #     """)
        #     leads = pd.read_sql(query, conn)

        # training_data = raw_training_data[cols].merge(leads[["leadid", "btd_flag"]], on="leadid", how="inner").drop("leadid", axis=1) #NOSONAR
        # validation_data = raw_validation_data[cols].merge(leads[["leadid", "btd_flag"]], on="leadid", how="inner")

        training_data = raw_training_data[train_cols]
        validation_data = raw_validation_data[val_cols]

        logger.info(f"Training dataset shape: {training_data.shape}")
        logger.info(f"Validation dataset shape: {validation_data.shape}")
        # return only features + btd_flag
        return training_data, validation_data
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        raise


def create_mltable(ml_client, training_data_path: str, table_path: str, mltable_name: str, period: str):
    """Create and register MLTable."""
    try:
        paths = [{"file": training_data_path}]
        tbl = mltable.from_delimited_files(paths)
        tbl = tbl.convert_column_types({"btd_flag": DataType.to_string()})
        tbl.save(path=table_path, colocated=True, show_progress=True, overwrite=True)

        my_data = Data(
            path=table_path,
            type=AssetTypes.MLTABLE,
            description="Preprocessed training data",
            name=mltable_name,
            tags={"period": period}
        )
        ml_client.data.create_or_update(my_data)
    except Exception as e:
        logger.error(f"Error creating MLTable: {e}")
        raise


def main():
    try:
        args = parse_args()
        credential, ml_client = get_ml_client()

        args.training_start_date = datetime.strptime(args.training_start_date, "%Y-%m-%d").date()
        args.training_end_date = datetime.strptime(args.training_end_date, "%Y-%m-%d").date()
        args.validation_start_date = datetime.strptime(args.validation_start_date, "%Y-%m-%d").date()
        args.validation_end_date = datetime.strptime(args.validation_end_date, "%Y-%m-%d").date()

        training_period = f"{args.training_start_date.strftime('%Y%m')}_{args.training_end_date.strftime('%Y%m')}"
        validation_period = f"{args.validation_start_date.strftime('%Y%m')}_{args.validation_end_date.strftime('%Y%m')}"
        local_dir = args.step_connection

        # ML workspace input data path
        ml_datastore = f"azureml://subscriptions/{SUBSCRIPTION_ID}/resourcegroups/{RESOURCE_GROUP}/workspaces/{WORKSPACE_NAME}/datastores/{DATASTORE_NAME}/paths"
        training_blob_path = f"{MODEL_TYPE}/input_data/training_data/{training_period}"
        validation_blob_path = f"{MODEL_TYPE}/input_data/validation_data/{validation_period}"

        training_data_path = f"{local_dir}/training_dataset.csv"
        validation_data_path = f"{local_dir}/validation_dataset.csv"

        raw_blob_service_client = BlobServiceClient(
            account_url=f"https://{RAW_STORAGE_ACCOUNT}.blob.core.windows.net",
            credential=credential
        )

        ml_blob_service_client = BlobServiceClient(
            account_url=f"https://{ML_STORAGE_ACCOUNT}.blob.core.windows.net",
            credential=credential
        )
        
        training_data, validation_data = create_dataset(blob_service_client=raw_blob_service_client, 
                                                        raw_container_name=RAW_CONTAINER, 
                                                        local_dir=local_dir, 
                                                        raw_training_blob_path=training_blob_path,
                                                        raw_validation_blob_path=validation_blob_path,
                                                        args=args)
        training_data.to_csv(training_data_path, index=False)
        validation_data.to_csv(validation_data_path, index=False)
        
        # Create MLTable for training data
        table_path = f"{ml_datastore}/{training_blob_path}"
        create_mltable(ml_client, training_data_path, table_path, args.mltable_name, training_period)
        logger.info(f"Created ML-table data assets for period: {training_period}")

        ml_container_client = ml_blob_service_client.get_container_client(ML_CONTAINER)
        ml_blob_client = ml_container_client.get_blob_client(f"{validation_blob_path}/validation_dataset.csv")
        
        # Upload validation data and register as Data asset
        with open(validation_data_path, "rb") as data:
            ml_blob_client.upload_blob(data, overwrite=True)
        
        validation_asset_name = "btd_model_validation_dataset"
        validation_data_asset = ml_client.data.create_or_update(
            Data(
                path=f"{ml_datastore}/{validation_blob_path}/validation_dataset.csv",
                type=AssetTypes.URI_FILE,
                description="Validation dataset (last 2 months).",
                name=validation_asset_name,
                tags={"period": validation_period},
            )
        )
        
        mlflow.log_metric(validation_asset_name, int(validation_data_asset.version))
        logger.info(f"Created validation data assets for period: {validation_period}")

        logger.info(f"Created raw training data and uploaded to: {RAW_STORAGE_ACCOUNT}/{RAW_CONTAINER}/{training_blob_path}")
        logger.info(f"Created raw validation data and uploaded to: {RAW_STORAGE_ACCOUNT}/{RAW_CONTAINER}/{validation_blob_path}")
        logger.info(f"Uploaded training data to: {ML_STORAGE_ACCOUNT}/{ML_CONTAINER}/{training_blob_path}")
        logger.info(f"Uploaded validation data to: {ML_STORAGE_ACCOUNT}/{ML_CONTAINER}/{validation_blob_path}")
        
    except Exception as e:
        logger.error("Error in data preprocess: %s", e)
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    with mlflow.start_run():
        main()
    logger.info("Data preprocess component completed.")
