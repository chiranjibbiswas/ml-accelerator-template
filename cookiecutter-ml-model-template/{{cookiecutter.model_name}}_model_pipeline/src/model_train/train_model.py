"""
train_model

This module submits and monitors an AutoML classification job in Azure ML, then extracts
the best child model run, logs its metrics, and persists information about the model.

Functions:
- submit_automl_job: defines and submits the AutoML classification job with given hyperparameters
- main: handles orchestration (parsing arguments, computing period tag, retrieving best run, logging, conditionally failing if metric below threshold)

Outputs:
- Raises error if chosen metric (e.g. AUC) is below acceptable threshold
"""


import argparse
import os
import traceback
import json

from azure.ai.ml import MLClient, automl, Input, Output
from azure.ai.ml.automl import ClassificationModels
from azure.ai.ml.constants import AssetTypes
from mlflow.tracking.client import MlflowClient
import mlflow

from utils import get_logger, get_ml_client, TARGET_VARIABLE
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_connection", type=str)  # dummy dependency
    parser.add_argument("--mltable_name", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--compute_name", type=str)
    parser.add_argument("--primary_metric", type=str)
    parser.add_argument("--max_trials", type=str)
    parser.add_argument("--best_model", type=str)
    return parser.parse_args()


def submit_automl_job(ml_client: MLClient, args: argparse.Namespace):
    """Submit AutoML classification job."""
    try:
        data_asset = ml_client.data.get(name=args.mltable_name, label="latest")
        asset_id = f"azureml:{args.mltable_name}:{data_asset.version}"
        logger.info(f"AutoML training input: {asset_id}")

        training_data_input = Input(type=AssetTypes.MLTABLE, path=asset_id)

        classification_job = automl.classification(
            compute=args.compute_name,
            experiment_name=args.experiment_name,
            training_data=training_data_input,
            test_data_size=0.2,
            target_column_name=TARGET_VARIABLE,
            primary_metric=args.primary_metric,
            n_cross_validations=5,
            enable_model_explainability=True,
            tags={"primary_metric": args.primary_metric, "period": data_asset.tags.get('period','')},
            outputs={"best_model": Output(type="mlflow_model")},
        )

        classification_job.set_limits(
            timeout_minutes=360,
            trial_timeout_minutes=30,
            max_trials=int(args.max_trials),
            enable_early_termination=True,
            max_concurrent_trials=4,
        )

        classification_job.set_training(
            allowed_training_algorithms=[
                ClassificationModels.LIGHT_GBM,
                ClassificationModels.XG_BOOST_CLASSIFIER,
            ]
        )

        returned_job = ml_client.jobs.create_or_update(classification_job)
        logger.info(f"Created AutoML job: {returned_job.name}")
        return returned_job
    except Exception as e:
        logger.error("Error creating AutoML job: %s", e)
        logger.debug(traceback.format_exc())
        raise


def main():
    try:
        args = parse_args()
        _, ml_client = get_ml_client()
        automl_job = submit_automl_job(ml_client, args)

        logger.info(f"Job status URL: {automl_job.services['Studio'].endpoint}")
        ml_client.jobs.stream(automl_job.name)

        returned_automl_job = ml_client.jobs.get(automl_job.name)
        best_child_run_id = returned_automl_job.tags["automl_best_child_run_id"]

        job_run = ml_client.jobs.get(best_child_run_id)
        mlflow_client = MlflowClient()
        best_run = mlflow_client.get_run(best_child_run_id)

        best_model_metrics = best_run.data.metrics
        auc_score = best_model_metrics.get("AUC_weighted", 0)
        mlflow.log_metrics(best_model_metrics)

        info = {
            "automl_best_child_run_id": best_child_run_id,
            "automl_best_child_run_env": job_run.environment,
            "best_model_metrics": best_model_metrics,
        }

        os.makedirs(args.best_model, exist_ok=True)
        with open(os.path.join(args.best_model, "best_model_info.json"), "w") as fp:
            json.dump(info, fp, indent=2)

        if auc_score < 0.75:
            raise ValueError(f"Pipeline terminated: AUC {auc_score} < 0.75")
    except Exception as e:
        logger.error("Error in training component: %s", e)
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    with mlflow.start_run():
        main()
    logger.info("Training component completed.")
