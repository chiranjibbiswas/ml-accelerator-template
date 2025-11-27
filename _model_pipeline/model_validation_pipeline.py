import logging
import sys
import traceback
import os
from dotenv import load_dotenv



load_dotenv()

from azure.ai.ml import MLClient, dsl, load_component
from azure.identity import AzureCliCredential



# Create a logger
logger = logging.getLogger("Build pipeline")
logger.setLevel(logging.INFO)

# Create console handler with stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Avoid duplicate logs if handlers already exist
if not logger.handlers:
    logger.addHandler(handler)


@dsl.pipeline(name="btd_model_validation_pipeline")
def build_pipeline(best_child_run_id, 
                   validation_data_asset=None, 
                   validation_start_date=None, 
                   validation_end_date=None):
    try:
        config_var = {
            "SUBSCRIPTION_ID":os.getenv('subscription_id'),
            "RESOURCE_GROUP":os.getenv('resource_group'),
            "WORKSPACE_NAME":os.getenv('workspace_name'),
            "KEY_VAULT_NAME":os.getenv('key_vault_name'),
            "RAW_STORAGE_ACCOUNT":os.getenv('raw_storage_account_name'),
            "ML_STORAGE_ACCOUNT":os.getenv('ml_storage_account_name'),
            "RAW_CONTAINER":os.getenv('raw_container_name'),
            "ML_CONTAINER":os.getenv('ml_container_name'),
            "DATASTORE_NAME":os.getenv('datastore_name'),
            "MODEL_TYPE":"btd_model"
        }
        # Load YAML components
        score_component = load_component(source="config/validation_component.yml")
        score_step = score_component(
            best_child_run_id = best_child_run_id,
            validation_data_asset = validation_data_asset,
            validation_start_date = validation_start_date,
            validation_end_date = validation_end_date,
            compute_name = os.getenv('compute_cluster'), # compute cluster for batch endpoint
        )
        # Add environment variables dynamically
        score_step.environment_variables = config_var
        
    except Exception as e:
        logger.error("Error in building pipeline: %s", e)
        logger.debug(traceback.format_exc())
        raise e


def run_pipeline():
    # Run pipeline
    pipeline_job = build_pipeline(best_child_run_id = "gifted_deer_z207ns0xn0_74",
            validation_data_asset = "btd_model_validation_dataset:1",
            )
    # set pipeline level compute
    pipeline_job.settings.default_compute=os.getenv('compute_cluster')
    # set pipeline level datastore
    pipeline_job.settings.default_datastore=os.getenv('datastore_name')
    pipeline_job.settings.force_rerun=True
    job_run = ml_client.jobs.create_or_update(pipeline_job, experiment_name="btd_model_pipelines")
    logger.info(f"Created piepline job:{job_run.name}")

# run script
if __name__ == "__main__":
    credential=AzureCliCredential()
    ml_client = MLClient(credential=credential, 
                         subscription_id=os.getenv('subscription_id'), 
                         resource_group_name=os.getenv('resource_group'), 
                         workspace_name=os.getenv('workspace_name')
                        )
    run_pipeline()
    logger.info('Pipeline completed.')