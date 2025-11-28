import logging
import sys
import traceback
import os
from dotenv import load_dotenv

load_dotenv()

from azure.ai.ml import MLClient, dsl, load_component
from azure.identity import AzureCliCredential
from azure.ai.ml.entities import (
    Environment,
    BuildContext)


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


@dsl.pipeline(name="{{cookiecutter.mtp_name}}_model_training_pipeline")
def build_pipeline(training_start_date, training_end_date, validation_start_date, validation_end_date):
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
            "MODEL_TYPE":"{{cookiecutter.MODEL_TYPE}}"
        }
        
        # Load YAML components and register
        prep_component = load_component(source="config/preprocess_component.yml")
        train_component = load_component(source="config/training_component.yml")
        
        prep_step = prep_component(
            mltable_name= "{{cookiecutter.mltable_name}}",
            training_start_date = training_start_date,
            training_end_date = training_end_date,
            validation_start_date = validation_start_date,
            validation_end_date = validation_end_date,
        )
        # Add environment variables dynamically
        prep_step.environment_variables = config_var
        
        train_step = train_component(
            step_connection = prep_step.outputs.step_connection,
            mltable_name = "{{cookiecutter.mltable_name}}",
            experiment_name = "{{cookiecutter.experiment_name}}",
            compute_name = os.getenv('compute_cluster'),
            primary_metric = "{{cookiecutter.primary_metric}}",
            max_trials = "{{cookiecutter.max_trials}}"
        )
        # Add environment variables dynamically
        train_step.environment_variables = config_var
    except Exception as e:
        logger.error("Error in building pipeline: %s", e)
        logger.debug(traceback.format_exc())
        raise e


def run_pipeline():

    env_name = "{{cookiecutter.env_name}}"
    if os.getenv('create_compute_env').lower()=='true':
        create_env(env_name)
        logger.info(f'Creating pipeline component runtime environment: {env_name}')

    # Run pipeline
    pipeline_job = build_pipeline(training_start_date = "{{cookiecutter.training_start_date}}",
            training_end_date = "{{cookiecutter.training_end_date}}",
            validation_start_date = "{{cookiecutter.validation_start_date}}",
            validation_end_date = "{{cookiecutter.validation_end_date}}")
    # set pipeline level compute
    pipeline_job.settings.default_compute= os.getenv('compute_cluster')
    # set pipeline level datastore
    pipeline_job.settings.default_datastore=os.getenv('datastore_name')
    pipeline_job.settings.force_rerun=False
    job_run = ml_client.jobs.create_or_update(pipeline_job, experiment_name="{{cookiecutter.experiment_name}}")
    logger.info(job_run.name)

    
def create_env(env_name):
    try:
        env = Environment(
            name=env_name, #"pipeline-comp-env",
            description="Custom env: ODBC driver + required Python libs",
            build=BuildContext(path="../component_env", dockerfile_path='dockerfile'),  # context includes Dockerfile + conda_dependencies.yml
        )
        # Create or update the environment in Azure ML
        ml_client.environments.create_or_update(env)
    except Exception as e:
        logger.error(f"Error create pipeline component environment. Error: {e}")
        raise e

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