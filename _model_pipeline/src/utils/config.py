"""
Configuration management for Azure services.
This module handles loading secrets from Azure Key Vault using Managed Identity and provides configurations for Synapse and MLClient.
Creates ml_client and credential for Azure ML operations.
Loads model features from a JSON file.

"""

import json
import os
 
from azure.ai.ml import MLClient
from azure.keyvault.secrets import SecretClient
from azure.identity import ManagedIdentityCredential
 
from .setup_logger import get_logger
logger = get_logger(__name__)


class Config:
    def __init__(self, Key_vault_name: str):
        # Env variables
        self.key_vault_name = Key_vault_name
        self.uami_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
        logger.debug(f"Environment variables loaded - KEY_VAULT_NAME set: {bool(self.key_vault_name)}")
       
        if not self.key_vault_name:
            logger.critical("KEY_VAULT_NAME not set in environment.")
            raise ValueError("KEY_VAULT_NAME not set in environment.")
 
        # Init Azure Key Vault client once
        if self.uami_client_id:
            self._credential = ManagedIdentityCredential(client_id=self.uami_client_id)
        else:
            raise RuntimeError("DEFAULT_IDENTITY_CLIENT_ID is not set. Cannot use UAMI.")
        self._secret_client = SecretClient(
                                    vault_url=f"https://{self.key_vault_name}.vault.azure.net",
                                    credential=self._credential
                                )
        logger.info("Azure Key Vault client initialized successfully.")
 
        # Load Required API Configurations from JSON secret
        try:
            api_config_json = self._get_secret("API-Config")  # stored as JSON in KV
            api_config = json.loads(api_config_json)
            
            self.synapse_server = api_config.get("SYNAPSE_SERVER")
            self.synapse_db = api_config.get("SYNAPSE_DB")      
            self.synapse_lookup_schema = api_config.get("SYNAPSE_LOOKUP_SCHEMA")
            self.synapse_user = api_config.get('SYNAPSE_USER')
            self.synapse_password = api_config.get('SYNAPSE_PASSWORD')
            
            logger.info("Synapse configuration loaded from Key Vault.")
        except Exception as e:
            logger.critical(f"Failed to load Synapse configuration: {e}")
            raise RuntimeError(f"Failed to load Synapse configuration: {e}")
   
    def _get_secret(self, secret_name: str) -> str:
        value = self._secret_client.get_secret(secret_name).value
        logger.debug(f"Secret retrieved successfully- (length={len(value)})")
        return value


def get_ml_client() -> tuple[ManagedIdentityCredential, MLClient]:
    """Authenticate using Managed Identity and return an MLClient instance along with its credential."""
    
    SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
    RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
    WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")

    uami_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
    if not uami_client_id:
        raise RuntimeError("DEFAULT_IDENTITY_CLIENT_ID is not set. Cannot use fallback UAMI.")

    credential = ManagedIdentityCredential(client_id=uami_client_id)
    credential.get_token("https://management.azure.com/.default")
    logger.info("ManagedIdentityCredential succeeded.")

    return credential, MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)


def get_model_feature(model_type) -> list[str]:
    """
    Load model features from features.json file inside utils folder.
    Returns a list of features in lowercase.
    """
    base_dir = os.path.dirname(__file__)
    features_file = os.path.join(base_dir, f"{model_type}_features.json")

    with open(features_file, "r") as fp:
        features = [f.lower() for f in json.load(fp).get("features", [])]

    if not features:
        raise RuntimeError("Input features cannot be empty list.")

    return features

def load_env():
    """Load all environment variables"""
    SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
    RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
    WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
    KEY_VAULT_NAME = os.getenv("KEY_VAULT_NAME")
    RAW_STORAGE_ACCOUNT = os.getenv("RAW_STORAGE_ACCOUNT")
    ML_STORAGE_ACCOUNT = os.getenv("ML_STORAGE_ACCOUNT")
    ML_CONTAINER = os.getenv("ML_CONTAINER")
    RAW_CONTAINER = os.getenv("RAW_CONTAINER")
    DATASTORE_NAME = os.getenv("DATASTORE_NAME")
    MODEL_TYPE = os.getenv("MODEL_TYPE")
    if MODEL_TYPE=='btd_model':
        TARGET_VARIABLE = 'btd_flag'
    else:
        TARGET_VARIABLE = 'appointment_set'
    return SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, KEY_VAULT_NAME, RAW_STORAGE_ACCOUNT, ML_STORAGE_ACCOUNT, RAW_CONTAINER, ML_CONTAINER, DATASTORE_NAME, MODEL_TYPE, TARGET_VARIABLE
    

