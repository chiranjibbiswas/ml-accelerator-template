import os
import pandas as pd
import numpy as np
import logging
import yaml
import json
from typing import List
import joblib
from pathlib import Path

# Globals
g_model = None
g_logger = None
g_model_features = None
output_path = None


def get_features_from_mlmodel(mlmodel_path):
    """
    Parses an MLmodel file and extracts the feature names from its signature.

    Args:
        mlmodel_path (str): The full path to the MLmodel file.

    Returns:
        list: A list of feature names, or an empty list if not found.
    """
    feature_names = []
    
    if not os.path.exists(mlmodel_path):
        print(f"Error: The file {mlmodel_path} does not exist.")
        return feature_names

    try:
        with open(mlmodel_path, 'r') as f:
            mlmodel_content = yaml.safe_load(f)

        signature = mlmodel_content.get('signature')
        if signature and 'inputs' in signature:
            # The inputs are stored as a JSON string within the YAML
            inputs_json = signature['inputs']
            inputs_schema = json.loads(inputs_json)
            
            for column in inputs_schema:
                if 'name' in column:
                    feature_names.append(column['name'])
    
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON signature: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return feature_names


def init():
    """Initialize model once when the batch endpoint starts"""
    global g_model, g_logger, g_model_features, output_path

    g_logger = logging.getLogger("azureml")
    g_logger.setLevel(logging.INFO)

    output_path = os.environ["AZUREML_BI_OUTPUT_PATH"]
    model_dir = os.environ["AZUREML_MODEL_DIR"]
    model_rootdir = os.listdir(model_dir)[0]
    model_path = os.path.join(model_dir, model_rootdir)
    g_logger.info(f"Loading MLflow model from: {model_path}")
    model_file = os.path.join(model_path, 'model.pkl')
    g_logger.info(f"Loading MLflow model from: {model_path}")
    g_model_features = get_features_from_mlmodel(os.path.join(model_path, 'MLmodel'))
    g_logger.info(f"Total Model features: {len(g_model_features)}")
    # with open(model_file, "rb") as file:
    g_model = joblib.load(model_file)
    g_logger.info(f"Azure output path: {output_path}")

def load_data(file_path: str, features) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""
    try:
        data = pd.read_csv(file_path)
        data.columns = [col.lower() for col in data.columns]
        data = data[['leadid']+[c.lower() for c in features]].copy()
        data.columns = ['leadid'] + features   
        g_logger.info(f"Data loaded successfully from {file_path}, shape: {data.shape}")
        return data
    except Exception as e:
        g_logger.error(f"Error loading data from {file_path}: {e}")
        raise e


def run(mini_batch: List[str]):
    try:
        for file_path in mini_batch:
            g_logger.info(f"file_path: {file_path}")
            if not file_path.split('/')[-1].endswith('.csv'):
                continue
            data = load_data(file_path, g_model_features)
            pred = g_model.predict_proba(data[g_model_features])
            if isinstance(pred, np.ndarray):
                pred = pd.DataFrame(pred, columns=['btd_0_score', 'btd_1_score'])
            else:
                pred.columns = ['btd_0_score', 'btd_1_score']
            
            data[['btd_0_score', 'btd_1_score']] = pred
            output_file_name = Path(file_path).stem
            output_file_path = os.path.join(output_path, output_file_name + ".csv")

            data[['leadid', 'btd_0_score', 'btd_1_score']].to_csv(output_file_path, index=False)
            g_logger.info(f"Prediction have been saved to blob: {output_file_path}")
            
        return mini_batch
    except Exception as e:
        g_logger.error(f"Error while batch inference: {e}")
        raise e