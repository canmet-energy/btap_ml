import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Union

import yaml
from pydantic import AnyHttpUrl, BaseModel, BaseSettings, Field, SecretStr

logger = logging.getLogger(__name__)


class AppConfig(BaseModel):
    """Application configuration."""
    # Paths to all BTAP calls within the docker container
    DOCKER_INPUT_PATH: str = 'C:/Users/dlau/Documents/btap_ml/input/'
    DOCKER_OUTPUT_PATH: str = 'C:/Users/dlau/Documents/btap_ml/output/'
    DOCKER_SRC_PATH: str = 'src/' #'../src/' # switch to 'src/' for energy
    # Bucket prefix to be used as part of the run folder being created for training and running
    TRAIN_BUCKET_NAME: str = 'training_model_'
    RUN_BUCKET_NAME: str = 'running_model_'
    # Select the type of model either 'mlp' for Multilayer Perceptron or 'rf' for Random Forest Regressor
    SELECTED_MODEL_TYPE: str = 'selected_model_type'
    # Shared parameters to denote the type of training or running being performed
    ENERGY: str = 'energy'
    COSTING: str = 'costing'
    # Prefix names used for costing and energy
    ENERGY_PREFIX: str = 'energy_'
    COSTING_PREFIX: str = 'costing_'
    # Shared parameters to denote the type of model
    MULTILAYER_PERCEPTRON: str = 'mlp'
    RANDOM_FOREST: str = 'rf'
    # Bucket used to store weather data
    WEATHER_BUCKET_NAME: str = 'weather'
    # Bucket used to store building/energy preprocessing data
    PREPROCESSING_BUCKET_NAME: str = 'preprocessing'
    # Bucket used to store feature selection data
    FEATURE_SELECTION_BUCKET_NAME: str = 'feature_selection'
    # Bucket used to store model training data
    TRAINING_BUCKET_NAME: str = 'model_training'
    # URL where weather files are stored
    WEATHER_DATA_STORE: AnyHttpUrl = 'https://raw.githubusercontent.com/NREL/openstudio-standards/nrcan/data/weather/'
    # File location of the file listing columns to ignore for costing predictions
    COSTING_COLUMNS_FILE: str = 'column_files/costing_columns_to_ignore.txt'
    # Parent level key within input_config.yml
    RANDOM_SEED: str = 'random_seed'
    BUILDING_PARAM_FILES: str = 'building_param_files'
    VAL_BUILDING_PARAM_FILE: str = 'val_building_param_file'
    ENERGY_PARAM_FILES: str = 'energy_param_files'
    VAL_ENERGY_PARAM_FILE: str = 'val_energy_param_file'
    ESTIMATOR_TYPE: str = 'estimator_type'
    PARAM_SEARCH: str = 'param_search'
    FEATURES_FILE: str = 'features_file'
    TRAINED_MODEL_FILE: str = 'trained_model_file'
    OHE_FILE: str = 'ohe_file'
    CLEANED_COLUMNS_FILE: str = 'cleaned_columns_file'
    SCALER_X_FILE: str = 'scaler_X_file'
    SCALER_Y_FILE: str = 'scaler_y_file'
    BUILDING_BATCH_PATH: str = 'batch_building_inputs'
    SIMULATION_START_DATE: str = 'simulation_start_date'
    SIMULATION_END_DATE: str = 'simulation_end_date'
    USE_UPDATED_MODEL: str = 'use_updated_model'
    USE_DROPOUT: str = 'use_dropout'
    # Specify any static filename keys
    PREPROCESSING_FILENAME: str = 'preprocessing'
    FEATURE_SELECTION_FILENAME: str = 'feature_selection'
    TRAINED_MODEL_FILENAME_MLP: str = 'trained_model_mlp.h5'
    TRAINED_MODEL_FILENAME_RF: str = 'trained_model_rf.joblib'
    TRAINED_MODEL_FILENAME_GB: str = 'trained_model_gb.joblib'
    SCALERX_FILENAME: str = 'scaler_X.pkl'
    SCALERY_FILENAME: str = 'scaler_y.pkl'
    OHE_FILENAME: str = 'ohe.pkl'
    CLEANED_COLUMNS_FILENAME: str = 'cleaned_columns.json'
    TRAINING_RESULTS_FILENAME: str = 'training_results'
    RUNNING_DAILY_RESULTS_FILENAME: str = 'daily_energy_predictions.csv'
    RUNNING_AGGREGATED_RESULTS_FILENAME: str = 'aggregated_energy_predictions.csv'
    RUNNING_COSTING_RESULTS_FILENAME: str = 'costing_predictions.csv'

# There's a JSON file available with required credentials in it
def json_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    """
    Load credentials from the file system. No longer used but retained in case
    needed in the future
    """
    # Maximum number of attempts befaore failing
    max_attempts = 3
    attempt_interval = 5

    creds_file = settings.__config__.json_settings_path

    for attempt in range(max_attempts):
        if creds_file.exists():
            logger.info("Credentials file found. Reading from %s", creds_file)
            return json.loads(creds_file.read_text())
        else:
            logger.warn("Credentials file %s not found. Waiting %s seconds to try again.", creds_file, attempt_interval)
            time.sleep(attempt_interval)

    # Wasn't able to read the credentials. Error out.
    logger.error("Unable to read storage credentials from %s", creds_file)
    sys.exit(1)


class Settings(BaseSettings):
    """Application settings. All of these can be set by the environment to override anything coded here."""
    # Set up application specific information
    APP_CONFIG: AppConfig = AppConfig()
    class Config:
        # Prefix our variables to avoid collisions with other programs
        env_prefix = 'BTAP_'

        # Ignore extra values present in the JSON data
        extra = 'ignore'

def get_config(config_file: str):
    """Load the specified configuration file.

    Args:
        config_file: Path to the config file relative to the default bucket.

    Returns:
        Dictionary of configuration information.
    """
    # Create a path to the config from the namespace
    config_file_path = Path(config_file)
    with open(str(config_file_path), 'rb') as outfile:
        contents = yaml.safe_load(outfile)
    return contents

def create_directory(path: str) -> None:
    """
    Given a path, create the directory if it exists

    Args:
        path: directory to be created
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
