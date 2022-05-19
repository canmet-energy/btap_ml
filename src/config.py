import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
import s3fs
import yaml
from pydantic import AnyHttpUrl, BaseModel, BaseSettings, Field, SecretStr

logger = logging.getLogger(__name__)


class AppConfig(BaseModel):
    """Application configuration."""
    # Paths to all BTAP calls within the docker container
    DOCKER_INPUT_PATH: str = '/home/btap_ml/input/'
    DOCKER_OUTPUT_PATH: str = '/home/btap_ml/output/'
    # Bucket prefix to be used as part of the run folder being created for training and running
    TRAIN_BUCKET_NAME: str = 'training_model_'
    RUN_BUCKET_NAME: str = 'running_model_'
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
    # Parent level key within input_config.yml
    RANDOM_SEED: str = ':random_seed'
    WEATHER_KEY: str = ':epw_file'
    BUILDING_PARAM_FILES: str = ':building_param_files'
    VAL_BUILDING_PARAM_FILES: str = ':val_building_param_files'
    ENERGY_PARAM_FILES: str = ':energy_param_files'
    VAL_ENERGY_PARAM_FILES: str = ':val_energy_param_files'
    ESTIMATOR_TYPE: str = ':estimator_type'
    PARAM_SEARCH: str = ':param_search'
    FEATURES_FILE: str = ':features_file'
    TRAINED_MODEL_FILE: str = ':trained_model_file'
    OHE_FILE: str = ':ohe_file'
    SCALER_X_FILE: str = ':scaler_X_file'
    SCALER_Y_FILE: str = ':scaler_y_file'
    BUILDING_BATCH_PATH: str = ':batch_building_inputs'
    SIMULATION_START_DATE: str = ':simulation_start_date'
    SIMULATION_END_DATE: str = ':simulation_end_date'
    # Specify any static filename keys
    PREPROCESSING_FILENAME: str = 'preprocessing'
    FEATURE_SELECTION_FILENAME: str = 'feature_selection'
    TRAINED_MODEL_FILENAME: str = 'trained_model.h5'
    SCALERX_FILENAME: str = 'scaler_X.pkl'
    SCALERY_FILENAME: str = 'scaler_y.pkl'
    TRAINING_RESULTS_FILENAME: str = 'training_results'
    RUNNING_DAILY_RESULTS_FILENAME: str = 'daily_energy_predictions.csv'
    RUNNING_AGGREGATED_RESULTS_FILENAME: str = 'aggregated_energy_predictions.csv'

# There's a JSON file available with required credentials in it
def json_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    """Load credentials from the file system."""
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

        # Where to load JSON settings from
        minio_tenant: str = 'standard'
        json_settings_path: Path = Path(f'/vault/secrets/minio-{minio_tenant}-tenant-1.json')

        # Ignore extra values present in the JSON data
        extra = 'ignore'


def establish_s3_connection(endpoint_url: str, access_key: str, secret_key: SecretStr) -> s3fs.S3FileSystem:
    """Used to create a connection to an S3 data store.

    Args:
        endpoint_url: The URL for the data store.
        access_key: The access key used to access the data store.
        secret_key: The secret key used to access the data store.

    Returns:
        An s3fs file system object.
    """
    logger.info("Establishing connection to S3 server: %s", endpoint_url)
    s3 = s3fs.S3FileSystem(
        anon=False,
        key=access_key,
        secret=secret_key.get_secret_value(),
        client_kwargs={
            'endpoint_url': endpoint_url,
        },
    )
    # Wait longer for connections to blob storage to get established
    s3.connect_timeout = 60

    return s3

def get_config(config_file: str):
    """Load the specified configuration file from blob storage.

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

def access_minio(path: str, operation: str, data: Union[str, pd.DataFrame]):
    """
    Used to read and write to minio.

    Args:
        tenant: default value is standard
        bucket: nrcan-btap
        path: file path where data is to be read from or written to
        data: for write operation, it contains the data to be written to minio

    Returns:
       Dataframe containing the data downladed from minio is returned for read operation and for write operation , null value is returned.
    """

    logger.info("%s minio data at %s", operation, path)

    # Get settings for the environment
    settings = Settings()

    # Establish S3 connection
    s3 = establish_s3_connection(settings.MINIO_URL,
                                 settings.MINIO_ACCESS_KEY,
                                 settings.MINIO_SECRET_KEY)

    logger.info("s3 connection %s", s3)

    # s3fs doesn't seem to like Path objects, so use a posix path string for operations
    full_posix_path = settings.NAMESPACE.joinpath(path).as_posix()

    if operation == 'read':
        if 'csv' in path:
            data = pd.read_csv(s3.open(full_posix_path, mode='rb'))
        elif 'xls' in path:
            data = pd.read_excel(s3.open(full_posix_path, mode='rb'))
        else:
            data = s3.open(full_posix_path, mode='rb')
    else:
        # If no data is pathed, pass the opened file to write directly to
        if data == '':
            data = s3.open(full_posix_path, mode='wb')
        # If data is passed, directly write the data to the file
        else:
            with s3.open(full_posix_path, 'wb') as f:
                f.write(data)
                data = ''

    return data

def create_directory(path: str) -> None:
    """
    Given a path, create the directory if it exists

    Args:
        path: directory to be created
    """
    if not os.path.isdir(path):
        os.mkdir(path)
