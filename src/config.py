import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
import s3fs
from pydantic import AnyHttpUrl, BaseModel, BaseSettings, Field, SecretStr

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class AppConfig(BaseModel):
    """Application configuration."""
    # Bucket used to store weather data
    WEATHER_BUCKET_NAME: str = 'weather'
    # URL where weather files are stored
    WEATHER_DATA_STORE: AnyHttpUrl = 'https://raw.githubusercontent.com/NREL/openstudio-standards/nrcan/data/weather/'
    # Parent level key in the BTAP CLI config weather data is stored under.
    # The EPW file key will be under this.
    BUILDING_OPTS_KEY: str = ':building_options'
# There's a JSON file available with required credentials in it
def json_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    return json.loads(settings.__config__.json_settings_path.read_text())
class Settings(BaseSettings):
    """Application settings. All of these can be set by the environment to override anything coded here."""
    # Set up application specific information
    APP_CONFIG: AppConfig = AppConfig()

    MINIO_URL: AnyHttpUrl = Field(..., env='MINIO_URL')
    MINIO_ACCESS_KEY: str = Field(..., env='MINIO_ACCESS_KEY')
    MINIO_SECRET_KEY: SecretStr = Field(..., env='MINIO_SECRET_KEY')

    NAMESPACE: Path = Path('nrcan-btap')

    class Config:
        # Prefix our variables to avoid collisions with other programs
        env_prefix = 'BTAP_'

        # Where to load JSON settings from
        minio_tenant: str = 'standard'
        json_settings_path: Path = Path(f'/vault/secrets/minio-{minio_tenant}-tenant-1.json')

        # Ignore extra values present in the JSON data
        extra = 'ignore'

        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            return (init_settings, json_config_settings_source, env_settings, file_secret_settings)

settings = Settings()
logger.debug("Loaded the following settings: %s", settings)

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
        #config_kwargs={'connect_timeout': 10}
    )
    # Wait longer for connections to blob storage to get established
    s3.connect_timeout = 60

    return s3


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
    
    # Establish S3 connection
    s3 = establish_s3_connection(settings.MINIO_URL,
                                 settings.MINIO_ACCESS_KEY,
                                 settings.MINIO_SECRET_KEY)

    logger.info("%s s3 connection %s", s3)
    
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
        with s3.open(full_posix_path, 'wb') as f:
            f.write(data)
            data = ''

    return data