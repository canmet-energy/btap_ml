import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import s3fs
from pydantic import AnyHttpUrl, BaseSettings, SecretStr


# There's a JSON file available with required credentials in it
def json_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    return json.loads(settings.__config__.json_settings_path.read_text())

class Settings(BaseSettings):
    MINIO_URL: AnyHttpUrl
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: SecretStr

    NAMESPACE: Path = Path('nrcan-btap')

    class Config:
        minio_tenant: str = 'standard'
        json_settings_path: Path = Path(f'/vault/secrets/minio-{minio_tenant}-tenant-1.json')
        # Ignore extra values present in the JSON data
        extra = 'ignore'

        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            return (init_settings, json_config_settings_source, env_settings, file_secret_settings)

settings = Settings()


def establish_s3_connection(endpoint_url: str, access_key: str, secret_key: SecretStr) -> s3fs.S3FileSystem:
    """Used to create a connection to an S3 data store.

    Args:
        endpoint_url: The URL for the data store.
        access_key: The access key used to access the data store.
        secret_key: The secret key used to access the data store.

    Returns:
        An s3fs file system object.
    """
    s3 = s3fs.S3FileSystem(
        anon=False,
        key=access_key,
        secret=secret_key.get_secret_value(),
        client_kwargs={
            'endpoint_url': endpoint_url,
        }
    )

    return s3


def access_minio(path, operation, data):
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
    # Establish S3 connection
    s3 = establish_s3_connection(settings.MINIO_URL,
                                 settings.MINIO_ACCESS_KEY,
                                 settings.MINIO_SECRET_KEY)

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
