import json

import pandas as pd
import s3fs


def establish_s3_connection(endpoint_url: str, access_key: str, secret_key: str) -> s3fs.S3FileSystem:
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
        secret=secret_key,
        client_kwargs={
            'endpoint_url': endpoint_url,
        }
    )

    return s3


def access_minio(tenant, bucket, path, operation, data):
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
    with open(f'/vault/secrets/minio-{tenant}-tenant-1.json') as f:
        creds = json.load(f)

    minio_url = creds['MINIO_URL']
    access_key = creds['MINIO_ACCESS_KEY'],
    secret_key = creds['MINIO_SECRET_KEY']

    # Establish S3 connection
    s3 = establish_s3_connection(minio_url, access_key, secret_key)

    if operation == 'read':
        if 'csv' in path:
            data = pd.read_csv(s3.open('{}/{}'.format(bucket, path), mode='rb'))
        elif 'xls' in path:
            data = pd.read_excel(s3.open('{}/{}'.format(bucket, path), mode='rb'))
        else:
            data = s3.open('{}/{}'.format(bucket, path), mode='rb')
    else:
        with s3.open('{}/{}'.format(bucket, path), 'wb') as f:
            f.write(data)
            data = ''

    return data
