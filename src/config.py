import s3fs
import json
import pandas as pd


def access_minio(tenant,bucket,path,operation,data):
    """
    Used to read and write to minio.

    Args:
        tenant: default value is standard
        bucket: nrcan-btap
        path: file path where data is to be read from or written to
        data: for write operation, it contains the data to be written to minio

    Returns:
       Dataframe containing the data downladed from minio is returned  
       for read operation and for write operation , null value is returned. 
    """
    with open(f'/vault/secrets/minio-{tenant}-tenant-1.json') as f:
        creds = json.load(f)
        
    minio_url = creds['MINIO_URL']
    access_key=creds['MINIO_ACCESS_KEY']
    secret_key=creds['MINIO_SECRET_KEY']

    # Establish S3 connection
    s3 = s3fs.S3FileSystem(
        anon=False,
        key=access_key[0],
        secret=secret_key,
        #use_ssl=False, # Used if Minio is getting SSL verification errors.
        client_kwargs={'endpoint_url': minio_url,
                        #'verify':False
        })

    if operation == 'read':
            if 'csv' in path :
                data = pd.read_csv(s3.open('{}/{}'.format(bucket, path), mode='rb'))
            elif 'xls' in path:
                data =  pd.read_excel(s3.open('{}/{}'.format(bucket, path), mode='rb'))
            else:
                 data = s3.open('{}/{}'.format(bucket, path), mode='rb')
    else:
        with s3.open('{}/{}'.format(bucket, path), 'wb') as f:
            f.write(data)
            data = ''
       
    return data
