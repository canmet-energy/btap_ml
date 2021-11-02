import os
import kfp
import json
import pipeline as pl

build_param =""
client = kfp.Client()

client._context_setting['namespace'] ='nrcan-btap'

        
        
minio_tenant = "standard"  # probably can leave this as is

with open(f'/vault/secrets/minio-{minio_tenant}-tenant-1.json') as f:
        creds = json.load(f)
        minio_url = creds['MINIO_URL']


minio_url = creds['MINIO_URL']
minio_access_key = creds['MINIO_ACCESS_KEY']
minio_secret_key = creds['MINIO_SECRET_KEY']

result = client.create_run_from_pipeline_func(
    pl.btap_pipeline,
    arguments={
             "minio_tenant" : "standard",
            "bucket":"nrcan-btap",
            "build_params":"input_data/output_2021-10-04.xlsx",
            "energy_hour":"input_data/total_hourly_res_2021-10-04.csv",
            "weather":"input_data/montreal_epw.csv",
            "build_params_val":"input_data/output.xlsx",
            "energy_hour_val":"input_data/total_hourly_res.csv",
            "output_path":"output_data/preprocessing_out",
            "featureestimator":"output_data/feature_out",
            "featureoutput_path": "lasso",               
            "param_search":"no"
              }
    )
