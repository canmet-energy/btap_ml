import os
import kfp
import json
import pipeline as pl

build_param =""
client = kfp.Client()
result = client.create_run_from_pipeline_func(
    pl.btap_pipeline,
    arguments={"tenant":"standard",
               "bucket":"nrcan-btap",
               "in_build_params":"input_data/output_2021-10-04.xlsx",
               "in_hour":"input_data/total_hourly_res_2021-10-04.csv",
               "in_weather":"input_data/montreal_epw.csv",
               "in_build_params_val":"input_data/output.xlsx",
               "in_hour_val":"input_data/total_hourly_res.csv",
               "output_path":"output_data/preprocessing_out",
               "featureoutput_path":"output_data/feature_out",
               "param_search":"no"
              }
    )
