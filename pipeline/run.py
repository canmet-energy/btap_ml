import json
import os

import kfp

import pipeline as pl

build_param =""
client = kfp.Client()
result = client.create_run_from_pipeline_func(
    pl.btap_pipeline,
    arguments={
               "energy_hour":"input_data/total_hourly_res_elec_2021-11-05.csv",
               "build_params":"input_data/output_elec_2021-11-05.xlsx",
               "weather":"input_data/montreal_epw.csv",
               "output_path":"output_data/preprocessing_out",
               "energy_hour_val":"",
               "build_params_val":"",
               "energy_hour_gas":"input_data/total_hourly_res_gas_2021-11-05.csv",
               "build_params_gas":"input_data/output_gas_2021-11-05.xlsx",
               "featureestimator":"lasso",
               "featureoutput_path":"output_data/feature_out",
               "param_search":"no"


              }
    )
