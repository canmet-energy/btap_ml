#import config as acm
import time
from pathlib import Path

import kfp
import pandas as pd
from kfp import components, dsl
from kfp.components import func_to_container_op, load_component_from_file
from minio import Minio


@dsl.pipeline(name='Btap Pipeline',
              description='MLP employed for Total energy consumed regression problem',
              )

#define your pipeline
# def btap_pipeline(energy_hour,build_params,weather,output_path,energy_hour_val,
#                   build_params_val,energy_hour_gas,build_params_gas,featureestimator,featureoutput_path,param_search):
def btap_pipeline(
                    build_params="input_data/output_elec_2021-11-05.xlsx",
                    energy_hour="input_data/total_hourly_res_elec_2021-11-05.csv",
                    weather="weather/CAN_QC_Montreal-Trudeau.Intl.AP.716270_CWEC2016.epw.parquet",
                    build_params_val="",
                    energy_hour_val="",
                    output_path="output_data/preprocessing_out",
                    featureoutput_path="output_data/feature_out",
                    featureestimator="lasso",
                    param_search="no",
                    #build_params_gas= 'input_data/output_gas_2021-11-05.xlsx',
                    #energy_hour_gas ='input_data/total_hourly_res_gas_2021-11-05.csv',
                    build_params_gas= '',
                    energy_hour_gas ='',
                    predictoutput_path="output_data/predict_out",
        ):

    """
    TODO: UPDATE PARAMETERS
        config_file: str,
         output_path: str="",
         random_seed: int=7,
        # Weather
         weather_file: str="",
         skip_weather_generation: bool=False,
        # Preprocessing
         hourly_energy_electric_file: str="",
         building_params_electric_file: str="",
         val_hourly_energy_file: str="",
         val_building_params_file: str="",
         hourly_energy_gas_file: str="",
         building_params_gas_file: str="",
         skip_file_preprocessing: bool="",
        # Feature selection
         preprocessed_data_file: str="",
         estimator_type: str="",
         skip_feature_selection: bool="",
        # Training
         selected_features_file: str="",
         perform_param_search: str="",
         skip_model_training: bool="",
    """

    # Loads the yaml manifest for each component
    preprocess = load_component_from_file('yaml/preprocessing.yml')
    feature_selection = load_component_from_file('yaml/feature_selection.yml')
    predict = load_component_from_file('yaml/predict.yml')

    preprocess_ = preprocess(

                             in_hour=energy_hour,
                             in_build_params=build_params,
                             in_weather=weather,
                             output_path=output_path,
                             in_hour_val=energy_hour_val,
                             in_build_params_val=build_params_val,
                             in_hour_gas=energy_hour_gas,
                             in_build_params_gas =build_params_gas,


                            )

    preprocess_.set_memory_request('16Gi').set_memory_limit('32Gi')


    #preprocess_output_ref = preprocess_.output

    feature_selection_ = feature_selection(
                                           in_obj_name=output_path,
                                           estimator_type=featureestimator,
                                           output_path=featureoutput_path)
    feature_selection_.set_memory_request('16Gi').set_memory_limit('32Gi')
    #feature_selection_.set_cpu_request('1').set_cpu_limit('1')

    #feature_output_ref = feature_selection_.output
    feature_selection_.after(preprocess_)

    predict_ = predict(
                       in_obj_name=output_path,
                       features=featureoutput_path,
                       param_search=param_search,
                       output_path=predictoutput_path )
    predict_.set_memory_request('16Gi').set_memory_limit('32Gi')
    #predict_.set_cpu_request('1').set_cpu_limit('1')
    predict_.after(feature_selection_)


if __name__ == '__main__':
    experiment_yaml_zip = 'pipeline.yaml'
    kfp.compiler.Compiler().compile(btap_pipeline, experiment_yaml_zip)
    print(f"Exported pipeline definition to {experiment_yaml_zip}")
