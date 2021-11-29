import kfp
from kfp import dsl
from kfp.components import func_to_container_op
from kfp.components import load_component_from_file
import config as acm
import pandas as pd
from minio import Minio


@dsl.pipeline(name='Btap Pipeline',
              description='MLP employed for Total energy consumed regression problem',
              )

#define your pipeline
# def btap_pipeline(energy_hour,build_params,weather,output_path,energy_hour_val,
#                   build_params_val,energy_hour_gas,build_params_gas,featureestimator,featureoutput_path,param_search):
def btap_pipeline(  build_params="input_data/output_2021-10-04.xlsx",
                    energy_hour="input_data/total_hourly_res_2021-10-04.csv",
                    weather="input_data/montreal_epw.csv",
                    build_params_val="input_data/output.xlsx",
                    energy_hour_val="input_data/total_hourly_res.csv",
                    output_path="output_data/preprocessing_out",
                    featureoutput_path="output_data/feature_out",
                    featureestimator="lasso",
                    param_search="no",
                    build_params_gas= 'input_data/output_gas_2021-11-05.xlsx', 
                    energy_hour_gas ='input_data/total_hourly_res_gas_2021-11-05.csv'):
                        
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
    preprocess_output_ref = preprocess_.outputs['Output']
    
    feature_selection_ = feature_selection(
                                           in_obj_name=preprocess_output_ref,
                                           estimator_type=featureestimator,
                                           output_path=featureoutput_path)
    feature_output_ref = feature_selection_.outputs['Output']
    
    predict_ = predict(
                       in_obj_name=preprocess_output_ref,
                       features=feature_output_ref,
                       param_search=param_search)


if __name__ == '__main__':
    experiment_yaml_zip = 'pipeline.yaml'
    kfp.compiler.Compiler().compile(btap_pipeline, experiment_yaml_zip)
    print(f"Exported pipeline definition to {experiment_yaml_zip}")
