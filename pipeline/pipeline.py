import kfp
from kfp import dsl
from kfp.components import func_to_container_op
from kfp.components import load_component_from_file

@dsl.pipeline(name='Btap Pipeline', 
              description='MLP employed for Total energy consumed regression problem',
              )

#define your pipeline
def btap_pipeline(minio_tenant:str,bucket:str,build_params:str,energy_hour:str,weather:str,build_params_val:str,energy_hour_val:str,
                  output_path:str,featureestimator:str,featureoutput_path:str,param_search:str):
#     tenant ='minio_tenant'
    # Loads the yaml manifest for each component
    preprocess = load_component_from_file('yaml/preprocessing.yaml')
    feature_selection = load_component_from_file('yaml/feature_selection.yaml')
    predict = load_component_from_file('yaml/predict.yaml')
    
    preprocess_ = preprocess(
                             tenant=minio_tenant,
                             bucket=bucket,
                             in_build_params=build_params,
                             in_hour=energy_hour,
                             in_weather=weather,
                             in_build_params_val=build_params_val,
                             in_hour_val=energy_hour_val,
                             output_path=output_path
                             
                            )
 
    feature_selection_ = feature_selection(tenant=minio_tenant,
                                           bucket=bucket,
                                           in_obj_name=preprocess_.output,
                                           estimator_type=featureestimator,
                                           output_path=featureoutput_path)
    
    predict_ = predict(tenant=minio_tenant,
                       bucket=bucket,
                       in_obj_name=preprocess_.output,
                       features=feature_selection_.output,
                       param_search=param_search)
 
    
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(btap_pipeline, 'pipeline.yaml')
    print(f"Exported pipeline definition to pipeline.yaml")
    
  