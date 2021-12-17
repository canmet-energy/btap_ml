Kubeflow pipeline
=================

A kubeflow pipeline exists to run the training and modelling steps automatically. To run the pipleline the docker
image needs to be built.  Due to restrictions on which image registries are accepted on AAW, for this project images
were built on `STATCAN AAW contrib <https://github.com/StatCan/aaw-contrib-containers>`_. Other platforms may allow
to build the docker image on any external registry.

.. note::

   - You will need to get the Docker image digest SHA or tag which is to be updated in the yaml files, everytime the
     image is rebuilt and a new reference is created.
   - The source code and yaml for the pipleline is located in the pipeline folder.


Check the arguements in the pipeline.py file are correct

- build_params: The minio location and filename the building simulation I/O file. This would be the path for the electric hourly file if it exist.
- energy_hour: The minio location and filename for the hourly energy consumption file is located. This would be the path for the electric hourly file if it exist.
- weather: The minio location and filename for the converted  weather file to be read.
- build_params_val: The minio location and filename for the building simulation I/O file for the validation set, if it exist.
- energy_hour_val: The minio location and filename for the hourly energy consumption file for the validation set, if it exist.
- output_path: The minio location and filename where the output file should be written.
- featureoutput_path:  The minio location and filename where the feature selection output file should be written.
- featureestimator: The feature selection estimator type, default value is 'lasso'
- param_search: The values are 'yes' or 'no'
- build_params_gas: The minio location and filename the building simulation I/O file. This would be the path for the gas hourly file if it exist.
- energy_hour_gas: The minio location and filename for the hourly energy consumption file is located. This would be the path for the gas hourly file if it exist.
- predictoutput_path:  The minio location and filename where the predict output file should be written.


After ensuring the paths specified in the pipeline.py is verifed::

   python3 pipeline.py
   python3 upload-pipeline.py

Monitor the pipeline run from `Recent Pipeline Runs on AAW <https://kubeflow.aaw.cloud.statcan.ca/?ns=nrcan-btap>`_   and once completed successfullly, the respective output files will be stored on minio. Alternatively, click on `BTAP ML <https://kubeflow.aaw.cloud.statcan.ca/_/pipeline/#/experiments>`_ , then click on the desired run and check its log to monitor the pipeline run.


.. note::
   Ensure you are in nrcan-btap namespace when you check the links above. 
