Build and Run Surrogate Model
================================
Ensure the input files are in the input_data directories in minio then follow the following 4 steps:

Preprocessing
----------------------
To download data from minio, clean the data and split the data into test and train sets.

python3 preprocessing.py --in_build_params input_data/output_elec_2021-11-05.xlsx --in_hour input_data/total_hourly_res_elec_2021-11-05.csv --in_weather input_data/montreal_epw.csv --output_path output_data/preprocessing_out --in_build_params_gas input_data/output_gas_2021-11-05.xlsx --in_hour_gas input_data/total_hourly_res_gas_2021-11-05.csv
During preprocessing, the hourly energy consumption file is transposed such that each

Hyperparameter Search
----------------------

Hyperparameter Search
----------------------

Hyperparameter Search
----------------------




You can use the tensorboard dashboard to inspect the performance of the model.

1. Open ``notebooks/tensorboard.ipynb``.
2. Run all the contents of the notebook.
3. Navigate to the appropriate port in a web browser.

.. note::

   The tensorboard opens on a random port inside your notebook container. The URL looks like
   https://kubeflow.aaw.cloud.statcan.ca/notebook/nrcan-btap/<notebook_name>/proxy/<port>/.
