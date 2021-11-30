New Datafiles (Quick Run)
================================
In cases where there are significant changes in the building simulation I/O whereby new parameters are introducted (e.g if we were working with Electric energy sources and we decided to incorporate gas energy sources), then an hyperparameter search would be appropriate as described below:


1. Preprocessing
----------------------
To download data from minio, clean the data and split the data into test and train sets::

    python3 preprocessing.py --in_build_params input_data/output_elec_2021-11-05.xlsx --in_hour input_data/total_hourly_res_elec_2021-11-05.csv --in_weather input_data/montreal_epw.csv --output_path output_data/preprocessing_out --in_build_params_gas input_data/output_gas_2021-11-05.xlsx --in_hour_gas input_data/total_hourly_res_gas_2021-11-05.csv

.. note::

   Update the input parameters to point to the appropriate input files and ouput_path in minio. Only the filenames would need to be changed. Check **Usage** for full detail description of each parameter.

2. Feature Selection
----------------------
After preprocessing the data, the features to be used in building the surrogate model can be selected::

     python3 feature_selection.py --in_obj_name output_data/preprocessing_out --output_path output_data/feature_out --estimator_type lasso


3. Build Surrogate Model and Predict
--------------------------------------------
The output from step 1 and 2 are used as input into this stage by running::

    python3 predict.py --param_search yes --in_obj_name output_data/preprocessing_out --features output_data/feature_out --output_path output_data/predict_out


4. Download Results
--------------------------------------------
The results from the prediction is available in minio in the output_path specified in step 3.
