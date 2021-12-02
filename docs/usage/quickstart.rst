Quickstart
==========

If you just want to run all of the steps from start to finish, here are the steps to run. You may be able to skip some
of these depending on if data has already been prepared.

Prepare weather data
--------------------

Gather required weather data and place it on blob storage::

   $ python prepare_weather.py input_data/sample-config.yml

Preprocessing
-------------

Clean the input data and split the it into test and train sets::

    python3 preprocessing.py --in_build_params input_data/output_elec_2021-11-05.xlsx --in_hour input_data/total_hourly_res_elec_2021-11-05.csv --in_weather input_data/montreal_epw.csv --output_path output_data/preprocessing_out --in_build_params_gas input_data/output_gas_2021-11-05.xlsx --in_hour_gas input_data/total_hourly_res_gas_2021-11-05.csv

.. note::

   Update the input parameters to point to the appropriate input files and ouput_path in minio. Only the filenames
   would need to be changed. Check :py:mod:`preprocessing` for full detail description of each parameter.

Feature Selection
------------------

After preprocessing the data, the features to be used in building the surrogate model can be selected::

     python3 feature_selection.py --in_obj_name output_data/preprocessing_out --output_path output_data/feature_out --estimator_type lasso


Build Surrogate Model and Predict
----------------------------------

The output from proprocessing and feature selection are used as input into this stage by running::

    python3 predict.py --param_search no --in_obj_name output_data/preprocessing_out --features output_data/feature_out --output_path output_data/predict_out


Download Results
----------------

The results from the prediction is available in minio in the output_path specified in the build step.
