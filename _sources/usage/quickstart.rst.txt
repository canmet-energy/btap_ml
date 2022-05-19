Quickstart
==========

If you just want to run all of the steps from training a model to using the model to obtain predictions, here are the steps to run. You may be able to skip some
of these depending on if data has already been prepared.

Each of the files can use an input configuration .yml file which contains all inputs to be used alongside descriptions of the inputs.
If used, only the input file, which will be referenced as input_config.yml, will be passed when calling the training process.
Each file can also use values passed to through command line arguements. All values except the config file are optional.
Thus, the config file must still be passed, but the CLI arguements will be given priority. A mix of command line and
config files can also be used in the case where certain parts of a process should be skipped.

.. note::

   All of the scripts for the project are in the ``src/`` folder of the repository. All CLI parameters can be viewed
   by calling the python file with ``--help`` arguement (Example: ``python train_model_pipeline.py --help``).

.. note::

   Every file can be run independently, but the recommended steps are to run one of the two pipelines below.

Training a model
----------------

The first step to perform is to generate a trained Machine Learning model to later be used for obtaining predictions on energy values.
Obtaining a trained model can be performed by calling the ``train_model_pipeline.py`` file and passing the appropriate arguements.
Assuming that only the ``input_config.yml`` file is used and located inside of the current directory, the arguement to pass is::

    python train_model_pipeline.py input_config.yml

This will process the specified weather file (outputting a .parquet file), preprocess all input files (outputting a .json file and .pkl file),
perform feature selection (outputting a .json file), and train the model (outputting a .h5 trained model, two .pkl files, a .json file, and a .csv file).

Getting model predictions
-------------------------

With a trained Machine Learning model in the form of a .h5 file, the model will be used to obtain the predicted daily Megajoules per meter squared
and the predicted annual Gigajoules per meter squared for a batch of specified building files within a specified directory.
Obtaining a trained model can be performed by calling the ``run_model.py`` file and passing the appropriate arguements.
Assuming that only the ``input_config.yml`` file is used and located inside of the current directory, the arguement to pass is::

    python train_model_pipeline.py input_config.yml

This will process the specified weather file to be used, preprocess all input data, and obtain predictions for all input data (outputting two .csv files
which contain the prediction outputs alongside identifiers for each building in the form building_file_name/building_index).

Prepare weather data
--------------------

Gather required weather data and place it on blob storage::

   $ python prepare_weather.py input_data/sample-lhs_2021-10-04.yml

Preprocessing
-------------

Clean the input data and split the it into test and train sets::

    python3 preprocessing.py --in_build_params input_data/output_elec_2021-11-05.xlsx --in_hour input_data/total_hourly_res_elec_2021-11-05.csv --in_weather weather/CAN_QC_Montreal-Trudeau.Intl.AP.716270_CWEC2016.epw.parquet --output_path output_data/preprocessing_out --in_build_params_gas input_data/output_gas_2021-11-05.xlsx --in_hour_gas input_data/total_hourly_res_gas_2021-11-05.csv

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
