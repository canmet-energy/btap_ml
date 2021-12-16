With data loaded to blob storage, run ``proprocessing.py`` to clean and split the data into test and train sets::

    $ python3 preprocessing.py --in_build_params input_data/output_elec_2021-11-05.xlsx --in_hour input_data/total_hourly_res_elec_2021-11-05.csv --in_weather input_data/montreal_epw.csv --output_path output_data/preprocessing_out --in_build_params_gas input_data/output_gas_2021-11-05.xlsx --in_hour_gas input_data/total_hourly_res_gas_2021-11-05.csv

.. note::

   Update the input parameters to point to the appropriate input files and ouput_path in minio. Only the filenames
   would need to be changed. Check :py:mod:`preprocessing` for full detail description of each parameter.

Aside from supporting optional different files for electricity and gas, if you have mutliple input files they
need to be manually combined before passing them as input. If you only have a single input file, provide it as
the value to ``--in_build_params``.

What this does
^^^^^^^^^^^^^^

During preprocessing, the hourly energy consumption file is transposed such that each ``datapoint_id`` has 8760 rows
(365 * 24). Hence, for a simulation run containing 5000 ``datapoint_id``, there would be 5000 * 8760 rows which would
be 43.8 million rows. In order to avoid the preprocessing stage to become computationally expensive due to large
datapoints created, the transposed hourly energy is is aggregated to daily energy for each datapoint_id. Similarly,
the weather information is aggregated from hourly to daily, so that it can be merged with the hourly energy file.
In essensece, the simulation I/O file(s), weather file and the hourly energy consumption file(s) are all merged to one
dataframe which then splitted for training and testing purposes.

.. note::

    The total daily energy computed for each ``datapoint_id`` is converted to Mega Joules per meter square (mj_per_m_sq),
    which is derived by converting the total energy used provided in the simulation I/O file to Mega Joules and then
    dividing the result by the building floor area.

In the case where the validation set is not provided, the dataset provided is splitted into 70% training set, 20% test
set and 10% validation set. However, when validation set is provided, the dataset is splitted only into 80% training
set and 20% test set. It should be noted that the data splitting is performed using GroupShuffleSplit function which
ensures that all data instances for a datapoint_id are included in the respective data set they are split into which
implies a datapoint_id used for training would not have any instance in the test set.


.. note:: The json file created at the end of preprocessing has the following keys:
    'features': This is all the features after data cleaning and preprocessing. Note that these are not the final features used for modelling. 
    'y_train': This contains the y_train dataset. 
    'X_train': This contains the X_train dataset. 
    'X_test': This contains the X_test dataset. 
    'y_test': This contains the y_test dataset. 
    'y_test_complete': This is similar to the y_test dataset, but has the additional column datapoint_id which would be used needed in creating the final output file after predict.py is run. 
    'X_validate': This contains the X_validate dataset. 
    'y_validate':  This contains the y_validate dataset. 
    'y_validate_complete': This is similar to the y_validate dataset, but has the additional column datapoint_id which would be used needed in creating the final output file after predict.py is run.