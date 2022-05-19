With the inputs specified through the command line or through ``input_config.py``,
run ``proprocessing.py`` to clean and split the data into test and train sets::

    python preprocessing.py input_config.py

.. note::

   Update the input parameters to point to the appropriate input files.
   Check :py:mod:`preprocessing` for full detail description of each parameter.

Aside from supporting optional different files for electricity and gas, if you have mutliple input files they
need to be manually combined before passing them as input.

What this does
^^^^^^^^^^^^^^

During preprocessing, the hourly energy consumption file is transposed such that each ``datapoint_id`` has 8760 rows
(365 * 24). Hence, for a simulation run containing 5000 ``datapoint_id`` values, there would be 5000 * 8760 rows which would
be 43.8 million rows. In order to avoid the preprocessing stage to become computationally expensive due to large
datapoints created, the transposed hourly energy file is aggregated to daily energy for each ``datapoint_id``. Similarly,
the weather information is aggregated from hourly to daily, so that it can be merged with the hourly energy file.
In essence, the simulation I/O file(s), weather file and the hourly energy consumption file(s) are all merged to one
dataframe which is then split for training and testing purposes (training set, testing set, validation set).

.. note::

    The total daily energy computed for each ``datapoint_id`` is converted to Megajoules per meter square (mj_per_m_sq),
    which is derived by converting the total energy used provided in the simulation I/O file to Megajoules and then
    dividing the result by the building floor area in meters.

In the case where the validation set is not provided, the dataset provided is split into a 70% training set, 20% test
set and 10% validation set. However, when a validation set is provided, the dataset is split only into a 80% training
set and 20% test set. It should be noted that the data splitting is performed using GroupShuffleSplit function which
ensures that all data instances for a ``datapoint_id`` are included in the respective data group they are split into which
implies that a ``datapoint_id`` used for training would not have any instance in the test set.

.. note::

    The json file created at the end of preprocessing has the following keys:

    * ``features``: This is all the features after data cleaning and preprocessing. Note that these are not the final features used for modelling.
    * ``y_train``: This contains the y_train dataset.
    * ``X_train``: This contains the X_train dataset.
    * ``X_test``: This contains the X_test dataset.
    * ``y_test``: This contains the y_test dataset.
    * ``y_test_complete``: This is similar to the y_test dataset, but has the additional column datapoint_id which would be used needed in creating the final output file after predict.py is run.
    * ``X_validate``: This contains the X_validate dataset.
    * ``y_validate``:  This contains the y_validate dataset.
    * ``y_validate_complete``: This is similar to the y_validate dataset, but has the additional column datapoint_id which would be used needed in creating the final output file after predict.py is run.
