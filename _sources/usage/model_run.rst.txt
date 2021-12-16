Re-run Model
============

If the data is already preprocessed and the features are selected with output files from :py:mod:`preprocessing` and
:py:mod:`feature_selection` already existing in minio you can simply build the surrogate model and generate predictions.


.. include:: predict.rst

Download results
----------------

The results from the prediction is available in minio in the ``output_path`` specified in prediction step.

.. note:: The json file created at the end of model run has the following keys:
    'test_daily_metric': This contains the DAILY evaluation result from using the TEST dataset
    'test_annual_metric': This contains the ANNUAL evaluation result from using the TEST dataset
    'output_df':This is a dataframe containining the datapoint_id, predicted energy value and the actual energy value from using the test dataset. Note the annual energy result is shown here.
    'val_daily_metric': This contains the DAILY evaluation result from using the VALIDATION dataset
    'val_annual_metric': This contains the ANNUAL evaluation result from using the VALIDATION dataset
    'output_val_df': This is a dataframe containining the datapoint_id, predicted energy value and the actual energy value from using the VALIDATION dataset. Note the annual energy result is shown here.

