Re-run Model
============

If the data is already preprocessed and the features are selected with output files from :py:mod:`preprocessing` and
:py:mod:`feature_selection` already existing in minio you can simply build the surrogate model and generate predictions.


.. include:: predict.rst

Download results
----------------

The results from the prediction is available in minio in the ``output_path`` specified in prediction step.
