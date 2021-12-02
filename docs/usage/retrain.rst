Train the model
===============

If there are new data files that significantly change the types of information the model knows how to predict you need
to retrain the model. This involves performing a series of steps to prepare the data and use it to retrain the model.
In the case that you want to retrain the model but are not dealing with significant changes in the data (for example,
using the same weather data) you can skip the weather data preparation or cleaning and splitting steps, as appropriate.

.. contents::
   :depth: 1
   :local:

Preparation of weather data
---------------------------

.. include:: ../data_prep/weather.rst


Clean and split the data
------------------------

.. include:: preprocessing.rst

Feature selection
-----------------

.. include:: feature_selection.rst


Build the surrogate model and predict energy use
------------------------------------------------

.. include:: predict.rst

Download results
----------------

The results from the prediction is available in minio in the ``output_path`` specified in prediction step.
