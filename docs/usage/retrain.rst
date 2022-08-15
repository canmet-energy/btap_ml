Train the model
===============

If there are new data files that significantly change the types of information the model knows how to predict you need to retrain the model.
In the case that you want to retrain the model but are not dealing with significant changes in the data cleaning and splitting steps, as needed.
The steps of training a model are called inside of ``train_model_pipeline.py``, but each of the individual steps performed
are also discussed below.

.. contents::
   :depth: 1
   :local:

Training a model
----------------

.. include:: train_model_pipeline.rst

Preparation of weather data
---------------------------

.. include:: weather.rst


Clean and split the data
------------------------

.. include:: preprocessing.rst

Feature selection
-----------------

.. include:: feature_selection.rst


Build the surrogate model and predict energy use
------------------------------------------------

.. include:: predict.rst
