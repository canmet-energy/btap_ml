After preprocessing the data, the features to be used in building the surrogate model can be selected
(with all inputs coming from the command line or from the ``input_config.yml`` file)::

    python feature_selection.py input_config.yml

The parameters to the above script are documented at :py:mod:`feature_selection`.

To ensure that the most relevant features are used in building the surrogage model, feature selection is performed
to search for the most optimal features from the preprocessed data.

.. note::

    Feature selection is performed using the X_train and the y_train set.

To determine which features are useful for predicting the total energy and total costing we use:

* MultiTaskLassoCV

Feature selection can be perfomed using any of the following estimator types when predicting the total energy or total costing:

* Linear Regression
* XGBRegressor
* ElasticnetCV
* LassoCV

Some details on the options are:

* XGBRegressor and ElasticnetCV are often slow in selecting the features.
* ElasticnetCV and LassoCV frequently selected same features.
* Although, LassoCV is used as the default estimator for feature selection, any of the other estimator type can be used by specifying the respective esimator type to the estimator_type parameter when performing feature selection.

.. note::

    The json file that is created contains the key value "features" which represents the final features selected for modelling.
