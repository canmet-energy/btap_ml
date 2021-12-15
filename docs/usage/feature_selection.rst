After preprocessing the data, the features to be used in building the surrogate model can be selected::

    $ python3 feature_selection.py --in_obj_name output_data/preprocessing_out --output_path output_data/feature_out --estimator_type lasso

The parameters to the above script are documented in the :py:mod:`feature_selection`.

In order to ensure the most relevant features are used in building the surrogage model, feature_selection is performed
to search for the most optimal features from preprocessed data.

.. note::

    Feature selection is performed using the X_train and the y_train set.

Feature selection could be perfomed using either of the following estimator type listed below:

* Linear Regression
* XGBRegressor
* ElasticnetCV
* LassoCV

For the purpose of this project, XGBRegressor was often slow in selecting the features. Often times, ElasticnetCV and
LassoCV selected same features. Although, LassoCV is used as the default estimator for feature selection, any of the
other estimator type can be used by specifying the respective esimator type to the estimator_type parameter when
performing feature selection.
