Feature selection
=================
In order to ensure the most relevant features are used in building the surrogage model, feature_selection is performed to search for the most optimal features from preprocessed data.

.. Note:: Feature selection is performed using the X_train and the y_train set.


Feature selection could be perfomed using either of the following estimator type listed below:
* Linear Regression
* XGBRegressor
* ElasticnetCV
* LassoCV

For the purpose of this project, XGBRegressor was often slow in selecting the features. Often times, ElasticnetCV and LassoCV selected same features. Although, LassoCV is used as the default estimator for feature selection, any of the other estimator type can be used by specifying the respective esimator type to the estimator_type parameter when performing feature selection.

Below are the input parameters required for Feature Selection
* in_obj_name: Minio locationa and name of data file to be read, ideally the output file generated from preprocessing i.e. preprocessing.out.
* estimator_type: The type of estimator to be used, default is lasso.
* output_path: The minio location and filename where the output file should be written.

Feautre selection is performed by running::

    python3 feature_selection.py --in_obj_name output_data/preprocessing_out --output_path output_data/feature_out --estimator_type lasso
