Model building
==============
In order to build the surrogate model, output generated during preprocessing and feature selection are used as inputs. The surrogate model is developed using neural network with Multi-layer perceptron (MLP) in Keras.

Hyperparameter Search
----------------------
There are two options when building the model: (1) with hyperparameter search, and (2) without hyperparameter search.
Hyperparameter search is controlled by the ``--param_search`` switch on.

.. Note:: The time taken for the hyperparameter search to be completed will be dependent on the number of parameters searched for , the number of options provided for each parameter search and the size of the dataset. Where possible, its advisable to keep the search parameter and options concise.

The following are the various types of hyperparameters searched for and their values:

* activation: The options are 'relu','tanh', or 'sigmoid'. Other activation functions for neural nets can also be applied, the most optimal activation function for the surrogate model are currently considered.
* regularizers: The options are 1e-1, 1e-3, 1e-4, 1e-5. However to control the hyparameter search time, the current options explored are 1e-4, and 1e-5
* num of layers: This parameter determines how DEEP  the neural network would be. Any numerical value can be specified here as option, however, from several iterations, it is observed that a depth of 1 provides the optimal result.
* units: This is the number of units in each layer of the network. Numerical value can be provided here. From results from various iterations, a range of 8 to 100 at the most should be explored.
* dropout_rate: The options specified here should be less than 1, below are the current options considered 0.1, 0.2, 0.3. More values could be included if necessary.
* optimizer: The options are rmsprop', 'adam', 'sgd'. Other optimizers for neural nets can also be considered, the most optimal optimisers for the surrogate model are currently considered.

As a rule of thumb, before opting to choose the option of setting the param_search to no, it is advisable to have run an hyperparameter search at least once. Once the hyperparameter search is completed, the best values will be passed to build the surrogate model. When the model is run without the hyperparameter search, the best parameter known from the hyperparameter search can be inputted for each respective field while calling the create_model function.

.. Note:: The best parameter values will usually be displayed in the terminal when hyperparameter search is completed, otherwise, the search results generated can also be viewed from ../output/parameter_search/btap/


Specifically, the following are the input parameters required at to run the predict.py:

* in_obj_name: minio locationa and name of data file to be read, ideally the output file generated from preprocessing i.e. preprocessing_out.
* features: minio locationa and name of data file to be read, ideally the output file generated from feature selection i.e. feature_out.
* param_search: This parameter is used to determine if hyperparameter search can be performed or not, accepted value is yes or no.
* output_path': The minio location and filename where the output file should be written.

Without hyperparameter search::

   python3 predict.py --param_search no --in_obj_name output_data/preprocessing_out --features output_data/feature_out --output_path output_data/predict_out

With hyperparameter search::

   python3 predict.py --param_search yes --in_obj_name output_data/preprocessing_out --features output_data/feature_out --output_path output_data/predict_out


Evaluation
-----------
The model is trained using the training set, and the model is evaluated using the test set and the validation set generated during preprocessing. The output from the model predicts the daily energy consumption, which can be aggregated for each datapoint_id to derive the annual energy consumed. Since the daily energy consumed is in Mega Joules per meter square (mj_per_m_sq), it is converted to Giga Joules per meter square (gj_per_m_sq) which is the unit the target variable (net_site_eui_gj_per_m_sq) is represented.

The model is first evaluated by comparing the predicted daily energy with actual daily energy which was aggregated from the hourly energy file for each datapoint_id. Next, the predicted annaul energy and the actual energy (net_site_eui_gj_per_m_sq) is evaluated. The following metrics are used in evaluating the model:

* Mean Absolute Error (mae)
* Mean Squared Error (mse)
* Root Mean Squared Error (mape)

The dictionary returned as final output from the model consist of the following key values:

* test_daily_metric: The metric for the daily energy predicted evaluation result from the test set.
* test_annual_metric: The metric for the annual energy predicted evaluation result from the test set.
* output_df: A dataframe containing the datapoint_id, predicted value and total energy values for the test set.
* val_daily_metric: The metric for the daily energy predicted evaluation result from the validation set.
* val_annual_metric: The metric for the annual energy predicted evaluation result from the validation set.
* output_val_df: A dataframe object containing the datapoint_id, predicted value and total energy values for the validation set.


Tensorboard
-----------

You can use the tensorboard dashboard to inspect the performance of the model.

1. Open ``notebooks/tensorboard.ipynb``.
2. Run all the contents of the notebook.
3. Navigate to the appropriate port in a web browser.

.. note::

   The tensorboard opens on a random port inside your notebook container. The URL looks like
   https://kubeflow.aaw.cloud.statcan.ca/notebook/nrcan-btap/<notebook_name>/proxy/<port>/.
