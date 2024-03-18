The outputs from the preprocessing and feature selection steps are used as input to ``predict.py``
to derive a trained model for predicting the energy and costing::

    python predict.py input_config.yml

This script will output the trained model as a .h5 file for a Multilayer Perceptron model and .joblib for a Random Forest Model alongisde
information on the training and testing performed for analysis. The outputs from this step will be used when obtaining predictions with the model.

The parameters to the above script are documented at: :py:mod:`predict`.

The Machine Learning model which is used for training will vary depending on whether parameter tuning is or is not
performed (total outputs only) and depending on whether the user selects the more complex or simplistic model.
When hyperparameter tuning is used (total outputs only), the `Keras Hyperband Tuner <https://keras.io/api/keras_tuner/tuners/hyperband/>`_ is configured
to optimize the parameters based on the loss. The tuner also uses early stopping by monitoring the loss with a
patience of 5. Once the tuner finishes, the optimized model is built and used for training.

If hyperparameter tuning is not performed, a default model which has been evaluated as a baseline for performance is used.
There are two options for the default model, one with more hidden nodes and a lower learning rate and another with less hidden nodes and a larger learning rate.

* Larger model: This model uses a single layer of 10000 nodes with the ReLU activation function. An optional 10% dropout layer is then applied. The adam optimizer is used with a learning rate of 0.0001.
* Smaller model: This model uses a single layer of 56 nodes with the ReLU activation function. An optional 10% dropout layer is then applied. The adam optimizer is used with a learning rate of 0.001.

A customized RMSE loss function is defined and used for training which takes a sum of the labels and predictions before computing the RMSE loss.

The outputs will include breakdowns on the model's overall performance and the model's performance for each building type and climate zone.
When working with breakdowns of the energy and costing values, these breakdowns will be done for all types of costing and energy.
