The output from the preprocessing and feature selection outputs are used as input to ``predict.py``
to derive a trained model::

    python predict.py input_config.yml

This script will output the trained model as a .h5 file alongisde information on the training and testing
performed for analysis. The outputs from this step will be used when obtaining predictions with the model.

The parameters to the above script are documented at: :py:mod:`predict`.

The Machine Learning model which is used for training will vary depending on whether parameter tuning is or is not
performed. When used, the [Keras Hyperband Tuner](https://keras.io/api/keras_tuner/tuners/hyperband/) is configured
to optimize the parameters based on the loss. The tuner also uses early stopping by monitoring the loss with a
patience of 5. Once the tuner finishes, the optimized model is built and used for training.

If hyperparameter tuning is not performed, the default model which has been evaluated as a baseline for performance is used. This model uses a dense layer of 56 nodes with the ReLU activation function and a dropout layer of 10% applied afterwards. The adam optimizer is used with a learning rate of 0.001.

A customized RMSE loss function is defined and used for training which takes a sum of the labels and predictions before computing the RMSE loss.
