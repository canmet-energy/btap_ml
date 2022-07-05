Training a model can be done by calling the ``train_model_pipeline.py`` file. This file combines each
preprocessing step to allow the entire training process to be invoked from one call. The process
begins by creating a unique output directory based on the date and time which the process begins.
All outputs will be located within the new directory.

Using the ``input_config.yml`` file or the passed command line arguements (which can be viewed by
passing the ``--help`` arguement), each required input will be validated before being used within the
program.

The first step performed is the processing of weather data via the ``prepare_weather.py`` file.

Next, the data will be passed to ``preprocessing.py`` to load the building data and the energy data,
and merge the two with the weather data. Following additional data cleaning, the data is split into
train/test/validation sets, where the validation set is either split from the regular input data or
is loaded from an explicitly provided file to be used for validation.

Before training the Machine Learning model, the optimal features will be selected by calling
``feature_selection.py``. This process will use one of several avilable tools for feature selection
to derive a list of features which will be used for training. This omits any feature which will not be
useful in the training process.

Using the selected features and preprocessed datasets, ``predict.py`` will be called to train the model.
Hyperparameter tuning can be performed at the cost of additional time complexity or the default
model architecture can be used. The trained model and test results will be output for future use.
