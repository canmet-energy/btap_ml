Training a model can be done by calling the ``train_model_pipeline.py`` file. This file combines each
preprocessing step to allow the entire training process to be invoked from one call. The process
begins by creating a unique output directory based on the date and time which the process begins.
All outputs will be located within the new directory.

Using the ``input_config.yml`` file or the passed command line arguements (which can be viewed by
passing the ``--help`` arguement), each required input will be validated before being used within the
program.

The data will first be loaded by ``preprocessing.py``, which load the building data and the energy data,
and merges the two with the weather data (for energy training). For costing, no weather or energy data is loaded.
Following additional data cleaning, the data is split into train/test/validation sets,
where the validation set is either split from the regular input data or
is loaded from an explicitly provided file to be used for validation.

.. note::

    All input data will be used 'as is', where transformations using specific calculations should be done
    before the data is used for processing.

Before training the Machine Learning model, the optimal features will be selected by calling
``feature_selection.py``. This process will use one of several avilable tools for feature selection
to derive a list of features which will be used for training. This omits any feature which will not be
useful in the training process. The availability of each feature selection tool will vary depending on
whether the total energy/costing is predicted or whether the energy/costing breakdowns are predicted.

There are options from selecting between two models: Either a Multilayer Perceptron (MLP) or a Random Forest (RF)
model. User can specify the type of being model that needs to be trained by sepcifying 'mlp' for MultiLayer Perceptron
or 'rf' for Random Forest Model in the selected_model_type parameter in input_config.yml.

Using the selected features and preprocessed datasets, ``predict.py`` will be called to train the model.
Hyperparameter tuning can be performed at the cost of additional time complexity or the default
model architecture can be used. The trained model and test results will be output for future use.
In the output ``.json file``, there will be breakdowns on the model's performance for each building type
and climate zone
