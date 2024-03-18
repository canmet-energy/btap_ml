Running a model
===============

A trained model and its related outputs from :py:mod:`train_model_pipeline` can be used to obtain
energy and costing predictions for input building and weather data over a specified period of time (typically
from January 1 to December 31 with no added day for a leap year). Note that costing is not daily and thus the
timespan and weather information is not applicable for the costing process.

Predictions can be obtained by running the following command, with all needed inputs being
input through the command line or through the provided ``input_config.yml`` file::

    python run_model.py input_config.py

The file begins by initializing the preprocessing step which will process all input files
and identify/load the weather data from present climate zones. An input directory of .xlsx
building output files are passed to ``preprocessing.py`` to be preprocessed into a single
dataset which will then be passed through the trained model to obtain the predictions.

The dataset will be adjusted with the selected features .json file which is generated
when training the specified .h5 trained model in the case of Multilyaer Perceptron or .joblib
for the Random Forest model.
The input data is also scaled with the same scalers used when training. This ensures that all input
data is of the same format as what has been used for the training.

The predictions are then made by the model, with the output values placed within three csv files,
which link the predictions to the specific buildings within the specific building files.
The ``daily_energy_predictions.csv`` file outputs the predicted daily energy for a building on
each day in Megajoules per meter squared. The ``aggregated_energy_predictions.csv`` file
contains the aggregated daily energy outputs in Gigajoules per meter squared for each building within
the specified start and end dates. These typically are 365 days to represent a full year.
The ``costing_predictions.csv`` file outputs the costing predictions alongside the corresponding rows
from the input files.
